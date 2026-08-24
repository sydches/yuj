"""Engine-enforced planning phase for coding sessions.

Plan mode is deliberately mechanical.  It classifies the requested tool and
its arguments, never model prose or tool output.  While active, only bounded
inspection, a fail-closed read-only shell subset, the one plan-file write, and
the explicit exit tool are admitted.
"""
from __future__ import annotations

import html
import re
import shlex
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .command_redirect import split_shell_fragments


PLAN_FILE = ".solver/plan.md"
PLAN_MODES = ("off", "required")
_INSPECTION_TOOLS = frozenset({"read", "glob", "grep", "list_definitions"})
_PLAN_TOOL_SURFACE = frozenset({
    *_INSPECTION_TOOLS, "bash", "write", "exit_plan_mode",
})
_SIMPLE_READ_ONLY_COMMANDS = frozenset({
    "basename", "cat", "cksum", "cmp", "cut", "date", "df", "diff", "dirname",
    "du", "echo", "egrep", "false", "fd", "fgrep", "file", "find", "grep",
    "head", "hostname", "id", "jq", "less", "locate", "ls", "md5sum", "more",
    "printf", "pwd", "readlink", "realpath", "rg", "sed", "sha1sum",
    "sha256sum", "sort", "stat", "tail", "test", "true", "type", "uname",
    "uniq", "wc", "which", "whoami",
})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)


@dataclass(frozen=True)
class PlanModeDecision:
    """One pre-dispatch plan-mode decision."""

    allowed: bool
    message: str = ""


def plan_mode_required(cfg) -> bool:
    """Return whether the effective config requires an explicit plan."""
    return str(getattr(cfg, "plan_mode", "off") or "off") == "required"


def filter_plan_mode_schemas(tool_schemas: list[dict], *, active: bool) -> list[dict]:
    """Return the model-facing phase allowlist without changing base schemas."""
    if not active:
        return tool_schemas
    return [
        schema for schema in tool_schemas
        if schema.get("function", {}).get("name") in _PLAN_TOOL_SURFACE
    ]


def effective_model_tool_schemas(session):
    """Read the dynamic surface with compatibility for small test doubles."""
    try:
        return session.model_tool_schemas
    except AttributeError:
        return getattr(session, "_tool_schemas", None)


def has_plan_mode_transition(events: Iterable[Mapping[str, object]]) -> bool:
    return any(
        event.get("event") in {"plan_mode_enter", "plan_mode_exit"}
        for event in events
    )


def _active_from_trace(events: Iterable[Mapping[str, object]]) -> bool:
    active = True
    for event in events:
        if event.get("event") == "plan_mode_enter":
            active = True
        elif event.get("event") == "plan_mode_exit":
            active = False
    return active


def plan_mode_active_from_trace(
    events: Iterable[Mapping[str, object]],
) -> bool:
    """Project the latest task-level plan phase from raw transitions."""
    return _active_from_trace(events)


def _completed_plan_turns(events: Iterable[Mapping[str, object]]) -> int:
    """Count model turns after the most recent enter and before an exit."""
    count = 0
    active = False
    for event in events:
        event_type = event.get("event")
        if event_type == "plan_mode_enter":
            active = True
            count = 0
        elif event_type == "plan_mode_exit":
            active = False
        elif event_type == "turn" and active:
            count += 1
    return count


def _plan_path(cwd: str | Path) -> Path:
    return Path(cwd).resolve() / PLAN_FILE


def is_exact_plan_path(cwd: str | Path, value: object) -> bool:
    """Accept only the lexical plan path inside the task working directory.

    Resolving the candidate but not the expected path also rejects a symlinked
    ``.solver`` directory or plan file that would escape the task repository.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    try:
        return candidate.resolve() == _plan_path(cwd)
    except OSError:
        return False


def is_plan_write(tool_name: str, arguments: Mapping[str, object], cwd: str) -> bool:
    return tool_name == "write" and is_exact_plan_path(cwd, arguments.get("path"))


def _git_is_read_only(argv: list[str]) -> bool:
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in {"-C", "--git-dir", "--work-tree", "--namespace"}:
            index += 2
            continue
        if token.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
            index += 1
            continue
        if token in {"--no-pager", "--paginate", "--literal-pathspecs"}:
            index += 1
            continue
        break
    if index >= len(argv):
        return False
    subcommand = argv[index]
    args = argv[index + 1:]
    if any(
        arg == "--output" or arg.startswith("--output=")
        or arg.startswith("--open-files-in-pager")
        or arg in {"--ext-diff", "--textconv"}
        for arg in args
    ):
        return False
    if subcommand in {
        "blame", "cat-file", "describe", "diff", "grep", "log", "ls-files",
        "ls-tree", "merge-base", "rev-parse", "show", "status",
    }:
        return True
    if subcommand == "branch":
        mutating = {
            "-c", "-C", "-d", "-D", "-m", "-M", "--copy", "--delete",
            "--edit-description", "--move", "--set-upstream-to",
            "--unset-upstream", "--create-reflog",
        }
        return all(arg.startswith("-") for arg in args) and not any(
            arg in mutating or any(arg.startswith(f"{flag}=") for flag in mutating)
            for arg in args
        )
    if subcommand == "config":
        return bool(args) and args[0] in {"--get", "--get-all", "--get-regexp", "--list"}
    if subcommand == "remote":
        return not args or args[0] in {"-v", "get-url", "show"}
    if subcommand == "tag":
        return not args or args[0] in {"-l", "--list"}
    if subcommand == "worktree":
        return bool(args) and args[0] == "list"
    return False


def _has_unquoted_expansion(fragment: str) -> bool:
    """Reject argument expansion that can turn safe text into mutating flags."""
    quote = ""
    escaped = False
    for char in fragment:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            elif quote == '"' and char in {"$", "`"}:
                return True
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in {"$", "`", "*", "?", "[", "{"}:
            return True
    return False


def _fragment_is_read_only(fragment: str) -> bool:
    # Fail closed on shell features that can hide execution or file writes.
    if any(token in fragment for token in (">", "<")):
        return False
    if _has_unquoted_expansion(fragment):
        return False
    try:
        argv = shlex.split(fragment, posix=True)
    except ValueError:
        return False
    if not argv:
        return False
    # Prefix assignments can activate command helpers (for example Git's
    # external diff) and a path can name an arbitrary lookalike executable.
    if _ASSIGNMENT_RE.fullmatch(argv[0]) or "/" in argv[0]:
        return False
    verb = argv[0].rsplit("/", 1)[-1]
    if verb == "git":
        return _git_is_read_only(argv)
    if verb not in _SIMPLE_READ_ONLY_COMMANDS:
        return False
    args = argv[1:]
    if any(
        arg == "--output" or arg.startswith("--output=") for arg in args
    ):
        return False
    if verb == "sort" and any(
        arg == "-o" or (arg.startswith("-o") and len(arg) > 2) for arg in args
    ):
        return False
    if verb == "printf" and "-v" in args:
        return False
    if verb == "date" and any(
        arg in {"-s", "--set"} or arg.startswith(("--set=", "--file="))
        for arg in args
    ):
        return False
    if verb == "hostname" and any(
        arg not in {
            "-d", "--domain", "-f", "--fqdn", "-i", "--ip-address",
            "-I", "--all-ip-addresses", "-s", "--short",
        }
        for arg in args
    ):
        return False
    if verb == "sed":
        if not any(arg in {"-n", "--quiet", "--silent"} for arg in args):
            return False
        if any(
            arg in {"-e", "--expression", "-f", "--file", "-i", "--in-place"}
            or arg.startswith(("--expression=", "--file=", "--in-place="))
            or (arg.startswith("-i") and len(arg) > 2)
            for arg in args
        ):
            return False
        scripts = [arg for arg in args if not arg.startswith("-")]
        if not scripts or re.fullmatch(
            r"(?:\d+|\$)(?:,(?:\d+|\$))?p", scripts[0]
        ) is None:
            return False
    if verb == "find" and any(
        arg in {
            "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls",
            "-fprint", "-fprint0", "-fprintf",
        }
        for arg in args
    ):
        return False
    return True


def bash_is_read_only(arguments: Mapping[str, object]) -> bool:
    """Classify one bash tool call using a conservative positive allowlist."""
    if bool(arguments.get("background", False)):
        return False
    command = arguments.get("cmd")
    if not isinstance(command, str) or not command.strip():
        return False
    fragments = split_shell_fragments(command)
    if not fragments or any(fragment.operator_after == "&" for fragment in fragments):
        return False
    return all(_fragment_is_read_only(fragment.text) for fragment in fragments)


def render_plan_mode_error(tool_name: str, message: str, max_chars: int) -> str:
    body = html.escape(message, quote=False)
    opening = (
        f'<tool_result tool_name="{html.escape(tool_name, quote=True)}" '
        'status="error" error_kind="plan_mode" v="1">'
    )
    closing = "</tool_result>"
    cap = max(0, int(max_chars))
    available = cap - len(opening) - len(closing) - 2
    if cap and available < len(body):
        body = body[:max(0, available - 3)] + ("..." if available >= 3 else "")
    return f"{opening}\n{body}\n{closing}"


class PlanModeController:
    """Session-local phase state rehydrated only from raw trace transitions."""

    def __init__(
        self,
        *,
        cwd: str,
        cfg,
        events: Iterable[Mapping[str, object]],
        event_sink: Callable[[dict[str, object]], None],
    ) -> None:
        event_list = tuple(events)
        self.cwd = cwd
        self.cfg = cfg
        self.required = plan_mode_required(cfg)
        self.active = self.required and _active_from_trace(event_list)
        self.prior_turns = _completed_plan_turns(event_list)
        self._event_sink = event_sink

    def is_plan_write(self, tool_name: str, arguments: Mapping[str, object]) -> bool:
        return is_plan_write(tool_name, arguments, self.cwd)

    def check(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        *,
        turn: int,
        active: bool | None = None,
    ) -> PlanModeDecision:
        phase_active = self.active if active is None else active
        if not phase_active:
            return PlanModeDecision(True)
        allowed = False
        if tool_name in _INSPECTION_TOOLS or tool_name == "exit_plan_mode":
            allowed = True
        elif tool_name == "bash":
            allowed = bash_is_read_only(arguments)
        elif self.is_plan_write(tool_name, arguments):
            allowed = True

        if not allowed:
            return PlanModeDecision(
                False,
                f"Plan mode is active; {tool_name} was not executed. Use read, glob, "
                "grep, list_definitions, a classified read-only bash command, write "
                f"only {PLAN_FILE}, or exit_plan_mode after writing the plan.",
            )

        max_turns = int(getattr(self.cfg, "plan_mode_max_turns", 0) or 0)
        ordinal = self.prior_turns + int(turn) + 1
        if (
            max_turns > 0
            and ordinal > max_turns
            and tool_name != "exit_plan_mode"
            and not self.is_plan_write(tool_name, arguments)
        ):
            return PlanModeDecision(
                False,
                f"Plan mode reached its {max_turns}-turn limit. Write {PLAN_FILE} "
                "and call exit_plan_mode; this tool was not executed.",
            )
        return PlanModeDecision(True)

    def exit(self, *, turn: int) -> str:
        max_chars = int(getattr(self.cfg, "max_output_chars", 20000))
        if not self.active:
            return render_plan_mode_error(
                "exit_plan_mode", "Plan mode is not active.", max_chars
            )
        path = _plan_path(self.cwd)
        if not is_exact_plan_path(self.cwd, PLAN_FILE):
            return render_plan_mode_error(
                "exit_plan_mode",
                f"Cannot exit plan mode: {PLAN_FILE} is not a regular task path.",
                max_chars,
            )
        try:
            plan = path.read_text()
        except FileNotFoundError:
            return render_plan_mode_error(
                "exit_plan_mode",
                f"Cannot exit plan mode: {PLAN_FILE} does not exist.",
                max_chars,
            )
        except (OSError, UnicodeError):
            return render_plan_mode_error(
                "exit_plan_mode",
                f"Cannot exit plan mode: {PLAN_FILE} is not a readable text file.",
                max_chars,
            )
        if not plan.strip():
            return render_plan_mode_error(
                "exit_plan_mode",
                f"Cannot exit plan mode: {PLAN_FILE} is empty.",
                max_chars,
            )
        self._event_sink({
            "event": "plan_mode_exit",
            "turn": int(turn),
            "turn_number": int(turn),
            "plan_chars": len(plan),
        })
        self.active = False
        return (
            '<tool_result tool_name="exit_plan_mode" status="ok" v="1">\n'
            "Plan accepted; implementation tools are now unlocked.\n"
            "</tool_result>"
        )


__all__ = [
    "PLAN_FILE", "PLAN_MODES", "PlanModeController", "PlanModeDecision",
    "bash_is_read_only", "effective_model_tool_schemas",
    "filter_plan_mode_schemas", "has_plan_mode_transition", "is_exact_plan_path",
    "is_plan_write", "plan_mode_active_from_trace", "plan_mode_required",
    "render_plan_mode_error",
]
