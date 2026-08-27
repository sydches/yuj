"""Named sequential subagents for the model-facing ``task`` tool.

Agent descriptors own model/profile and tool policy.  This module owns the
runtime boundary: deterministic IDs, isolated child traces, replay from those
traces, read-only shell admission, and child token accounting.  A child uses
the same task cwd and sandbox policy as its parent but a separate in-memory
conversation and model client.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .._shared.paths import project_root
from .._shared.toml_compat import load_toml
from .solver import assemble_system_prompt, resolve_system_prompt_source
from .tool_specs import ACTIVE_TOOL_NAMES


_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_TASK_ID_RE = re.compile(r"^task-(\d{6})$")
_READ_ONLY_FORBIDDEN_TOOLS = frozenset({
    "write",
    "edit",
    "notebook_edit",
    "apply_patch",
    "udiff",
    "exec_cell",
    "run_tests",
    "bash_poll",
    "bash_kill",
    "terminal_start",
    "terminal_io",
    "task",
})
_READ_ONLY_SIMPLE_COMMANDS = frozenset({
    "cat",
    "grep",
    "head",
    "ls",
    "pwd",
    "stat",
    "tail",
    "wc",
})
_SHELL_CONTROL_TOKENS = ("\n", ";", "&&", "||", "|", ">", "<", "`", "$(")


class AgentConfigError(ValueError):
    """A named agent descriptor is invalid or unsafe."""


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    model_profile: str
    tools: tuple[str, ...]
    system_prompt_file: Path
    max_turns: int
    read_only: bool = True


@dataclass(frozen=True, slots=True)
class SubagentModelBinding:
    """Fresh child transport plus its profile-adjusted Config."""

    client: Any
    config: Any
    resolution: Any = None
    dedicated: bool = True


@dataclass(frozen=True, slots=True)
class SubagentOutcome:
    task_id: str
    agent: str
    result: str
    turns: int
    prompt_tokens: int
    completion_tokens: int
    own_prompt_tokens: int
    own_completion_tokens: int
    finish_reason: str
    done: bool

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _agent_name(value: object) -> str:
    if not isinstance(value, str) or not _AGENT_NAME_RE.fullmatch(value):
        raise AgentConfigError(
            "agent must start with a letter, digit, or underscore and use "
            "only letters, digits, '.', '_' or '-'"
        )
    return value


def load_agent_spec(name: str, agents_dir: Path | None = None) -> AgentSpec:
    """Load and fail-closed validate ``agents/<name>.toml``."""
    normalized = _agent_name(name)
    root = Path(agents_dir or (project_root() / "agents")).resolve()
    descriptor = root / f"{normalized}.toml"
    if not descriptor.is_file():
        raise AgentConfigError(f"unknown subagent {normalized!r}: {descriptor} not found")
    data = load_toml(descriptor)
    if set(data) != {"agent"} or not isinstance(data.get("agent"), dict):
        raise AgentConfigError(
            f"{descriptor} must contain exactly one [agent] table"
        )
    table = data["agent"]
    allowed_fields = {
        "model_profile", "tools", "system_prompt_file", "max_turns", "read_only"
    }
    unknown = set(table) - allowed_fields
    if unknown:
        raise AgentConfigError(
            f"{descriptor} has unknown agent fields: {sorted(unknown)}"
        )

    model_profile = table.get("model_profile")
    if not isinstance(model_profile, str) or not _AGENT_NAME_RE.fullmatch(model_profile):
        raise AgentConfigError(
            f"{descriptor}: agent.model_profile must be a profile name"
        )
    tools_value = table.get("tools")
    if not isinstance(tools_value, list) or not tools_value:
        raise AgentConfigError(
            f"{descriptor}: agent.tools must be a non-empty string list"
        )
    if any(not isinstance(item, str) for item in tools_value):
        raise AgentConfigError(
            f"{descriptor}: agent.tools must contain only strings"
        )
    tools = tuple(tools_value)
    if len(set(tools)) != len(tools):
        raise AgentConfigError(f"{descriptor}: agent.tools contains duplicates")
    unknown_tools = set(tools) - set(ACTIVE_TOOL_NAMES)
    if unknown_tools:
        raise AgentConfigError(
            f"{descriptor}: agent.tools has unknown tools: {sorted(unknown_tools)}"
        )

    read_only = table.get("read_only", True)
    if not isinstance(read_only, bool):
        raise AgentConfigError(f"{descriptor}: agent.read_only must be a boolean")
    unsafe = set(tools) & _READ_ONLY_FORBIDDEN_TOOLS if read_only else set()
    if unsafe:
        raise AgentConfigError(
            f"{descriptor}: read-only agent cannot allow tools: {sorted(unsafe)}"
        )

    max_turns = table.get("max_turns")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise AgentConfigError(
            f"{descriptor}: agent.max_turns must be an integer >= 1"
        )
    prompt_value = table.get("system_prompt_file")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise AgentConfigError(
            f"{descriptor}: agent.system_prompt_file must be a non-empty path"
        )
    prompt_path = (descriptor.parent / prompt_value).resolve()
    try:
        prompt_path.relative_to(root)
    except ValueError as exc:
        raise AgentConfigError(
            f"{descriptor}: agent.system_prompt_file must stay under {root}"
        ) from exc
    if prompt_path.suffix.lower() != ".md" or not prompt_path.is_file():
        raise AgentConfigError(
            f"{descriptor}: agent.system_prompt_file must name an existing .md file"
        )
    return AgentSpec(
        name=normalized,
        model_profile=model_profile,
        tools=tools,
        system_prompt_file=prompt_path,
        max_turns=max_turns,
        read_only=read_only,
    )


def prepare_readonly_bash(command: str) -> tuple[str | None, str]:
    """Validate and render a shell-safe inspection command.

    This is intentionally an allowlist.  A read-only subagent still runs in
    the normal sandbox, whose task cwd is writable for the parent, so an
    unknown command must be rejected rather than guessed non-mutating.
    """
    if not isinstance(command, str) or not command.strip():
        return None, "empty command"
    if any(token in command for token in _SHELL_CONTROL_TOKENS):
        return None, "shell control, substitution, or redirection is not read-only"
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return None, "command is not valid shell syntax"
    if not parts:
        return None, "empty command"
    if "/" in parts[0]:
        return None, "command paths are not allowed in read-only bash"
    verb = parts[0]
    if verb not in _READ_ONLY_SIMPLE_COMMANDS:
        return None, f"command {verb!r} is not in the read-only allowlist"
    # Quote every model-provided token and resolve the executable through the
    # shell's implementation-defined safe PATH. This prevents a writable task
    # directory or configured PATH from substituting a mutating executable.
    return "command -p " + shlex.join(parts), ""


def readonly_bash_decision(command: str) -> tuple[bool, str]:
    """Return whether ``command`` passes the read-only shell policy."""
    prepared, reason = prepare_readonly_bash(command)
    return prepared is not None, reason


def _terminal_event(trace_path: Path, task_id: str, agent: str) -> dict[str, Any]:
    terminal: dict[str, Any] | None = None
    for raw in trace_path.read_text().splitlines():
        if not raw.strip():
            continue
        event = json.loads(raw)
        if event.get("event") == "subagent_result":
            terminal = event
    if terminal is None:
        raise RuntimeError(f"subagent trace {trace_path} has no subagent_result")
    if terminal.get("id") != task_id or terminal.get("agent") != agent:
        raise RuntimeError(f"subagent trace identity mismatch for {task_id}")
    result = terminal.get("result")
    if not isinstance(result, str):
        raise RuntimeError(f"subagent trace {trace_path} has no string result")
    if int(terminal.get("result_chars", -1)) != len(result):
        raise RuntimeError(f"subagent trace {trace_path} result length mismatch")
    digest = hashlib.sha256(result.encode("utf-8")).hexdigest()
    if terminal.get("result_sha256") != digest:
        raise RuntimeError(f"subagent trace {trace_path} result hash mismatch")
    return terminal


class SubagentRuntime:
    """Run-wide task allocator, replay stream, and child metrics owner."""

    def __init__(
        self,
        run_root: Path,
        *,
        agents_dir: Path | None = None,
        client_factory: Callable[[Any, AgentSpec, str], SubagentModelBinding] | None = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.agents_dir = Path(agents_dir or (project_root() / "agents"))
        self.client_factory = client_factory
        self._lock = threading.RLock()
        self._next_id = self._discover_next_id()
        self._replay_index = 0
        self._calls = 0
        self._own_prompt_tokens = 0
        self._own_completion_tokens = 0

    def _discover_next_id(self) -> int:
        root = self.run_root / "subagents"
        highest = 0
        if root.is_dir():
            for child in root.iterdir():
                match = _TASK_ID_RE.fullmatch(child.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1

    def _allocate_id(self) -> str:
        with self._lock:
            task_id = f"task-{self._next_id:06d}"
            self._next_id += 1
            return task_id

    def _record_own_usage(self, outcome: SubagentOutcome) -> None:
        with self._lock:
            self._calls += 1
            self._own_prompt_tokens += outcome.own_prompt_tokens
            self._own_completion_tokens += outcome.own_completion_tokens

    def metrics_payload(self) -> dict[str, int]:
        with self._lock:
            prompt = self._own_prompt_tokens
            completion = self._own_completion_tokens
            return {
                "calls": self._calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }

    def execute(self, parent: Any, agent: object, prompt: object) -> str:
        """Run or replay one child and emit the parent's summary event."""
        if not getattr(parent.cfg, "tools_task_enabled", False):
            return "ERROR: task tool is disabled by tools.task_enabled"
        try:
            agent_name = _agent_name(agent)
        except AgentConfigError as exc:
            return f"ERROR: {exc}"
        if not isinstance(prompt, str) or not prompt.strip():
            return "ERROR: task prompt must be a non-empty string"
        depth = int(getattr(parent, "_subagent_level", 0) or 0)
        cap = int(getattr(parent.cfg, "tools_subagent_depth", 1) or 0)
        if depth >= cap:
            return f"ERROR: subagent depth cap {cap} reached"

        replaying = bool(getattr(parent.client, "is_replay", False))
        if replaying:
            outcome = self._replay(parent, agent_name)
        else:
            try:
                spec = load_agent_spec(agent_name, self.agents_dir)
            except AgentConfigError as exc:
                return f"ERROR: {exc}"
            outcome = self._run_live(parent, spec, prompt)

        self._record_own_usage(outcome)
        if replaying:
            ledger = getattr(parent, "_role_token_ledger", None)
            if ledger is not None:
                ledger.record(
                    f"subagent.{outcome.agent}",
                    prompt_tokens=outcome.own_prompt_tokens,
                    completion_tokens=outcome.own_completion_tokens,
                )
        parent._subagent_prompt_tokens += outcome.prompt_tokens
        parent._subagent_completion_tokens += outcome.completion_tokens
        parent._subagent_calls += 1
        parent._emit(
            "subagent",
            session_number=parent._session_number,
            turn_number=parent._current_turn,
            id=outcome.task_id,
            agent=outcome.agent,
            turns=outcome.turns,
            tokens=outcome.tokens,
            result_chars=len(outcome.result),
        )
        return outcome.result

    def _binding(self, parent: Any, spec: AgentSpec, task_id: str) -> SubagentModelBinding:
        factory = self.client_factory or getattr(
            parent.client, "_subagent_client_factory", None
        )
        if callable(factory):
            binding = factory(parent, spec, task_id)
            if not isinstance(binding, SubagentModelBinding):
                raise TypeError("subagent client factory returned an invalid binding")
            return binding
        raise RuntimeError(
            "task tool requires a configured subagent model client factory"
        )

    def _run_live(self, parent: Any, spec: AgentSpec, prompt: str) -> SubagentOutcome:
        from ._loop.profile_resolution import _apply_profile_preamble, _resolve_token_estimator
        from ._loop.trace_schema import emit_trace_event
        from .context import FullTranscript
        from .loop import Session

        task_id = self._allocate_id()
        child_dir = self.run_root / "subagents" / task_id
        child_dir.mkdir(parents=True, exist_ok=False)
        child_trace = child_dir / ".trace.jsonl"
        binding: SubagentModelBinding | None = None
        child: Any = None
        result_obj: Any = None
        result_text = ""
        finish_reason = "error"
        done = False
        turns = 0
        prompt_tokens = 0
        completion_tokens = 0
        own_prompt_tokens = 0
        own_completion_tokens = 0
        try:
            binding = self._binding(parent, spec, task_id)
            child_cfg = replace(
                binding.config,
                profile_name=spec.model_profile,
                max_turns=min(
                    spec.max_turns,
                    int(parent.cfg.tools_subagent_max_turns),
                ),
                max_sessions=1,
                state_writer_enabled=False,
                tools_file_checkpoints_enabled=False,
                tools_background_enabled=False,
                injections_enabled=False,
                plan_mode="off",
                pre_mutation_turn_cap=0,
                rumination_enabled=False,
                done_guard_enabled=False,
                done_require_mutation=False,
                done_require_verify=False,
                done_require_pretest_parity=False,
                cache_affinity=False,
                cache_retention="off",
                advisor_enabled=False,
            )
            if "cfg" in getattr(binding.client, "__dict__", {}):
                binding.client.cfg = child_cfg
            if binding.dedicated and hasattr(binding.client, "set_transcript"):
                binding.client.set_transcript(child_dir / "transcript.log")

            arm = resolve_system_prompt_source(
                spec.system_prompt_file,
                imports_enabled=child_cfg.imports_enabled,
                allowed_dirs=(self.agents_dir.resolve(),),
                max_depth=child_cfg.imports_max_depth,
                unreadable_paths=child_cfg.unreadable_paths,
            )
            system_prompt = _apply_profile_preamble(
                assemble_system_prompt(
                    child_cfg.system_header,
                    resolved_arm=arm.content if arm is not None else None,
                ),
                binding.client,
            )
            context = FullTranscript(
                token_estimator=_resolve_token_estimator(binding.client)
                or (lambda messages: sum(len(str(item)) for item in messages) // 4)
            )
            with open(child_trace, "x") as trace_file:
                emit_trace_event(
                    trace_file,
                    "subagent_start",
                    id=task_id,
                    agent=spec.name,
                    parent_session_number=parent._session_number,
                    parent_turn_number=parent._current_turn,
                    depth=int(getattr(parent, "_subagent_level", 0)) + 1,
                    model_profile=spec.model_profile,
                    tools=list(spec.tools),
                    read_only=spec.read_only,
                    max_turns=child_cfg.max_turns,
                )
                child_env = dict(parent._effective_env)
                if spec.read_only:
                    child_env.pop("BASH_ENV", None)
                    child_env.pop("ENV", None)
                child = Session(
                    child_cfg,
                    binding.client,
                    system_prompt,
                    prompt,
                    parent.cwd,
                    context_manager=context,
                    trace_file=trace_file,
                    trace_path=child_trace,
                    state_path=None,
                    session_number=1,
                    output_control=parent.output_control,
                    universal_rewrites=parent.universal_rewrites,
                    forbidden_rules=parent.forbidden_rules,
                    redirect_rules=parent.redirect_rules,
                    redactions=parent.redactions,
                    output_parser=parent.output_parser,
                    tool_allowlist=frozenset(spec.tools),
                    subagent_level=int(getattr(parent, "_subagent_level", 0)) + 1,
                    subagent_runtime=self,
                    subagent_read_only=spec.read_only,
                    artifact_dir=child_dir,
                    ignore_policy=parent._ignore_policy,
                    effective_env=child_env,
                    allow_login_shell=(
                        False if spec.read_only else parent._allow_login_shell
                    ),
                )
                child._cache_usage_accumulator = getattr(
                    parent, "_cache_usage_accumulator", None
                )
                child._session_usage_accumulator = getattr(
                    parent, "_session_usage_accumulator", None
                )
                child._role_token_ledger = getattr(parent, "_role_token_ledger", None)
                role = binding.resolution or f"subagent.{spec.name}"
                child._active_model_resolution = role
                child._active_model_role = (
                    getattr(binding.resolution, "effective_role", None)
                    or f"subagent.{spec.name}"
                )
                result_obj = child.run()
                finish_reason = result_obj.finish_reason
                done = bool(result_obj.done)
                prompt_tokens = int(result_obj.total_prompt_tokens)
                completion_tokens = int(result_obj.total_completion_tokens)
                own_prompt_tokens = int(getattr(child, "_own_prompt_tokens", prompt_tokens))
                own_completion_tokens = int(
                    getattr(child, "_own_completion_tokens", completion_tokens)
                )
                turns = sum(
                    1 for event in child._trace_events if event.get("event") == "turn"
                )
                final_text = str(
                    getattr(child, "_final_text", "")
                    or getattr(child, "_last_assistant_content", "")
                    or ""
                )
                if done and final_text:
                    result_text = final_text
                elif done:
                    result_text = (
                        f"ERROR: subagent {task_id} returned no final text; "
                        f"task_id={task_id}"
                    )
                    done = False
                else:
                    result_text = (
                        f"ERROR: subagent {task_id} ended with {finish_reason}; "
                        f"task_id={task_id}"
                    )
                emit_trace_event(
                    trace_file,
                    "subagent_result",
                    id=task_id,
                    agent=spec.name,
                    turns=turns,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    own_prompt_tokens=own_prompt_tokens,
                    own_completion_tokens=own_completion_tokens,
                    tokens=prompt_tokens + completion_tokens,
                    finish_reason=finish_reason,
                    done=done,
                    result=result_text,
                    result_chars=len(result_text),
                    result_sha256=hashlib.sha256(result_text.encode("utf-8")).hexdigest(),
                )
        except Exception as exc:
            result_text = (
                f"ERROR: subagent {task_id} failed with {type(exc).__name__}: "
                f"{str(exc)[:400]}; task_id={task_id}"
            )
            if not child_trace.exists():
                child_trace.touch()
            with open(child_trace, "a") as trace_file:
                emit_trace_event(
                    trace_file,
                    "subagent_result",
                    id=task_id,
                    agent=spec.name,
                    turns=turns,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    own_prompt_tokens=own_prompt_tokens,
                    own_completion_tokens=own_completion_tokens,
                    tokens=prompt_tokens + completion_tokens,
                    finish_reason="error",
                    done=False,
                    result=result_text,
                    result_chars=len(result_text),
                    result_sha256=hashlib.sha256(result_text.encode("utf-8")).hexdigest(),
                )
            finish_reason = "error"
            done = False
        finally:
            if (
                binding is not None
                and binding.dedicated
                and hasattr(binding.client, "close_transcript")
            ):
                binding.client.close_transcript()

        return SubagentOutcome(
            task_id=task_id,
            agent=spec.name,
            result=result_text,
            turns=turns,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            own_prompt_tokens=own_prompt_tokens,
            own_completion_tokens=own_completion_tokens,
            finish_reason=finish_reason,
            done=done,
        )

    def _replay(self, parent: Any, agent: str) -> SubagentOutcome:
        events = list(getattr(parent.client, "subagent_events", ()) or ())
        if self._replay_index >= len(events):
            raise RuntimeError("replay source has no remaining subagent event")
        source_event = events[self._replay_index]
        self._replay_index += 1
        if source_event.get("agent") != agent:
            raise RuntimeError(
                "replay subagent mismatch: "
                f"recorded={source_event.get('agent')!r} live={agent!r}"
            )
        task_id = str(source_event.get("id") or "")
        if not _TASK_ID_RE.fullmatch(task_id):
            raise RuntimeError(f"replay subagent has invalid id {task_id!r}")
        source_trace_path = getattr(parent.client, "source_trace_path", None)
        if source_trace_path is None:
            raise RuntimeError("replay subagent requires the source parent trace")
        child_source = (
            Path(source_trace_path).parent / "subagents" / task_id / ".trace.jsonl"
        )
        terminal = _terminal_event(child_source, task_id, agent)
        result = str(terminal["result"])
        turns = int(terminal.get("turns", 0))
        prompt_tokens = int(terminal.get("prompt_tokens", 0))
        completion_tokens = int(terminal.get("completion_tokens", 0))
        if int(source_event.get("turns", -1)) != turns:
            raise RuntimeError(f"replay subagent {task_id} turn count mismatch")
        if int(source_event.get("tokens", -1)) != prompt_tokens + completion_tokens:
            raise RuntimeError(f"replay subagent {task_id} token count mismatch")
        if int(source_event.get("result_chars", -1)) != len(result):
            raise RuntimeError(f"replay subagent {task_id} result length mismatch")

        replay_child = self.run_root / "subagents" / task_id / ".trace.jsonl"
        if replay_child.resolve(strict=False) != child_source.resolve(strict=False):
            replay_child.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(child_source, replay_child)
        return SubagentOutcome(
            task_id=task_id,
            agent=agent,
            result=result,
            turns=turns,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            own_prompt_tokens=int(terminal.get("own_prompt_tokens", prompt_tokens)),
            own_completion_tokens=int(
                terminal.get("own_completion_tokens", completion_tokens)
            ),
            finish_reason=str(terminal.get("finish_reason") or ""),
            done=bool(terminal.get("done", False)),
        )


__all__ = [
    "AgentConfigError",
    "AgentSpec",
    "SubagentModelBinding",
    "SubagentOutcome",
    "SubagentRuntime",
    "load_agent_spec",
    "prepare_readonly_bash",
    "readonly_bash_decision",
]
