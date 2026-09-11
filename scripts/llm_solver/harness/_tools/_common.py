"""Cross-tool helpers: cwd-rooted paths, execution text, and XML rendering."""
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ..injections import UserTurnInjection


class ToolExecutionText(str):
    """String-compatible tool output carrying process facts out of the tool.

    The model-facing tool pipeline remains string based.  This subclass lets a
    process-backed tool preserve its real exit status and user-turn advice
    records until ``dispatch`` has copied them into private execution metadata.
    """

    def __new__(
        cls,
        text: str,
        *,
        exit_status: int | None,
        timed_out: bool = False,
        verification_status: str = "",
        runner_request: dict | None = None,
        execution_budget: dict | None = None,
        executed: bool | None = None,
        user_turn_injections: Iterable["UserTurnInjection"] = (),
    ) -> "ToolExecutionText":
        value = super().__new__(cls, text)
        value.exit_status = exit_status
        value.timed_out = bool(timed_out)
        value.verification_status = verification_status
        value.runner_request = runner_request
        value.execution_budget = execution_budget
        value.executed = executed
        value.user_turn_injections = tuple(user_turn_injections)
        return value


def _tool_advice(
    text: str, *, mechanism: str, tool_name: str, **ctx: object,
) -> "UserTurnInjection":
    """Build one tool-produced recovery record for user-turn delivery."""
    from ..injections import UserTurnInjection

    return UserTurnInjection(
        text=text,
        bucket="advice_injection",
        mechanism=mechanism,
        ctx={"tool_name": tool_name, **ctx},
    )


def _resolve(cwd: str, path: str):
    """Resolve a contained task path in the caller's selected file view.

    A bound executor owns native path aliases and filesystem operations.
    Without one, this helper resolves only local task paths. Outside absolute
    paths are refused, never reinterpreted as different files under cwd.
    Container mount metadata alone cannot authorize host access.
    """
    from ..task_path import bound_task_path
    bound = bound_task_path(cwd, path)
    if bound is not None:
        return bound
    cwd_p = Path(cwd).resolve()
    target = (cwd_p / path).resolve(strict=False)
    try:
        target.relative_to(cwd_p)
    except ValueError:
        raise ValueError(f"path escapes cwd: {path}") from None
    return target


def _resolve_read(
    cwd: str,
    path: str,
    *,
    readonly_roots: tuple[str, ...] = (),
) -> Path:
    """Resolve a read path under cwd or an explicit read-only skill root.

    Only absolute paths can select an external root. This keeps ordinary
    relative tool behavior rooted at the task while allowing the system
    prompt to disclose exact ``SKILL.md`` and resource paths.
    """
    from ..task_path import active_task_files, TaskPath, native_requested_path
    files = active_task_files(cwd)
    if files is not None:
        value = PurePosixPath(path)
        requested = native_requested_path(TaskPath(files, files.root), path, expand=False)
        if value.is_absolute() and not requested.path.is_relative_to(files.root):
            for raw_root in readonly_roots:
                root = PurePosixPath(raw_root)
                if root.is_absolute() and value.is_relative_to(root):
                    external = files.readonly_view(root)
                    return TaskPath(external, external.root / value.relative_to(root)).resolve()
        return _resolve(cwd, path)
    if path.startswith("/") and readonly_roots:
        target = Path(path).resolve(strict=False)
        for raw_root in readonly_roots:
            root = Path(raw_root).resolve(strict=False)
            if target == root or root in target.parents:
                return target
    return _resolve(cwd, path)


def _is_external_readonly_path(
    cwd: str,
    path: str,
    *,
    readonly_roots: tuple[str, ...] = (),
) -> bool:
    """Return whether an absolute target belongs to an external skill root."""
    if not path.startswith("/"):
        return False
    from ..task_path import active_task_files, TaskPath, native_requested_path
    files = active_task_files(cwd)
    if files is not None:
        target = PurePosixPath(path)
        requested = native_requested_path(TaskPath(files, files.root), path, expand=False)
        if requested.path.is_relative_to(files.root):
            return False
        return any(target.is_relative_to(PurePosixPath(root)) for root in readonly_roots)
    target = Path(path).resolve(strict=False)
    cwd_path = Path(cwd).resolve(strict=False)
    if target == cwd_path or cwd_path in target.parents:
        return False
    return any(
        target == (root := Path(raw_root).resolve(strict=False))
        or root in target.parents
        for raw_root in readonly_roots
    )


def _require_external_readable(
    cwd: str,
    target: Path,
    *,
    unreadable_paths: tuple[str, ...] = (),
) -> None:
    """Apply configured masks to an otherwise allowed external read."""
    from ..task_path import TaskPath
    if isinstance(target, TaskPath):
        # Native task access already uses the selected command file view.
        return
    cwd_path = Path(cwd).resolve(strict=False)
    if target == cwd_path or cwd_path in target.parents or not unreadable_paths:
        return
    from ..project_instructions import _UnreadableMatcher

    if _UnreadableMatcher(cwd_path, unreadable_paths).blocks(target):
        raise FileNotFoundError(str(target))


def _path_hint(cwd: str, path: str) -> str:
    """Suggest an existing path after a file-not-found error.

    Strip leading dot and slash characters, then suggest the resulting
    cwd-relative path only when it exists.
    """
    stripped = path.lstrip("./")
    if stripped != path:
        try:
            candidate = _resolve(cwd, stripped)
            if candidate.exists():
                return f" (did you mean '{stripped}'?)"
        except (ValueError, OSError):
            pass
    return ""


def _xml_attr(s: str) -> str:
    """Escape a string for inclusion as an XML attribute value."""
    return (
        s.replace("&", "&amp;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _xml_body(s: str) -> str:
    """Escape a string for inclusion inside an XML element body.

    Body content needs only `&`, `<`, `>` escaped (no quote escapes —
    quotes are attribute-only delimiters). Used by `list_definitions`
    to keep docstrings / decorator-arg literals from terminating the
    `<list_definitions>` envelope when they contain the literal tag string.
    """
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _paginated_envelope(
    *, tool: str, pattern: str, scope: str, lines: list[str],
    page: int, per_page: int, before_text: str | None = None,
) -> str:
    """Wrap ``lines`` in a ``<search_result/>`` envelope for grep/glob."""
    total = len(lines)
    if per_page <= 0:
        per_page = total or 1
    page = max(1, page)
    start = (page - 1) * per_page
    end = start + per_page
    shown_slice = lines[start:end]
    next_page = page + 1 if end < total else 0
    opening = (
        f'<search_result tool="{tool}" total="{total}" '
        f'shown="{len(shown_slice)}" page="{page}" '
        f'next_page="{next_page}" pattern="{_xml_attr(pattern)}" '
        f'scope="{_xml_attr(scope)}">'
    )
    body = "\n".join(shown_slice) if shown_slice else ""
    result = f"{opening}\n{body}\n</search_result>"
    before = "\n".join(lines) if before_text is None else before_text
    from ..savings import get_ledger
    get_ledger().record_transform(
        bucket="search_pagination",
        layer="harness",
        mechanism=f"{tool}_page",
        before=before,
        after=result,
        surface="tool_output",
        change_count=max(1, total - len(shown_slice)),
        ctx={
            "tool": tool,
            "total": total,
            "shown": len(shown_slice),
            "page": page,
            "next_page": next_page,
            "pattern": pattern,
            "scope": scope,
        },
    )
    return result
