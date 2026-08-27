"""Cross-tool helpers: cwd-rooted paths, execution text, and XML rendering."""
from pathlib import Path, PurePosixPath


class ToolExecutionText(str):
    """String-compatible tool output carrying process facts out of the tool.

    The model-facing tool pipeline remains string based.  This subclass lets a
    process-backed tool preserve its real exit status until ``dispatch`` has
    copied it into private execution metadata, before later harness reminders
    decorate the visible text.
    """

    def __new__(
        cls,
        text: str,
        *,
        exit_status: int | None,
        timed_out: bool = False,
    ) -> "ToolExecutionText":
        value = super().__new__(cls, text)
        value.exit_status = exit_status
        value.timed_out = bool(timed_out)
        return value


def _resolve(cwd: str, path: str) -> Path:
    """Resolve a tool path relative to cwd with containment.

    Path handling:

    - Relative paths: joined against ``cwd``.
    - Absolute paths already inside ``cwd``: used as-is after
      ``resolve(strict=False)``.
    - Absolute paths outside ``cwd``: re-rooted under ``cwd`` by
      stripping the leading slash. This preserves the sandbox
      perimeter (an attempt to read ``/etc/passwd`` resolves to
      ``<cwd>/etc/passwd`` which safely won't exist).

    The result is normalized against ``..`` and symlinks via
    ``Path.resolve(strict=False)`` and must remain inside ``cwd``
    after resolution; out-of-cwd resolution raises ``ValueError`` so
    the caller can surface a structured ERROR. The harness's FS tools
    (read/write/edit/glob/grep/list_definitions) all run in-process
    and rely on this helper — not on the bash sandbox — for their
    cwd perimeter.
    """
    cwd_p = Path(cwd).resolve()
    if path.startswith("/"):
        abs_target = Path(path).resolve(strict=False)
        try:
            abs_target.relative_to(cwd_p)
            target = abs_target
        except ValueError:
            # Local import: harness.sandbox at module level drags the
            # full harness package (server client, openai) into every
            # tools consumer; tests import _resolve standalone.
            from ..sandbox import AMBIENT_CONTAINER, container_mode
            mode = container_mode()
            if mode is not None and mode != AMBIENT_CONTAINER:
                # docker-exec mode: bash runs inside a container that
                # mounts cwd AT /testbed (sandbox/__init__.py argv
                # builder), while FS tools run on the host. /testbed/...
                # is therefore an alias for cwd; any other absolute path
                # is container-local and a host-side re-root would write
                # a phantom the model's bash can never see (ledger #13,
                # v3 task 333: "OK: wrote 218 bytes" vs "No such file").
                parts = PurePosixPath(path).parts
                if parts[:2] == ("/", "testbed"):
                    target = (cwd_p.joinpath(*parts[2:])).resolve(strict=False)
                else:
                    raise ValueError(
                        f"absolute path {path} is container-local and not "
                        "visible to the write/read/edit tools; use a path "
                        "under /testbed or a relative path"
                    ) from None
            else:
                # bwrap/ambient: re-root for sandbox containment.
                target = (cwd_p / path.lstrip("/")).resolve(strict=False)
    else:
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
    cwd_path = Path(cwd).resolve(strict=False)
    if target == cwd_path or cwd_path in target.parents or not unreadable_paths:
        return
    from ..project_instructions import _UnreadableMatcher

    if _UnreadableMatcher(cwd_path, unreadable_paths).blocks(target):
        raise FileNotFoundError(str(target))


def _path_hint(cwd: str, path: str) -> str:
    """Suggest a corrected path when a file-not-found error occurs.

    Catches the `.seaborn/X` → `seaborn/X` pattern where the model
    confuses `.solver/` (hidden dir) with the package directory.
    """
    stripped = path.lstrip("./")
    if stripped != path:
        candidate = Path(cwd) / stripped
        if candidate.exists():
            return f" (did you mean '{stripped}'?)"
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
