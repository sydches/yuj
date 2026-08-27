"""read tool: return file contents with line numbers + optional reminders."""
from ...config import Config
from ..sandbox.ignore_policy import active_ignore_policy
from ._common import _path_hint, _require_external_readable, _resolve_read


def _record_read_reminder(
    kind: str, path: str, before: str, after: str,
) -> str:
    """Record one exact read-tool reminder insertion."""
    from ..savings import get_ledger
    get_ledger().record_transform(
        bucket="tool_result_reminder",
        layer="harness",
        mechanism=f"read_{kind}",
        before=before,
        after=after,
        surface="tool_output",
        change_count=1,
        ctx={"kind": kind, "path": path},
    )
    return after


def read(path: str, *, cwd: str, offset: int = 0, limit: int = 0,
         cfg: Config | None = None) -> str:
    """Read a file, return contents with line numbers.

    When ``cfg`` is provided, appends a ``<system-reminder>`` block to
    the result in two cases:
      - the caller's ``limit`` capped the output before EOF
        (``read_truncated_reminder``);
      - the file exists but is 0-byte (``read_empty_reminder``).

    Reminders are off when ``cfg`` is None — preserves the signature
    for non-dispatch callers (tests, direct imports).

    Argument validation: offset/limit must be >= 0. Negative values are
    refused with a structured ERROR; previously they fell through to
    Python's negative slicing and produced output with negative line numbers.
    """
    if offset < 0:
        return f"ERROR: offset must be >= 0, got {offset}"
    if limit < 0:
        return f"ERROR: limit must be >= 0, got {limit}"
    try:
        target = _resolve_read(
            cwd,
            path,
            readonly_roots=tuple(
                getattr(cfg, "skills_readable_dirs", ()) or ()
            ) if cfg is not None else (),
        )
        if cfg is not None:
            _require_external_readable(
                cwd,
                target,
                unreadable_paths=tuple(
                    getattr(cfg, "unreadable_paths", ()) or ()
                ),
            )
        policy = active_ignore_policy(cwd)
        if policy is not None and (
            target == policy.root or policy.root in target.parents
        ):
            policy.require_visible(target, is_dir=target.is_dir())
        if target.is_dir():
            return (
                f"ERROR: {path} is a directory — "
                f"use glob to list contents."
            )
        raw_full = target.read_text()
        all_lines = raw_full.splitlines()
        total = len(all_lines)
        if offset > 0:
            lines = all_lines[offset:]
        else:
            lines = all_lines
        truncated = False
        returned = len(lines)
        if limit > 0 and returned > limit:
            lines = lines[:limit]
            returned = limit
            truncated = True
        raw_selected = "\n".join(lines)
        from ..savings import get_ledger
        get_ledger().record_transform(
            bucket="read_projection",
            layer="harness",
            mechanism="read_range_selection",
            before=raw_full,
            after=raw_selected,
            surface="tool_output",
            change_count=max(1, total - returned),
            ctx={"path": path, "offset": offset, "limit": limit},
        )
        start = (offset or 0) + 1
        numbered = [f"{start + i}: {line}" for i, line in enumerate(lines)]
        body = "\n".join(numbered)
        get_ledger().record_transform(
            bucket="read_projection",
            layer="harness",
            mechanism="read_line_numbering",
            before=raw_selected,
            after=body,
            surface="tool_output",
            change_count=max(1, len(lines)),
            ctx={"path": path, "start_line": start},
        )
        if cfg is None:
            return body
        if total == 0:
            tail = cfg.read_empty_reminder.format(path=path)
            result = body + ("\n" if body else "") + tail
            return _record_read_reminder("empty", path, body, result)
        if offset >= total and offset > 0:
            tail = cfg.read_offset_past_eof_reminder.format(
                offset=offset, total=total, path=path)
            return _record_read_reminder(
                "offset_past_eof", path, body, tail,
            )
        if truncated:
            tail = cfg.read_truncated_reminder.format(
                returned_lines=returned, path=path)
            result = body + "\n" + tail
            return _record_read_reminder("truncated", path, body, result)
        return body
    except FileNotFoundError:
        return f"ERROR: file not found: {path}" + _path_hint(cwd, path)
    except UnicodeDecodeError:
        return (
            f"ERROR: file {path} is not UTF-8 text "
            f"(use bash with `file <path>` or `xxd` to inspect bytes)."
        )
    except Exception as e:
        return f"ERROR: {e}"
