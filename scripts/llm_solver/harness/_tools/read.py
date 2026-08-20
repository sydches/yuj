"""read tool: return file contents with line numbers + optional reminders."""
from ...config import Config
from ._common import _path_hint, _resolve


def _record_read_reminder(kind: str, path: str, reminder_chars: int) -> None:
    """Record a read-tool reminder injection on the savings ledger.

    Bucket ``tool_result_reminder``; ``kind`` ∈ {"truncated", "empty"}.
    Input chars = 0 (nothing pre-existed); output chars = reminder length,
    so ``delta_chars`` is the positive cost paid to inject the block.
    """
    from ..savings import get_ledger
    get_ledger().record(
        bucket="tool_result_reminder",
        layer="harness",
        mechanism=f"read_{kind}",
        input_chars=0,
        output_chars=int(reminder_chars),
        measure_type="exact",
        ctx={"kind": kind, "path": path},
    )


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
        target = _resolve(cwd, path)
        if target.is_dir():
            return (
                f"ERROR: {path} is a directory — "
                f"use glob to list contents."
            )
        all_lines = target.read_text().splitlines()
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
        start = (offset or 0) + 1
        numbered = [f"{start + i}: {line}" for i, line in enumerate(lines)]
        body = "\n".join(numbered)
        if cfg is None:
            return body
        if total == 0:
            tail = cfg.read_empty_reminder.format(path=path)
            _record_read_reminder("empty", path, len(tail))
            return body + ("\n" if body else "") + tail
        if offset >= total and offset > 0:
            tail = cfg.read_offset_past_eof_reminder.format(
                offset=offset, total=total, path=path)
            _record_read_reminder("offset_past_eof", path, len(tail))
            return tail
        if truncated:
            tail = cfg.read_truncated_reminder.format(
                returned_lines=returned, path=path)
            _record_read_reminder("truncated", path, len(tail))
            return body + "\n" + tail
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
