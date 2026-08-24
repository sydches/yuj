"""edit tool: replace first occurrence of old_str with new_str in a file."""
from ...config import Config
from ._common import _path_hint, _resolve


def _whitespace_normalized_match(text: str, old_str: str) -> tuple[int, int] | None:
    """Back-compat shim — delegates to the new cascade module.

    Retained for external callers and for tests that import the name
    directly. New code should import from ``edit_replacers`` instead.
    """
    from ..edit_replacers import whitespace_normalized
    return whitespace_normalized(text, old_str)


def _record_edit_recovery(mechanism: str, path: str, old_str_len: int) -> None:
    """Record that a fuzzy-edit strategy rescued a non-exact match.

    Bucket ``fuzzy_edit_recovery`` on the savings ledger. Char delta
    is zero — the event records that the cascade fired, not a token
    saving. Use ``ctx.strategy`` when aggregating.
    """
    from ..savings import get_ledger
    get_ledger().record(
        bucket="fuzzy_edit_recovery",
        layer="harness",
        mechanism=mechanism,
        input_chars=old_str_len,
        output_chars=old_str_len,
        measure_type="estimate",
        ctx={"strategy": mechanism, "path": path},
    )


def _format_candidates_block(text: str, candidates, path: str) -> str:
    """Render a ranked-candidate block for strict-mode miss reporting.

    Uses XML-shape envelope so the agent can parse by tag shape. Each
    inner <candidate> quotes the exact substring of ``text`` between
    the candidate's (start, end) offsets — the agent can copy it
    verbatim into a retry.
    """
    from ..edit_replacers import format_candidates_block
    return format_candidates_block(text, candidates, path)


def edit(path: str, old_str: str, new_str: str, *, cwd: str,
         cfg: Config | None = None) -> str:
    """Replace first occurrence of old_str with new_str in a file.

    Match policy is controlled by two cfg flags:

      edit_strict_match (default true) + edit_fuzzy_cascade_enabled
      (default false):
          exact match only; on miss, return a ranked <candidates/>
          block so the agent can choose and retry.

      edit_fuzzy_cascade_enabled = true:
          fall through to the cascade after an exact miss and
          auto-apply the first passing strategy.

    When cfg is None (test-only convenience), strict mode is used.
    """
    from ..edit_replacers import find_span, rank_candidates
    from ..post_edit import run_post_edit_checks
    if "\n" in path or "\x00" in path:
        return f"ERROR: path contains forbidden character (newline or NUL)"
    if old_str == "":
        return (
            "ERROR: old_str must be non-empty — an empty old_str would "
            "silently prepend new_str to the file. Choose a non-empty "
            "exact span from the existing file."
        )
    try:
        target = _resolve(cwd, path)
        try:
            previous_bytes = target.read_bytes()
        except FileNotFoundError:
            return f"ERROR: file not found: {path}" + _path_hint(cwd, path)
        try:
            text = previous_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return (
                f"ERROR: cannot edit {path}: file is not valid UTF-8 "
                "(likely a binary file). The exact edit dialect supports "
                "UTF-8 text only."
            )
        # Detect CRLF style so we can preserve it on write. read_bytes
        # gives us exact bytes; .decode normalises nothing. We work in
        # str-space (cascade matchers expect str), then re-encode with
        # the original line-ending style restored.
        is_crlf = b"\r\n" in previous_bytes
        if is_crlf:
            text = text.replace("\r\n", "\n")
        new_text: str | None = None
        head = ""
        # Pass 1: exact.
        if old_str in text:
            new_text = text.replace(old_str, new_str, 1)
            head = "OK"
        elif cfg is not None and cfg.edit_fuzzy_cascade_enabled:
            # Optional cascade: auto-apply the first matching strategy.
            hit = find_span(text, old_str)
            if hit is not None:
                mechanism, start, end = hit
                new_text = text[:start] + new_str + text[end:]
                head = f"OK ({mechanism.replace('_', '-')})"
                _record_edit_recovery(mechanism, path, len(old_str))
        if new_text is None:
            # Strict-mode miss (or cascade-miss): surface ranked
            # candidates so the agent can retry with correct bytes.
            k = cfg.edit_candidate_count if cfg is not None else 3
            candidates = rank_candidates(text, old_str, k=k)
            head = f"ERROR: old_str not found in {path}"
            block = _format_candidates_block(text, candidates, path)
            return f"{head}\n{block}" if block else head
        out_text = new_text.replace("\n", "\r\n") if is_crlf else new_text
        target.write_bytes(out_text.encode("utf-8"))
        res = run_post_edit_checks(path, cwd=cwd, cfg=cfg, trigger="edit")
        if res.action == "block":
            target.write_bytes(previous_bytes)
            return (
                f"ERROR: edit blocked by post-edit check "
                f"'{res.check_name}' for {path}{res.output}"
            )
        return head + res.output
    except FileNotFoundError:
        return f"ERROR: file not found: {path}" + _path_hint(cwd, path)
    except Exception as e:
        return f"ERROR: {e}"
