"""Content-blind classifiers for tool results.

The harness must not derive intelligence from task output. These helpers
observe only harness-generated markers — either the unified
``<tool_result status="...">`` envelope, gated on
``tools_unified_envelope_enabled``, or the legacy in-band markers
(``ERROR:`` wrapper, ``[exit code: N]`` suffix, ``[harness gate]``
prefix). They never inspect task content.

Envelope-first lookup is the current path. Legacy markers remain as a
fallback when the envelope is absent.
"""
import re


# Cheap envelope sniffer. The envelope is always the FIRST tag in the
# result text when present (see harness/tools.py::dispatch where the
# wrap step runs after every other transform). Match-fail returns None
# and the caller falls back to legacy in-band markers.
_ENVELOPE_RE = re.compile(
    r'^<tool_result\b[^>]*\bstatus="(?P<status>[a-z_]+)"',
)

# Inner-envelope sniffers. The four typed tools (run_tests, list_
# definitions, apply_patch) produce their own structured envelope BEFORE
# the unified <tool_result> wrap. classify_outcome / derive_envelope_
# status must recognise these so an error-shape inner envelope (e.g.
# <test_results status="timed_out"> or <list_definitions status="error">)
# is not misclassified as OK when D=off (no outer wrap). Per-tool
# status vocabularies map onto the unified {ok, error, blocked,
# timed_out} set.
_TEST_RESULTS_RE = re.compile(
    r'^<test_results\b[^>]*\bstatus="(?P<status>[a-z_0-9]+)"',
)
_LIST_DEFINITIONS_RE = re.compile(
    r'^<list_definitions\b[^>]*\bstatus="(?P<status>[a-z_]+)"',
)
_APPLY_PATCH_RE = re.compile(
    r'^<apply_patch\b[^>]*\bok="(?P<ok>true|false)"',
)

# Harness-appended exit-code marker (see harness/tools.py:257-259, 900).
# The marker is appended ONLY when exit_code != 0, on its own line, at
# end of result text. Anchoring to end-of-text closes the false-positive
# window where a successful command's stdout happens to contain the
# literal substring `[exit code: 5]` (e.g. a script printing diagnostic
# text that matches the harness marker shape). Optional " — annotation"
# tail covers _semantic_exit_annotation.
# The marker may be followed only by harness-appended `[HARNESS: ...]`
# hint blocks (env/pytest hints attach after it); without that allowance
# a hinted failure would parse as OK and the trace would record
# exit_status=0 for a failed command.
_EXIT_MARKER_TAIL_RE = re.compile(
    r"\n\[exit code:\s*(?P<code>\d+)(?:\s+—\s+[^\]]*)?\]"
    r"(?:\s*\n\[HARNESS:[^\]]*\])*\s*\Z"
)

# run_tests inner status → unified status. "passed" → ok; "timed_out"
# stays itself; everything else is an error class. New per-runner
# statuses added to _PYTEST_STATUS map should land here too.
_TEST_RESULTS_OK = {"passed"}
_TEST_RESULTS_TIMEOUT = {"timed_out"}


def _read_envelope_status(result: str) -> str | None:
    """Return the envelope's status attribute, or None if absent.

    Returns one of "ok" / "error" / "blocked" / "timed_out" when the
    envelope is present (matching derive_envelope_status output), None
    otherwise.

    Recognises four envelope shapes, in priority order:
      1. unified <tool_result status="..."> — the outer wrap from
         dispatch when ``tools_unified_envelope_enabled`` is on.
      2. <test_results status="..."> — run_tests native envelope.
      3. <list_definitions status="..."> — list_definitions native envelope.
      4. <apply_patch ok="true|false"> — apply_patch native envelope
         (uses ``ok`` rather than ``status``).
    """
    if not result:
        return None
    if result.startswith("<tool_result"):
        m = _ENVELOPE_RE.match(result)
        return m.group("status") if m else None
    if result.startswith("<test_results"):
        m = _TEST_RESULTS_RE.match(result)
        if m is None:
            return None
        s = m.group("status")
        if s in _TEST_RESULTS_OK:
            return "ok"
        if s in _TEST_RESULTS_TIMEOUT:
            return "timed_out"
        return "error"
    if result.startswith("<list_definitions"):
        m = _LIST_DEFINITIONS_RE.match(result)
        # Pre-F6 success envelopes have no status attr — treat the
        # bare-prefix match as ok (the legacy success contract).
        if m is None:
            return "ok"
        return m.group("status")
    if result.startswith("<apply_patch"):
        m = _APPLY_PATCH_RE.match(result)
        if m is None:
            return None
        return "ok" if m.group("ok") == "true" else "error"
    return None


def is_error_result(result: str) -> bool:
    """True if ``result`` represents a tool-level error.

    Recognises every typed envelope shape AND the legacy bare
    ``ERROR:`` prefix. Distinguished from content failures: a
    ``<test_results status="failed">`` is a real test outcome the
    model should react to, NOT a tool-level error — so this returns
    False for it.

    The unified ``<tool_result … status="error">`` wrap is treated
    as an error. ``status="timed_out"`` is treated as an error too:
    timeouts are tool failures (the harness gave up), not content
    outcomes.

    Used by guardrail post-checks to count escalating tool errors
    and by extractors to strip the prefix before signature matching.
    """
    if not result:
        return False
    if result.startswith("ERROR:"):
        return True
    # Envelope shapes: ask the existing reader. test_results' "error"
    # mapping covers timeouts AND non-passed/non-timeout statuses; for
    # the guardrail "is this a TOOL failure" question we want to
    # distinguish content "failed" from tool "error" — the reader
    # already returns "ok" for passed and "error" for the rest, so
    # `<test_results status="failed">` would land in "error" too.
    # Therefore: read the envelope status only when it is NOT a
    # test_results envelope; for test_results, "failed" is a content
    # outcome, NOT a tool error.
    if result.startswith("<test_results"):
        return False
    env = _read_envelope_status(result)
    return env in ("error", "timed_out")


def classify_outcome(result: str) -> str:
    """Return "OK" or "FAIL" from harness-generated markers only.

    Content-blind: inspects only strings the harness itself writes into
    the tool result. Never parses task output.

    Lookup order:
      1. Unified envelope (when present): status="ok" → OK,
         status ∈ {"error", "blocked", "timed_out"} → FAIL.
      2. Legacy in-band markers (envelope absent):
         - empty / no error markers → OK
         - starts with "ERROR:" → FAIL
         - "[exit code: N]" with N != 0 → FAIL
    """
    env = _read_envelope_status(result)
    if env is not None:
        return "OK" if env == "ok" else "FAIL"
    if not result:
        return "OK"
    if result.startswith("ERROR:"):
        return "FAIL"
    m = _EXIT_MARKER_TAIL_RE.search(result)
    if m and m.group("code") != "0":
        return "FAIL"
    return "OK"


def is_gate_blocked(result: str) -> bool:
    """True if the result is a harness gate message (tool was not executed).

    Reads the envelope's status="blocked" first, falls back to the
    legacy "[harness gate]" prefix. The legacy prefix appears INSIDE
    the envelope body when both are present, so the envelope sniff
    must run first or we'd double-count.
    """
    env = _read_envelope_status(result)
    if env is not None:
        return env == "blocked"
    return result.startswith("[harness gate]")


def derive_envelope_status(result: str) -> tuple[str, str | None]:
    """Return (status, error_kind) for the unified <tool_result> envelope.

    status ∈ {"ok", "error", "blocked", "timed_out"}. error_kind is a
    short categorical tag for "error" status only. It lets readers
    distinguish failures without parsing the in-band markers again.

    Content-blind: same observable contract as classify_outcome — only
    inspects strings the harness itself writes into the result text.

    Idempotent: re-deriving on an already-wrapped result returns the
    envelope's own attributes (so a double-wrap can be detected, and
    transforms that pre-wrap then post-process don't drift the status).

    Categorisation logic (in order):
      - already wrapped → echo envelope attributes
      - empty → ("ok", None)
      - is_gate_blocked → ("blocked", "harness_gate")
      - starts with "ERROR: command timed out" → ("timed_out", "timeout")
      - starts with "ERROR:" → ("error", "tool_exception")
      - "[exit code: N]" with N != 0 → ("error", "nonzero_exit")
      - otherwise → ("ok", None)
    """
    env = _read_envelope_status(result)
    if env is not None:
        # Echo: a re-derivation on already-wrapped text is informational
        # only; we cannot recover error_kind without parsing the full
        # attribute list, so report None and let callers handle.
        m = re.search(r'\berror_kind="([^"]+)"', result)
        return (env, m.group(1) if m else None)
    if not result:
        return ("ok", None)
    if is_gate_blocked(result):
        return ("blocked", "harness_gate")
    if result.startswith("ERROR: command timed out"):
        return ("timed_out", "timeout")
    if result.startswith("ERROR: stale_file:"):
        return ("error", "stale_file")
    if result.startswith("ERROR:"):
        return ("error", "tool_exception")
    m = _EXIT_MARKER_TAIL_RE.search(result)
    if m and m.group("code") != "0":
        return ("error", "nonzero_exit")
    return ("ok", None)


__all__ = ["classify_outcome", "is_gate_blocked", "derive_envelope_status"]
