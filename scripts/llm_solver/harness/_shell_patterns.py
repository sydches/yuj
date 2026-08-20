"""Shared shell-surface regexes used by multiple harness modules.

``TEST_COMMAND_RE`` — "is this bash call a test/verification command?" —
is derived from the union of every language quirk's
``verification_patterns`` (see ``language_quirks.all_verification_patterns``),
so it covers pytest / go / cargo / jest / ctest / generic identically and
can never drift from the per-runner TOMLs. A small hard-coded fallback is
kept only for the case where the quirk package can't be imported.
"""
import re

_FALLBACK = (
    r"\b(pytest|py\.test|python\s+-m\s+pytest|python3\s+-m\s+pytest|"
    r"unittest|cargo test|go test|ctest|npm test|pnpm test|yarn test)\b"
)


def _build_test_command_re() -> "re.Pattern[str]":
    try:
        from ..language_quirks import all_verification_patterns
        pats = all_verification_patterns()
        if pats:
            # Each quirk pattern is authored for re.VERBOSE|re.IGNORECASE
            # (see the language_quirks TOMLs); combine under the same flags.
            joined = "|".join(f"(?:{p})" for p in pats)
            return re.compile(joined, re.IGNORECASE | re.VERBOSE)
    except Exception:
        pass
    return re.compile(_FALLBACK, re.IGNORECASE)


TEST_COMMAND_RE = _build_test_command_re()
