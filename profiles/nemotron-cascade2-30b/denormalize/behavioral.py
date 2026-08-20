"""Add the profile's behavioral suffix to a Yuj system prompt.

``configure()`` loads the suffix from ``profile.toml``. The in-file value
supports direct module use. Non-Yuj modes do not add the suffix.
"""


_BEHAVIORAL_SUFFIX_FALLBACK = """

## Operational rules

Constraints, not advice.

1. **Test suite is the only verdict.** Run it. `python -c`, imports, `echo` are not verification.
2. **Test file = interface contract.** Read it once, early.
3. **Never modify existing test files.**
4. **Last action must be a test run.** No fresh verdict = not done.
5. **First action: mutate or test.** Pretest verdict is in context. Do not list/read/grep first.
6. **Iterate, not rewrite.** Read the error. Make a targeted edit. Do not start over.
7. **No re-runs without intervening mutation.** Flag variations don't count.
8. **Sessions resume.** Continue from where you stopped. Do not restart."""


# Populated by configure() at profile load time. Falls back to the
# in-file constant when configure() is not called or the profile's
# [behavioral].suffix is empty.
_BEHAVIORAL_SUFFIX: str = _BEHAVIORAL_SUFFIX_FALLBACK


def configure(behavioral_cfg: dict) -> None:
    """Accept the profile's [behavioral] dict.

    Called by ``server/profile_loader.py::_load_code_module`` with
    the parsed ``[behavioral]`` section from profile.toml before
    ``apply()`` is first invoked. Idempotent: repeated calls replace
    the suffix without side effects.
    """
    global _BEHAVIORAL_SUFFIX
    if not isinstance(behavioral_cfg, dict):
        return
    suffix = behavioral_cfg.get("suffix", "")
    if isinstance(suffix, str) and suffix:
        _BEHAVIORAL_SUFFIX = suffix


def apply(messages: list[dict]) -> list[dict]:
    """Append operational rules to the system prompt.

    Only activates when the commandments are present (with_yuj modes).
    wo_yuj modes do not add a behavioral injection.

    Idempotent: if the suffix is already in the content, return unchanged.
    Without this check, calling apply() twice on the same underlying
    messages list (which happens in the harness's raw-append fallback mode
    for the first 2 turns of each session, where context.get_messages()
    returns self._all_messages by reference and behavioral.apply() mutates
    messages[0] in place) would append the suffix twice → the model sees
    the behavioral rules duplicated at those turns, a cache-miss and
    ~2KB/turn waste. The check costs one substring search per turn.
    """
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") != "system":
        return messages
    content = first["content"]
    # Marker check first: on the common turn-2+ path the suffix is
    # already present, so the cheap marker substring hits and we
    # return without the more expensive "Commandments" scan. Only the
    # first injection turn pays both scans.
    if _BEHAVIORAL_MARKER in content:
        return messages
    if "Commandments" not in content:
        return messages
    messages[0] = {**first, "content": content + _BEHAVIORAL_SUFFIX}
    return messages


# Use a distinctive phrase from the suffix as the idempotency marker.
_BEHAVIORAL_MARKER = "## Operational rules"
