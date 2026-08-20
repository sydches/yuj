"""Behavioral denormalization — base profile.

The suffix text is loaded from ``_base/profile.toml`` ``[behavioral].suffix``
by the profile loader (see ``server/profile_loader.py``) which calls
``configure()`` on this module before its first ``apply()``. Prompt
literals live in config, not code. The in-file constant remains as
a fallback for bootstrap / test environments where configure() is
not called.

All models inherit ``_base`` in name, but the loader currently does
not wire this behavioral module into any downstream profile — the
``_base`` ``[denormalize].modules`` list is empty. Tests assert the
noop (test_base_behavioral_is_noop_when_not_loaded). Confirm the
profile loader wiring before assuming this suffix reaches the
model.

Contract:
  def apply(messages: list[dict]) -> list[dict]
  - Must accept and return OpenAI-format message list
  - May modify system prompt content; must not alter message structure
  - Must be idempotent and task-agnostic
"""


_BEHAVIORAL_SUFFIX_FALLBACK = """
Missing file referenced in errors? Write it — use error output, interface descriptions, and surrounding code as spec.
After every mutation, run the initial validation command. Read output. Fix. Repeat to exit code 0. No proxy checks.
All paths relative. No absolute paths."""


_BEHAVIORAL_SUFFIX: str = _BEHAVIORAL_SUFFIX_FALLBACK


def configure(behavioral_cfg: dict) -> None:
    """Accept the profile's [behavioral] dict from profile_loader."""
    global _BEHAVIORAL_SUFFIX
    if not isinstance(behavioral_cfg, dict):
        return
    suffix = behavioral_cfg.get("suffix", "")
    if isinstance(suffix, str) and suffix:
        _BEHAVIORAL_SUFFIX = suffix


def apply(messages: list[dict]) -> list[dict]:
    """Append base behavioral instructions to the system prompt."""
    if not messages or messages[0].get("role") != "system":
        return messages
    messages[0] = {
        **messages[0],
        "content": messages[0]["content"] + "\n" + _BEHAVIORAL_SUFFIX,
    }
    return messages
