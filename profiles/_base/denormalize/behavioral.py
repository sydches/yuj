"""Behavioral denormalization — base profile.

The suffix text is loaded from ``_base/profile.toml`` ``[behavioral].suffix``
by the profile loader (see ``server/profile_loader.py``) which calls
``configure()`` on this module before its first ``apply()``. Prompt
literals live in config, not code. Without an explicitly configured suffix,
direct module use leaves the messages unchanged. Task validation commands
must come from task-bound evidence, not a profile fallback.

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


_BEHAVIORAL_SUFFIX_FALLBACK = ""


_BEHAVIORAL_SUFFIX: str = _BEHAVIORAL_SUFFIX_FALLBACK


def configure(behavioral_cfg: dict) -> None:
    """Accept the profile's [behavioral] dict from profile_loader."""
    global _BEHAVIORAL_SUFFIX
    if not isinstance(behavioral_cfg, dict):
        return
    suffix = behavioral_cfg.get("suffix", "")
    if isinstance(suffix, str):
        _BEHAVIORAL_SUFFIX = suffix


def apply(messages: list[dict]) -> list[dict]:
    """Append base behavioral instructions to the system prompt."""
    if not _BEHAVIORAL_SUFFIX or not messages or messages[0].get("role") != "system":
        return messages
    messages[0] = {
        **messages[0],
        "content": messages[0]["content"] + "\n" + _BEHAVIORAL_SUFFIX,
    }
    return messages
