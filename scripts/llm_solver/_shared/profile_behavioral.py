"""Helper for reading a profile's ``[behavioral]`` section.

Each ``profiles/<model>/profile.toml`` may declare a ``[behavioral]``
table with ``suffix``, ``tool_call_style``, and ``reminder_placement``
fields. This module loads the sibling profile.toml once at import
time from a calling behavioral.py's ``__file__`` and returns the
settings dict. Profiles that have not migrated to the TOML form
(empty or missing ``[behavioral]``) get an empty dict; the caller
falls back to its own in-file constant.

Content-blind. Lives in ``_shared/`` so both harness and profile
modules can import without a cross-layer reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .toml_compat import load_toml


@dataclass(frozen=True)
class BehavioralConfig:
    """Parsed ``[behavioral]`` section from a profile's TOML.

    All fields are optional — absence is a first-class signal that
    the profile has not yet migrated. Callers inspect ``suffix`` and
    fall back to an in-file constant when it is empty.
    """
    suffix: str = ""
    tool_call_style: str = "openai_json"
    reminder_placement: str = "system"


def load_profile_behavioral(module_file: str) -> BehavioralConfig:
    """Locate the sibling ``profile.toml`` of ``module_file`` and
    return its ``[behavioral]`` section as a BehavioralConfig.

    Typical usage from ``profiles/<model>/denormalize/behavioral.py``:

        from ..._shared.profile_behavioral import load_profile_behavioral
        _CFG = load_profile_behavioral(__file__)
        _SUFFIX = _CFG.suffix or _BEHAVIORAL_SUFFIX  # fall-back

    Walks up from ``module_file`` until it finds a directory
    containing ``profile.toml``. Raises FileNotFoundError if no
    profile.toml is found in any ancestor — profiles must declare
    themselves.
    """
    here = Path(module_file).resolve().parent
    for ancestor in [here, *here.parents]:
        candidate = ancestor / "profile.toml"
        if candidate.is_file():
            break
    else:
        raise FileNotFoundError(
            f"No profile.toml found in any ancestor of {module_file}"
        )
    data = load_toml(candidate)
    behavioral = data.get("behavioral", {}) or {}
    return BehavioralConfig(
        suffix=str(behavioral.get("suffix", "")),
        tool_call_style=str(behavioral.get("tool_call_style", "openai_json")),
        reminder_placement=str(behavioral.get("reminder_placement", "system")),
    )
