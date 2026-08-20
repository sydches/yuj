"""Harness-side adaptive hurdle control (off by default).

Scaffold for in-process adaptive control: an off-by-default pause point plus the
typed control schemas, control ledger, prefix detector, lookup selection, and
TOML overlay apply surface. The apply surface composes the baseline config
with the selected candidate TOML.
"""

INTERVENTION_SPACE_VERSION = "toml_overlay_control_v1"
CONTROLLER_VERSION = "adaptive_policy_v0_replay"
PAUSE_BOUNDARY = "post_turn_pre_next_model_call"
