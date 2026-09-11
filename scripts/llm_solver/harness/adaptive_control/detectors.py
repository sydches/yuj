"""Zero live hurdle detector baseline.

The previous live hurdle detectors are retired. The adaptive controller,
debug ledger, control ledger, lookup plumbing, and intervention plumbing may
still exist, but no live hurdle signal is currently wired.

New detector rules must be added one at a time with explicit evidence and their
own version bump.
"""
from __future__ import annotations

from pathlib import Path

from .rules import build_registry, load_rule_catalog, retired_noop

DETECTOR_VERSION = "zero_detector_v0"
NAIVE_RED_VERSION = "diagnostic_observations_v1"

_NO_FIRE = "no_fire"



def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _turn_number(item: dict) -> int | None:
    value = item.get("turn_number", item.get("slot_idx"))
    try:
        turn = int(value)
    except (TypeError, ValueError):
        return None
    return turn if turn >= 0 else None


def is_naive_red_turn(item: dict, *, version: str = NAIVE_RED_VERSION) -> bool:
    """Select diagnostic observations, never test or hurdle evidence.

    Current sampling includes every named tool observation and projected slot.
    The historical lexical selection is available only by its explicit version.
    """
    if version == "naive_red_v0":
        return _legacy_naive_red_turn(item)
    if version != NAIVE_RED_VERSION:
        raise ValueError("unknown diagnostic selection version: " + version)
    if item.get("event", "tool_call") != "tool_call":
        return False
    return bool(item.get("tool_name") or item.get("op_kind"))


def _legacy_naive_red_turn(item: dict) -> bool:
    from ...language_quirks import _load_runner_quirk_dict
    legacy = _load_runner_quirk_dict("diagnostics")["legacy_naive_red_v0"]
    if item.get("event", "tool_call") != "tool_call":
        return False

    tool_name = (item.get("tool_name") or "").lower()
    args = (item.get("args_summary") or item.get("args_prefix") or "").lower()
    result = (item.get("result_summary") or "").lower()

    if _truthy(item.get("gate_blocked")):
        return True
    if tool_name in {"done", "submit"}:
        return True
    if _truthy(item.get("source_write_like")) or _truthy(item.get("write_like")):
        return True
    if _truthy(item.get("source_mutation")) or _truthy(item.get("effective_source_mutation")):
        return True
    if item.get("slot_state") in {"tool_error", "test_fail", "submit", "done"}:
        return True
    if item.get("op_kind") == "SUBMIT" or _truthy(item.get("submit_like_action")):
        return True
    if item.get("exec_outcome") in {"fail", "pass"} or _truthy(item.get("test_like_action")):
        return True
    if item.get("traceback_paths"):
        return True
    if "git" in args and "status" in args:
        return True
    if any(token in args for token in legacy["args_tokens"]):
        return True
    if any(token in result for token in legacy["result_tokens"]):
        return True
    return False


def naive_red_turns(items, *, version: str = NAIVE_RED_VERSION) -> set[int]:
    """Return the Stage-1 red-turn set. This does not decide hurdle status."""
    if items is None:
        return set()
    turns: set[int] = set()
    for item in items:
        if not isinstance(item, dict) or not is_naive_red_turn(item, version=version):
            continue
        turn = _turn_number(item)
        if turn is not None:
            turns.add(turn)
    return turns


detect_repeat_loop_without_new_evidence = retired_noop
detect_tool_error_burst_without_repair = retired_noop
detect_no_material_source_contact = retired_noop
detect_done_with_zero_source_mutations = retired_noop
detect_done_without_passing_verification_execution = retired_noop
detect_observed_test_failure_unrepaired_at_finish = retired_noop


def build_signal_detector_registry(
    rule_catalog_path: str | Path | None = None,
    detector_version: str = DETECTOR_VERSION,
) -> dict:
    """Build a live detector registry from an explicit rule catalog path.

    No path means no rules. This keeps code/config separated: importing the
    module never silently reaches into a study artifact or a generated file.
    """
    if not rule_catalog_path:
        return {}
    return build_registry(load_rule_catalog(rule_catalog_path), detector_version)


# Empty live registry: adaptive controller plumbing remains, but no detector
# can produce active_confirmed until a checked rule is explicitly wired.
SIGNAL_DETECTORS = build_signal_detector_registry()
