"""Runtime lookup selection tests."""
from __future__ import annotations

import _ac_bootstrap  # noqa: F401  (stub-parent bootstrap; must precede harness import)
from llm_solver.harness.adaptive_control import lookup_runtime as lr

SIG = "repeat_loop_without_new_evidence"


def _ready(rank=1, knob="loop.x"):
    return {"online_signal_id": SIG, "lookup_status": "ready_live_lookup",
            "reactive_lookup_allowed": "true", "live_eligible": "true",
            "executor_status": "implemented", "runtime_executor_id": "toml_overlay.apply",
            "rank_within_hurdle": str(rank), "intervention_id": f"toml_overlay.apply::{knob}",
            "candidate_config_path": f"configs/{knob}.toml"}


def _family_ready(rank=1, knob="loop.x", family="loop_churn"):
    row = _ready(rank=rank, knob=knob)
    row.update({
        "detector_family": family,
        "rank_within_family": str(rank),
        "ready_for_live": "true",
        "family_lookup_status": "ready_family_lookup",
    })
    return row


def _blocked(status):
    return {"online_signal_id": SIG, "lookup_status": status,
            "reactive_lookup_allowed": "true", "live_eligible": "false",
            "executor_status": "planned", "runtime_executor_id": "",
            "rank_within_hurdle": "1", "intervention_id": "no_executor::loop.x"}


def test_ready_row_selected():
    row, status, reason = lr.select([_ready()], SIG)
    assert status == "selected" and reason == ""
    assert row["intervention_id"] == "toml_overlay.apply::loop.x"


def test_inherited_ranking_lowest_rank_wins():
    row, status, _ = lr.select([_ready(rank=3, knob="b"), _ready(rank=1, knob="a")], SIG)
    assert status == "selected" and row["intervention_id"].endswith("::a")


def test_not_live_eligible_blocks():
    _, status, reason = lr.select([_blocked("blocked_not_live_eligible")], SIG)
    assert status == "blocked" and reason == "medicine_not_live_eligible"


def test_planned_executor_blocks():
    _, status, reason = lr.select([_blocked("blocked_planned_executor")], SIG)
    assert status == "blocked" and reason == "executor_planned_not_implemented"


def test_ready_without_candidate_config_path_is_not_selectable():
    row = _ready()
    row["candidate_config_path"] = ""
    chosen, status, reason = lr.select([row], SIG)
    assert chosen is None
    assert status == "blocked"
    assert reason == "lookup_not_ready"


def test_no_matching_signal_is_missing_row():
    _, status, reason = lr.select([_ready()], "some_other_signal")
    assert status == "blocked" and reason == "missing_lookup_row"


def test_no_lookup_file_is_missing_row():
    assert lr.load_lookup("") is None
    assert lr.load_lookup("/nonexistent/path.tsv") is None
    _, status, reason = lr.select(lr.load_lookup(""), SIG)
    assert status == "blocked" and reason == "missing_lookup_row"


def test_v0_style_zero_ready_blocks_not_selected():
    # mirrors the current generated lookup: rows present but none ready
    rows = [_blocked("blocked_live_disposition"), _blocked("blocked_planned_executor")]
    row, status, _ = lr.select(rows, SIG)
    assert row is None and status == "blocked"


def test_family_lookup_selects_by_detector_family_rank():
    row, status, reason = lr.select_by_family([
        _family_ready(rank=3, knob="loop.c"),
        _family_ready(rank=1, knob="loop.a"),
    ], "loop_churn")
    assert status == "selected"
    assert reason == ""
    assert row["intervention_id"].endswith("::loop.a")


def test_family_lookup_escalation_skips_same_bundle_key():
    first = _family_ready(rank=1, knob="loop.a")
    first["same_bundle_key"] = "groups:loop_detect;rumination_pressure"
    near_duplicate = _family_ready(rank=2, knob="loop.b")
    near_duplicate["same_bundle_key"] = first["same_bundle_key"]
    complementary = _family_ready(rank=3, knob="loop.c")
    complementary["same_bundle_key"] = "groups:state_surface"

    row, status, reason = lr.select_by_family(
        [first, near_duplicate, complementary],
        "loop_churn",
        exclude_ids=(first["intervention_id"],),
    )

    assert status == "selected"
    assert reason == ""
    assert row["intervention_id"].endswith("::loop.c")


def test_family_lookup_same_bundle_only_exhausts_candidate():
    first = _family_ready(rank=1, knob="loop.a")
    first["same_bundle_key"] = "groups:loop_detect"
    near_duplicate = _family_ready(rank=2, knob="loop.b")
    near_duplicate["same_bundle_key"] = first["same_bundle_key"]

    row, status, reason = lr.select_by_family(
        [first, near_duplicate],
        "loop_churn",
        exclude_ids=(first["intervention_id"],),
    )

    assert row is None
    assert status == "blocked"
    assert reason == "candidate_exhausted"


def test_family_lookup_blocks_without_ready_family_row():
    row = _family_ready()
    row["ready_for_live"] = "false"
    row["lookup_status"] = "blocked_not_live_eligible"
    row["family_lookup_status"] = "blocked_no_safe_mapping"
    chosen, status, reason = lr.select_by_family([row], "loop_churn")
    assert chosen is None
    assert status == "blocked"
    assert reason == "medicine_not_live_eligible"


def test_family_lookup_missing_family_blocks():
    chosen, status, reason = lr.select_by_family([_family_ready()], "env_discovery")
    assert chosen is None
    assert status == "blocked"
    assert reason == "missing_lookup_row"


def test_ranked_ladder_ignores_detector_family_and_uses_global_rank():
    later = _family_ready(rank=2, knob="loop.later", family="family_a")
    later["rank_within_ladder"] = "2"
    first = _family_ready(rank=9, knob="loop.first", family="family_b")
    first["rank_within_ladder"] = "1"

    chosen, status, reason = lr.select_ranked_ladder([later, first])

    assert status == "selected"
    assert reason == ""
    assert chosen["intervention_id"].endswith("::loop.first")


def test_ranked_ladder_advances_by_excluding_applied_interventions():
    first = _family_ready(rank=1, knob="loop.first")
    first["rank_within_ladder"] = "1"
    second = _family_ready(rank=2, knob="loop.second")
    second["rank_within_ladder"] = "2"

    chosen, status, reason = lr.select_ranked_ladder(
        [first, second],
        exclude_ids=(first["intervention_id"],),
    )

    assert status == "selected"
    assert reason == ""
    assert chosen["intervention_id"].endswith("::loop.second")


def test_ranked_ladder_exhausts_after_every_row_was_tried():
    only = _family_ready(rank=1, knob="loop.only")
    only["rank_within_ladder"] = "1"

    chosen, status, reason = lr.select_ranked_ladder(
        [only],
        exclude_ids=(only["intervention_id"],),
    )

    assert chosen is None
    assert status == "blocked"
    assert reason == "candidate_exhausted"


def test_ranked_ladder_size_counts_distinct_ready_interventions():
    first = _family_ready(rank=1, knob="loop.first")
    duplicate = dict(first)
    duplicate["rank_within_family"] = "2"
    blocked = _family_ready(rank=3, knob="loop.blocked")
    blocked["lookup_status"] = "blocked_not_live_eligible"

    assert lr.ranked_ladder_size([first, duplicate, blocked]) == 1


if __name__ == "__main__":
    for n in list(globals()):
        if n.startswith("test_"):
            globals()[n]()
    print("Lookup-runtime tests passed")
