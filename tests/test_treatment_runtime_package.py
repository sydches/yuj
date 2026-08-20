"""The public treatment data loads and keeps the released response order."""
import csv
from types import SimpleNamespace

from scripts.llm_solver.config import PROJECT_ROOT, load_config
from scripts.llm_solver.harness.adaptive_control import executors, lookup_runtime
from scripts.llm_solver.harness.adaptive_control.llm_detector_core import (
    load_atlas_families,
)
from scripts.llm_solver.harness.adaptive_control.schema import InterventionPayload
from scripts.llm_solver.harness.adaptive_control.trace_nets_detector import (
    evaluate_trace_nets,
)


def _repeat_event(turn: int) -> dict[str, object]:
    return {
        "event": "tool_call",
        "turn_number": turn,
        "args_summary": "same probe",
        "output_sha256": "same failing output",
        "pass_fail": "fail",
        "source_write_like": "False",
    }


def _read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_public_treatment_data_contains_only_released_runtime_fields():
    dictionary_path = (
        PROJECT_ROOT / "configs/treatment/hurdle_dictionary.trace_nets.v1.tsv"
    )
    ladder_path = PROJECT_ROOT / "configs/treatment/medicine_ladder.v1.tsv"

    dictionary_rows = _read_tsv(dictionary_path)
    assert "cell_count" not in dictionary_rows[0]
    assert {row["dictionary_version"] for row in dictionary_rows} == {
        "hurdle_dictionary_trace_nets_v1"
    }

    ladder_rows = _read_tsv(ladder_path)
    assert [row["rank_within_ladder"] for row in ladder_rows] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert {row["online_signal_id"] for row in ladder_rows} == {
        "treatment_v1_ladder"
    }


def test_treatment_runtime_resolves_selects_applies_and_restores(
    monkeypatch, tmp_path,
):
    cfg = load_config(
        user_config=PROJECT_ROOT / "configs/regimes/treatment.toml"
    )
    monkeypatch.chdir(tmp_path)

    families = load_atlas_families(
        cfg.llm_hurdle_detector_atlas_dictionary_path
    )
    assert [row["family"] for row in families] == [
        "repeat_wall",
        "reread_slump",
    ]

    events = [_repeat_event(turn) for turn in range(9, 13)]
    detector_session = SimpleNamespace(cfg=cfg, _trace_events=events)
    verdict = evaluate_trace_nets(detector_session, 12)
    assert verdict.hurdle_present == "yes"
    assert verdict.hurdle_family == "repeat_wall"

    rows = lookup_runtime.load_lookup(cfg.adaptive_control_lookup_table_path)
    assert [row["intervention_id"] for row in rows] == [
        "toml_overlay.apply::loop.loop_detect_on_default",
        "toml_overlay.apply::loop.duplicate_guard",
        "toml_overlay.apply::loop.loop_detect_recovery",
        "toml_overlay.apply::tools.unified_envelope",
        "toml_overlay.apply::loop.intent_gate_repeat",
    ]
    chosen, status, reason = lookup_runtime.select_ranked_ladder(rows)
    assert status == "selected"
    assert reason == ""
    assert chosen is not None

    for row in rows:
        session = SimpleNamespace(
            cfg=cfg,
            adaptive_control_resolved_baseline_cfg=cfg,
            adaptive_control_baseline_config_paths=(
                cfg.adaptive_control_baseline_config_paths
            ),
        )
        payload = InterventionPayload(
            intervention_id=row["intervention_id"],
            executor_id=row["runtime_executor_id"],
            timing_class=row["timing_class"],
            candidate_config_path=row["candidate_config_path"],
        )
        applied = executors.apply(session, payload)
        assert applied.applied is True
        assert applied.changed_config_fields

        restored = executors.restore_baseline(session)
        assert restored.applied is True

    assert chosen["intervention_id"].endswith("loop.loop_detect_on_default")
