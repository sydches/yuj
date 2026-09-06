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


def test_loop_activation_restores_threshold_and_baseline(tmp_path):
    from scripts.llm_solver.harness.guardrails import (
        Action, init_guardrail_state, loop_detect,
    )

    baseline = tmp_path / "disabled.toml"
    baseline.write_text("[loop]\nloop_detect_enabled = false\nloop_detect_threshold = 0\n")
    cfg = load_config(user_config=baseline)
    session = SimpleNamespace(
        cfg=cfg, adaptive_control_resolved_baseline_cfg=cfg,
        adaptive_control_baseline_config_paths=(str(baseline),),
    )
    payload = InterventionPayload(
        intervention_id="toml_overlay.apply::loop.loop_detect_on_default",
        executor_id="toml_overlay.apply", timing_class="immediate",
        candidate_config_path=str(PROJECT_ROOT / "configs/treatment/overlays/loop_detect.toml"),
    )
    applied = executors.apply(session, payload)
    assert applied.applied, applied.blocked_reason
    assert {"loop_detect_enabled", "loop_detect_threshold"} <= set(applied.changed_config_fields)
    assert session.cfg.loop_detect_enabled is True
    assert session.cfg.loop_detect_threshold == 5

    state = init_guardrail_state(session.cfg)
    sig = (("read", '{"path": "a.py"}'),)
    actions = [loop_detect(state, session.cfg, tool_calls_sig=sig).action for _ in range(6)]
    assert actions == [Action.PASS] * 4 + [Action.WARN, Action.END]
    changed = (("read", '{"path": "b.py"}'),)
    assert loop_detect(state, session.cfg, tool_calls_sig=changed).action == Action.PASS
    assert state.loop_detect_streak == 1
    assert state.loop_detect_warned is False

    assert executors.restore_baseline(session).applied
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.loop_detect_threshold == 0
    assert loop_detect(state, session.cfg, tool_calls_sig=changed).action == Action.PASS
    assert state.loop_detect_streak == 0


def _read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_duplicate_activation_warns_only_on_a_repeat(tmp_path):
    from scripts.llm_solver.harness.guardrails import Action, init_guardrail_state, duplicate_guard

    baseline = tmp_path / "disabled.toml"
    baseline.write_text("[loop]\nduplicate_guard_enabled = false\nduplicate_abort = 0\nduplicate_warn_count = 0\n")
    cfg = load_config(user_config=baseline)
    session = SimpleNamespace(cfg=cfg, _guards=init_guardrail_state(cfg), adaptive_control_resolved_baseline_cfg=cfg,
                              adaptive_control_baseline_config_paths=(str(baseline),))
    payload = InterventionPayload(
        intervention_id="toml_overlay.apply::loop.duplicate_guard",
        executor_id="toml_overlay.apply", timing_class="immediate",
        candidate_config_path=str(PROJECT_ROOT / "configs/treatment/overlays/duplicate_guard.toml"),
    )
    assert executors.apply(session, payload).applied
    state = session._guards
    assert state.recent_calls.maxlen == 2
    for sig in [("read a",), ("read b",), ("edit a",)]:
        assert duplicate_guard(state, session.cfg, tool_calls_sig=sig).action == Action.PASS
    repeated = duplicate_guard(state, session.cfg, tool_calls_sig=("edit a",))
    assert repeated.action == Action.WARN
    assert "2 identical" in repeated.text
    assert "ends" not in repeated.text and "disabled" not in repeated.text
    assert duplicate_guard(state, session.cfg, tool_calls_sig=("test a",)).action == Action.PASS
    assert executors.restore_baseline(session).applied
    assert session.cfg.duplicate_guard_enabled is False
    assert state.recent_calls.maxlen == 1


def test_duplicate_abort_keeps_its_own_window():
    from dataclasses import replace
    from scripts.llm_solver.harness.guardrails import Action, init_guardrail_state, duplicate_guard

    cfg = replace(load_config(), duplicate_guard_enabled=True, duplicate_abort=2, duplicate_warn_count=5)
    state = init_guardrail_state(cfg)
    assert duplicate_guard(state, cfg, tool_calls_sig=("old",)).action == Action.PASS
    assert duplicate_guard(state, cfg, tool_calls_sig=("new",)).action == Action.PASS
    assert duplicate_guard(state, cfg, tool_calls_sig=("new",)).action == Action.END


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
