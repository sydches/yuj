"""Tests for adaptive-control TOML overlays.

The live apply surface is one generic config-overlay path. It starts from the
baseline config, appends the selected candidate TOML, and
replaces session.cfg. It does not use one-field executors.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

from llm_solver.config import load_config
from llm_solver.harness._guardrails.state import init_guardrail_state
from llm_solver.harness.adaptive_control import executors
from llm_solver.harness.adaptive_control.schema import InterventionPayload
from llm_solver.harness.context_strategies import HalfLifeContext


def _write(path, text: str) -> str:
    path.write_text(text.strip() + "\n")
    return str(path)


def _session(tmp_path, baseline_path):
    cfg = load_config(user_config=baseline_path)
    return SimpleNamespace(
        cfg=cfg,
        adaptive_control_resolved_baseline_cfg=cfg,
        adaptive_control_baseline_config_paths=(str(baseline_path),),
    )


def _payload(candidate_path):
    return InterventionPayload(
        "toml_overlay.apply::loop.loop_detect_enabled",
        executors.TOML_OVERLAY_EXECUTOR_ID,
        "loop_reactive",
        candidate_config_path=str(candidate_path),
    )


def test_old_one_field_executor_is_missing():
    status, reason = executors.diagnose_apply("loop_control.set_field")
    assert status == "blocked"
    assert reason == "missing_executor"


def test_missing_baseline_blocks(tmp_path):
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        loop_detect_enabled = true
    """)
    session = SimpleNamespace(cfg=load_config())
    res = executors.apply(session, _payload(candidate))
    assert res.applied is False
    assert res.blocked_reason == "missing_baseline_config"


def test_toml_overlay_apply_changes_cfg(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        loop_detect_enabled = false
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        loop_detect_enabled = true
    """)
    session = _session(tmp_path, baseline)
    assert session.cfg.loop_detect_enabled is False

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert res.active_config_basis == "baseline_plus_candidate"
    assert res.baseline_config_paths == (baseline,)
    assert res.candidate_config_path == candidate
    assert session.cfg.loop_detect_enabled is True
    assert res.pre_digest != res.post_digest


def test_toml_overlay_restore_baseline_removes_candidate(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        loop_detect_enabled = false

        [output]
        compound_selective_trace_test_anchor_lines = 0
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        loop_detect_enabled = true

        [output]
        compound_selective_trace_test_anchor_lines = 4
    """)
    session = _session(tmp_path, baseline)

    applied = executors.apply(session, _payload(candidate))
    restored = executors.restore_baseline(session)

    assert applied.applied is True
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.compound_selective_trace_test_anchor_lines == 0
    assert restored.applied is True
    assert restored.executor_id == executors.TOML_OVERLAY_RESTORE_EXECUTOR_ID
    assert restored.active_config_basis == "baseline"
    assert restored.baseline_config_paths == (baseline,)
    assert restored.candidate_config_path == ""
    assert restored.applied_config_paths == (baseline,)
    assert restored.pre_digest != restored.post_digest


def test_toml_overlay_preserves_launch_model_override(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        loop_detect_enabled = false
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        loop_detect_enabled = true
    """)
    session = _session(tmp_path, baseline)
    session.cfg = dataclasses.replace(session.cfg, model="qwen3.6-35b-a3b")

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert session.cfg.model == "qwen3.6-35b-a3b"
    assert session.cfg.loop_detect_enabled is True
    assert "session.cfg" in res.refreshed_surfaces


def test_toml_overlay_preserves_start_only_scaffold_fields(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        max_turns = 60
        max_sessions = 1
        loop_detect_enabled = false
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        max_turns = 250
        max_sessions = 9
        loop_detect_enabled = true
    """)
    session = _session(tmp_path, baseline)

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert session.cfg.loop_detect_enabled is True
    assert session.cfg.max_turns == 60
    assert session.cfg.max_sessions == 1
    assert "max_turns" not in res.changed_config_fields
    assert "max_sessions" not in res.changed_config_fields


def test_dynamic_cfg_fields_do_not_require_context_attrs(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        loop_detect_enabled = false

        [output]
        max_output_chars = 900
        recent_tool_results_chars = 700
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        loop_detect_enabled = true

        [output]
        max_output_chars = 1000
        recent_tool_results_chars = 800
    """)
    session = _session(tmp_path, baseline)
    session.context = SimpleNamespace()

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert res.blocked_reason == ""
    assert res.changed_config_fields == (
        "loop_detect_enabled",
        "max_output_chars",
        "recent_tool_results_chars",
    )
    assert res.blocked_config_fields == ()
    assert session.cfg.loop_detect_enabled is True
    assert session.cfg.max_output_chars == 1000
    assert session.cfg.recent_tool_results_chars == 800
    assert "session.cfg" in res.refreshed_surfaces


def test_copied_context_field_refreshes_existing_attr(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [output]
        recent_tool_results_chars = 700
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [output]
        recent_tool_results_chars = 800
    """)
    session = _session(tmp_path, baseline)
    session.context = SimpleNamespace(
        _recent_tool_results_chars=700,
        _msg_cache=["stale"],
    )

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert session.cfg.recent_tool_results_chars == 800
    assert session.context._recent_tool_results_chars == 800
    assert session.context._msg_cache is None
    assert "context._recent_tool_results_chars" in res.refreshed_surfaces
    assert "session.cfg" in res.refreshed_surfaces


def test_failed_surface_validation_does_not_mutate_context(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [experiment]
        tool_desc = "yuj"

        [prompts]
        state_context_suffix = "OLD SUFFIX"
    """)
    session = _session(tmp_path, baseline)
    session.context = SimpleNamespace(
        _suffix="OLD SUFFIX",
        _msg_cache=["still valid"],
    )
    session.client = SimpleNamespace()
    session._tool_registry = SimpleNamespace()
    new_cfg = dataclasses.replace(
        session.cfg,
        tool_desc="opencode",
        state_context_suffix="NEW SUFFIX",
    )
    changed = {"tool_desc", "state_context_suffix"}

    with patch(
        "llm_solver.harness._loop.profile_resolution.apply_profile_to_schemas",
        side_effect=RuntimeError("schema rebuild failed"),
    ):
        ok, reason, refreshed, blocked = executors._refresh_runtime_surfaces(
            session,
            session.cfg,
            new_cfg,
            changed,
        )

    assert ok is False
    assert reason == "runtime_surface_refresh_failed"
    assert refreshed == ()
    assert blocked == ("state_context_suffix", "tool_desc")
    assert session.context._suffix == "OLD SUFFIX"
    assert session.context._msg_cache == ["still valid"]


def test_stateful_suffix_overlay_does_not_change_halflife_prompt(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [prompts]
        state_context_suffix = "OLD SUFFIX"
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [prompts]
        state_context_suffix = "NEW POST-RED LEDGER SUFFIX"
    """)
    session = _session(tmp_path, baseline)
    session.context = HalfLifeContext(context_size=100_000)
    session.context.add_user("TASK")
    assert session.context.get_messages()[-1]["content"] == "TASK"

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert session.cfg.state_context_suffix == "NEW POST-RED LEDGER SUFFIX"
    assert not hasattr(session.context, "_suffix")
    assert "context._suffix" not in res.refreshed_surfaces
    assert "session.cfg" in res.refreshed_surfaces
    assert session.context.get_messages()[-1]["content"] == "TASK"


def test_toml_overlay_refreshes_guardrail_derived_thresholds(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        duplicate_abort = 0
        rumination_enabled = false
        rumination_min_threshold = 0
        rumination_nudge_threshold = 0
        rumination_gate_arm_threshold = 0
        rumination_nudge_threshold_abs = 0
        rumination_gate_arm_threshold_abs = 0
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        duplicate_abort = 80
        rumination_enabled = true
        rumination_min_threshold = 0
        rumination_nudge_threshold_abs = 12
        rumination_gate_arm_threshold_abs = 999
    """)
    session = _session(tmp_path, baseline)
    session._guards = init_guardrail_state(session.cfg)
    session._guards.recent_calls.append(("old-call",))
    session._guards.non_write_calls_since_write = 7
    session._guards.same_target_count = 3

    assert session._guards.rumination_nudge_threshold == 0
    assert session._guards.rumination_arm_threshold == 0
    assert session._guards.recent_calls.maxlen == 1

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert session.cfg.rumination_enabled is True
    assert session._guards.rumination_nudge_threshold == 12
    assert session._guards.rumination_nudge_threshold_post_mutation == 12
    assert session._guards.rumination_arm_threshold == 999
    assert session._guards.recent_calls.maxlen == 80
    assert list(session._guards.recent_calls) == [("old-call",)]
    assert session._guards.non_write_calls_since_write == 7
    assert session._guards.same_target_count == 3
    assert "guard_state" in res.refreshed_surfaces
    assert "session.cfg" in res.refreshed_surfaces


def test_second_apply_is_non_stacking(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        loop_detect_enabled = false
        loop_detect_threshold = 5
    """)
    candidate_a = _write(tmp_path / "candidate_a.toml", """
        [loop]
        loop_detect_enabled = true
    """)
    candidate_b = _write(tmp_path / "candidate_b.toml", """
        [loop]
        loop_detect_threshold = 9
    """)
    session = _session(tmp_path, baseline)

    res_a = executors.apply(session, _payload(candidate_a))
    assert res_a.applied is True
    assert session.cfg.loop_detect_enabled is True

    res_b = executors.apply(session, _payload(candidate_b))
    assert res_b.applied is True
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.loop_detect_threshold == 9
    assert res_b.applied_config_paths == (baseline, candidate_b)


def test_sequential_midrun_switches_are_baseline_plus_current_candidate(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        loop_detect_enabled = false
        loop_detect_threshold = 5
        require_intent = false
        intent_abort_threshold = 0

        [output]
        solver_trace_lines = 12
        recent_tool_results_chars = 700
    """)
    k1 = _write(tmp_path / "k1.toml", """
        [loop]
        loop_detect_enabled = true
    """)
    k2 = _write(tmp_path / "k2.toml", """
        [output]
        solver_trace_lines = 24
        recent_tool_results_chars = 800
    """)
    k3 = _write(tmp_path / "k3.toml", """
        [loop]
        require_intent = true
        intent_abort_threshold = 3
    """)
    session = _session(tmp_path, baseline)
    session.context = SimpleNamespace(
        _trace_lines=12,
        _recent_tool_results_chars=700,
        _msg_cache=["stale"],
    )

    res_7 = executors.apply(session, _payload(k1))
    assert res_7.applied is True
    assert session.cfg.loop_detect_enabled is True
    assert session.cfg.solver_trace_lines == 12
    assert session.cfg.require_intent is False

    res_11 = executors.apply(session, _payload(k2))
    assert res_11.applied is True
    # k1 must not stack into the second switch.
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.solver_trace_lines == 24
    assert session.context._trace_lines == 24
    assert session.context._recent_tool_results_chars == 800
    assert session.context._msg_cache is None
    assert session.cfg.require_intent is False
    assert res_11.applied_config_paths == (baseline, k2)

    res_16 = executors.apply(session, _payload(k3))
    assert res_16.applied is True
    # k2 must not stack into the third switch.
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.solver_trace_lines == 12
    assert session.context._trace_lines == 12
    assert session.cfg.require_intent is True
    assert session.cfg.intent_abort_threshold == 3
    assert res_16.applied_config_paths == (baseline, k3)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_old_one_field_executor_is_missing()
        test_missing_baseline_blocks(p)
        test_toml_overlay_apply_changes_cfg(p)
        test_second_apply_is_non_stacking(p)
    print("TOML overlay apply tests passed")
