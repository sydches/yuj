from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from llm_solver.config import load_config
from llm_solver.harness.adaptive_control import executors
from llm_solver.harness.adaptive_control.schema import InterventionPayload


def _write(path, text: str) -> str:
    path.write_text(text.strip() + "\n")
    return str(path)


def _session(baseline_path):
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


def test_candidate_delta_keeps_resolved_runtime_budgets(tmp_path):
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
    """)
    session = _session(baseline)
    resolved = dataclasses.replace(
        session.cfg,
        max_output_chars=90_000,
        recent_tool_results_chars=100_000,
    )
    session.cfg = resolved
    session.adaptive_control_resolved_baseline_cfg = resolved

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert res.changed_config_fields == ("loop_detect_enabled",)
    assert session.cfg.max_output_chars == 90_000
    assert session.cfg.recent_tool_results_chars == 100_000


def test_restore_returns_exact_resolved_runtime_baseline(tmp_path):
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
    session = _session(baseline)
    resolved = dataclasses.replace(
        session.cfg,
        max_output_chars=90_000,
        recent_tool_results_chars=100_000,
    )
    session.cfg = resolved
    session.adaptive_control_resolved_baseline_cfg = resolved

    applied = executors.apply(session, _payload(candidate))
    restored = executors.restore_baseline(session)

    assert applied.applied is True
    assert restored.applied is True
    assert session.cfg == resolved


def test_apply_rebinds_client_to_committed_config(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [loop]
        loop_detect_enabled = false
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [loop]
        loop_detect_enabled = true
    """)
    session = _session(baseline)
    session.client = SimpleNamespace(cfg=session.cfg)

    res = executors.apply(session, _payload(candidate))

    assert res.applied is True
    assert session.client.cfg is session.cfg
    assert "client.cfg" in res.refreshed_surfaces


def test_skill_discovery_settings_are_startup_only(tmp_path):
    baseline = _write(tmp_path / "baseline.toml", """
        [prompts]
        skills_enabled = false
    """)
    candidate = _write(tmp_path / "candidate.toml", """
        [prompts]
        skills_enabled = true
    """)
    session = _session(baseline)

    res = executors.apply(session, _payload(candidate))

    assert res.applied is False
    assert res.blocked_reason == "config_refresh_not_declared"
    assert res.blocked_config_fields == ("skills_enabled",)
    assert session.cfg.skills_enabled is False
