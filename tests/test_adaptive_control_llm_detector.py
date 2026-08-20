from __future__ import annotations

import csv
import json

import pytest
from pathlib import Path
from types import SimpleNamespace

import _ac_bootstrap  # noqa: F401
from llm_solver.config import load_config
from llm_solver.harness.adaptive_control.llm_detector import (
    EVENT_TYPE,
    LLMDetectorVerdict,
    append_detector_log,
    build_detector_log_row,
    build_detector_packet,
    maybe_run_llm_hurdle_detector,
    parse_detector_verdict,
    render_detector_messages,
    run_detector_call,
)


def _atlas(path: Path) -> Path:
    path.write_text(
        "\t".join([
            "dictionary_version",
            "family",
            "cell_count",
            "description",
            "covered_by_prior",
            "uncovered",
            "partial_or_late",
            "cells",
        ])
        + "\n"
        + "\t".join([
            "hurdle_dictionary_v3",
            "permission_wall",
            "9",
            "root-owned /testbed write denial",
            "9",
            "0",
            "0",
            "c1;c2",
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


def _llm_atlas(path: Path) -> Path:
    path.write_text(
        "\t".join([
            "dictionary_version",
            "family",
            "cell_count",
            "description",
            "minimum_evidence",
            "do_not_call_when",
            "covered_by_prior",
            "uncovered",
            "partial_or_late",
            "cells",
        ])
        + "\n"
        + "\t".join([
            "hurdle_dictionary_llm_v1",
            "loop_churn",
            "12",
            "repetition without new evidence",
            "multiple repeated probes with no new information",
            "single failed command followed by active repair",
            "12",
            "0",
            "0",
            "c1;c2",
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


def _event(turn: int, **overrides) -> dict:
    row = {
        "event": "tool_call",
        "trace_schema_version": 1,
        "session_number": 1,
        "turn_number": turn,
        "tool_name": "bash",
        "args_summary": f"cmd='turn {turn}'",
        "result_summary": f"result {turn}",
        "gate_blocked": False,
        "write_like": False,
        "source_write_like": False,
        "source_write_paths": [],
        "prompt_tokens": 100 + turn,
        "completion_tokens": 10,
        "reasoning": "must not enter detector packet",
    }
    row.update(overrides)
    return row


def _response(
    *,
    hurdle_present: str,
    hurdle_family: str = "",
    confidence: str = "low",
    evidence_refs: list[str] | None = None,
    recommended_config: str = "",
    abstain_reason: str = "",
    decision_summary: str = "test detector decision",
    why_now: str = "test turn timing basis",
    new_facts_still_appearing: bool | None = None,
    rejected_families: list[str] | None = None,
    rejected_reason: str = "none",
    timing_basis: str = "test timing basis",
    uncertainty: str = "none",
) -> dict:
    return {
        "hurdle_present": hurdle_present,
        "hurdle_family": hurdle_family,
        "confidence": confidence,
        "evidence_refs": evidence_refs or [],
        "recommended_config": recommended_config,
        "abstain_reason": abstain_reason,
        "decision_summary": decision_summary,
        "why_now": why_now,
        "new_facts_still_appearing": new_facts_still_appearing,
        "rejected_families": rejected_families or [],
        "rejected_reason": rejected_reason,
        "timing_basis": timing_basis,
        "uncertainty": uncertainty,
    }


def test_packet_is_prefix_only_and_excludes_reasoning(tmp_path: Path) -> None:
    packet = build_detector_packet(
        atlas_dictionary_path=_atlas(tmp_path / "hurdle_dictionary.v3.tsv"),
        input_contract_path="contract.tsv",
        observation_turn=2,
        trace_events=[
            {"event": "session_start", "turn_number": 0, "reasoning": "ignore"},
            _event(1),
            _event(2, result_summary="x" * 50),
            _event(3, result_summary="future must not enter"),
        ],
        max_trace_events=10,
        max_field_chars=40,
    )

    assert packet.observation_turn == 2
    assert [row["turn_number"] for row in packet.trace_prefix] == [1, 2]
    assert all("reasoning" not in row for row in packet.trace_prefix)
    assert all(row["turn_number"] <= 2 for row in packet.trace_prefix)
    assert packet.omissions["raw_transcript_included"] is False
    assert packet.omissions["future_turns_included"] is False
    assert packet.omissions["post_run_artifacts_included"] is False
    assert packet.atlas_families == [
        {
            "family": "permission_wall",
            "cell_count": 9,
            "description": "root-owned /testbed write denial",
            "covered_by_prior": "9",
            "uncovered": "0",
            "partial_or_late": "0",
        }
    ]
    assert "[truncated" in packet.trace_prefix[-1]["result_summary"]


def test_packet_preserves_llm_dictionary_fields(tmp_path: Path) -> None:
    packet = build_detector_packet(
        atlas_dictionary_path=_llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv"),
        observation_turn=1,
        trace_events=[_event(1)],
    )

    assert packet.atlas_families == [
        {
            "family": "loop_churn",
            "cell_count": 12,
            "description": "repetition without new evidence",
            "covered_by_prior": "12",
            "uncovered": "0",
            "partial_or_late": "0",
            "minimum_evidence": "multiple repeated probes with no new information",
            "do_not_call_when": "single failed command followed by active repair",
        }
    ]


def test_packet_can_limit_trace_tail_without_future_peeking(tmp_path: Path) -> None:
    packet = build_detector_packet(
        atlas_dictionary_path=_atlas(tmp_path / "hurdle_dictionary.v3.tsv"),
        observation_turn=4,
        trace_events=[_event(1), _event(2), _event(3), _event(4), _event(5)],
        max_trace_events=2,
        max_state_snapshots=2,
    )

    assert [row["turn_number"] for row in packet.trace_prefix] == [3, 4]
    assert [row["turn_number"] for row in packet.raw_state_snapshots] == ["3", "4"]
    assert packet.omissions["trace_prefix_tool_calls_total"] == 4
    assert packet.omissions["trace_prefix_older_tool_calls_omitted"] == 2


def test_prompt_renders_detector_not_solver_contract(tmp_path: Path) -> None:
    packet = build_detector_packet(
        atlas_dictionary_path=_atlas(tmp_path / "hurdle_dictionary.v3.tsv"),
        observation_turn=1,
        trace_events=[_event(1)],
    )
    messages = render_detector_messages(packet)

    assert len(messages) == 2
    assert "not a task solver" in messages[0]["content"]
    assert "Return one JSON object only" in messages[0]["content"]
    assert "<think>" in messages[0]["content"]
    assert "first character" in messages[0]["content"]
    assert "raw transcript" in messages[0]["content"]
    assert "Evidence packet:" in messages[1]["content"]
    assert "decision_summary" in messages[1]["content"]
    assert "new_facts_still_appearing" in messages[1]["content"]
    assert "rejected_families" in messages[1]["content"]
    assert "Tiny additional observations" in messages[1]["content"]
    assert "zero source" in messages[1]["content"]
    assert ".solver/state.json" in messages[1]["content"]
    assert "not unlimited" in messages[1]["content"]
    assert "permission_wall" in messages[1]["content"]


def test_verdict_parser_accepts_json_and_requires_evidence_for_yes() -> None:
    verdict = parse_detector_verdict(
        json.dumps(_response(
            hurdle_present="yes",
            hurdle_family="permission_wall",
            confidence="high",
            evidence_refs=["host_task/.trace.jsonl:T2:result_summary"],
            recommended_config="permission_wall.toml",
            new_facts_still_appearing=False,
        ))
    )

    assert verdict == LLMDetectorVerdict(
        hurdle_present="yes",
        hurdle_family="permission_wall",
        confidence="high",
        evidence_refs=["host_task/.trace.jsonl:T2:result_summary"],
        recommended_config="permission_wall.toml",
        abstain_reason="",
        decision_summary="test detector decision",
        why_now="test turn timing basis",
        new_facts_still_appearing=False,
        rejected_families=[],
        rejected_reason="none",
        timing_basis="test timing basis",
        uncertainty="none",
    )


def test_verdict_parser_accepts_local_server_think_wrapper() -> None:
    verdict = parse_detector_verdict(
        """<think>

</think>

{
  "hurdle_present": "no",
  "hurdle_family": "",
  "confidence": "high",
  "evidence_refs": [],
  "recommended_config": "",
  "abstain_reason": "no hurdle evidence yet",
  "decision_summary": "no hurdle yet",
  "why_now": "useful investigation is still in progress",
  "new_facts_still_appearing": true,
  "rejected_families": ["loop_churn"],
  "rejected_reason": "the repeated action is still producing useful facts",
  "timing_basis": "no hurdle timing basis yet",
  "uncertainty": "none"
}
"""
    )

    assert verdict.hurdle_present == "no"
    assert verdict.confidence == "high"
    assert verdict.abstain_reason == "no hurdle evidence yet"


def test_verdict_parser_ignores_evidence_json_before_final_verdict() -> None:
    verdict = parse_detector_verdict(
        """<think>
{"turn_number": "12", "tool_name": "bash"}
</think>

{
  "hurdle_present": "yes",
  "hurdle_family": "red_discipline",
  "confidence": "medium",
  "evidence_refs": ["host_task/.trace.jsonl:T12:result_summary"],
  "recommended_config": "",
  "abstain_reason": "",
  "decision_summary": "visible red was mishandled",
  "why_now": "the invalid clearance is visible at this turn",
  "new_facts_still_appearing": false,
  "rejected_families": ["loop_churn"],
  "rejected_reason": "the issue is red mishandling, not repeated probing",
  "timing_basis": "T12 result_summary shows the red and invalid clearance",
  "uncertainty": "none"
}
"""
    )

    assert verdict.hurdle_present == "yes"
    assert verdict.hurdle_family == "red_discipline"


def test_run_detector_call_and_log_row(tmp_path: Path) -> None:
    packet = build_detector_packet(
        atlas_dictionary_path=_atlas(tmp_path / "hurdle_dictionary.v3.tsv"),
        observation_turn=1,
        trace_events=[_event(1)],
    )

    def fake_model(messages):
        assert "Evidence packet:" in messages[1]["content"]
        return json.dumps(_response(
            hurdle_present="no",
            confidence="low",
            abstain_reason="no hurdle evidence yet",
            new_facts_still_appearing=True,
        ))

    messages, raw_response, verdict = run_detector_call(packet, fake_model)
    row = build_detector_log_row(
        packet=packet,
        messages=messages,
        raw_response=raw_response,
        verdict=verdict,
    )
    assert row["event_type"] == EVENT_TYPE
    assert row["packet_sha256"]
    assert row["messages_sha256"]
    assert row["verdict"]["hurdle_present"] == "no"

    log_path = tmp_path / "llm_detector.jsonl"
    append_detector_log(log_path, row)
    saved = json.loads(log_path.read_text(encoding="utf-8"))
    assert saved["event_type"] == EVENT_TYPE
    assert saved["verdict"]["abstain_reason"] == "no hurdle evidence yet"
    assert saved["verdict"]["decision_summary"] == "test detector decision"
    assert saved["verdict"]["new_facts_still_appearing"] is True


def test_llm_hurdle_detector_config_loads_from_toml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "llm_detector.toml"
    cfg_path.write_text(
        """
[llm_hurdle_detector]
enabled = true
cadence_turns = 2
atlas_dictionary_path = "atlas.tsv"
input_contract_path = "contract.tsv"
log_path = "detector.jsonl"
max_trace_events = 33
max_field_chars = 444
max_state_snapshots = 12
prompt_version = "custom_prompt_v1"
""",
        encoding="utf-8",
    )

    cfg = load_config(user_config=cfg_path)

    assert cfg.llm_hurdle_detector_enabled is True
    assert cfg.llm_hurdle_detector_cadence_turns == 2
    assert cfg.llm_hurdle_detector_atlas_dictionary_path == "atlas.tsv"
    assert cfg.llm_hurdle_detector_input_contract_path == "contract.tsv"
    assert cfg.llm_hurdle_detector_log_path == "detector.jsonl"
    assert cfg.llm_hurdle_detector_max_trace_events == 33
    assert cfg.llm_hurdle_detector_max_field_chars == 444
    assert cfg.llm_hurdle_detector_max_state_snapshots == 12
    assert cfg.llm_hurdle_detector_prompt_version == "custom_prompt_v1"


class _FakeDetectorClient:
    def __init__(self, response: dict | list[dict]):
        self.responses = list(response) if isinstance(response, list) else [response]
        self.calls = []

    def chat(self, messages, tools, turn=0):
        self.calls.append({"messages": messages, "tools": tools, "turn": turn})
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return SimpleNamespace(
            content=json.dumps(response),
            tool_calls=[],
        )


def _detector_cfg(tmp_path: Path, atlas_path: Path, *, enabled=True, cadence=1):
    return SimpleNamespace(
        model="test-model",
        profile_name="test-profile",
        context_size=8192,
        adaptive_control_evidence_regime="causal_live",
        adaptive_control_detector_mode="adaptive",
        adaptive_control_detector_version="llm_hurdle_detector_v0",
        adaptive_control_candidate_config_path="configs/interventions/permission_wall.toml",
        adaptive_control_intervention_target="toml_overlay",
        llm_hurdle_detector_enabled=enabled,
        llm_hurdle_detector_cadence_turns=cadence,
        llm_hurdle_detector_atlas_dictionary_path=str(atlas_path),
        llm_hurdle_detector_input_contract_path="contract.tsv",
        llm_hurdle_detector_log_path=str(tmp_path / "llm_detector.jsonl"),
        llm_hurdle_detector_max_trace_events=4,
        llm_hurdle_detector_max_field_chars=200,
        llm_hurdle_detector_max_state_snapshots=4,
        llm_hurdle_detector_prompt_version="test_prompt_v1",
    )


def _write_family_lookup(
    path: Path,
    *,
    candidate_config_path: Path,
    second_candidate_config_path: Path | None = None,
    third_candidate_config_path: Path | None = None,
) -> Path:
    cols = [
        "detector_family",
        "online_signal_id",
        "lookup_status",
        "reactive_lookup_allowed",
        "live_eligible",
        "ready_for_live",
        "executor_status",
        "runtime_executor_id",
        "rank_within_ladder",
        "rank_within_family",
        "rank_within_hurdle",
        "intervention_id",
        "candidate_config_path",
        "timing_class",
        "primary_knob",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "detector_family": "loop_churn",
            "online_signal_id": "repeat_loop_without_new_evidence",
            "lookup_status": "ready_live_lookup",
            "reactive_lookup_allowed": "true",
            "live_eligible": "true",
            "ready_for_live": "true",
            "executor_status": "implemented",
            "runtime_executor_id": "toml_overlay.apply",
            "rank_within_ladder": "1",
            "rank_within_family": "1",
            "rank_within_hurdle": "1",
            "intervention_id": "toml_overlay.apply::loop.loop_detect_enabled",
            "candidate_config_path": str(candidate_config_path),
            "timing_class": "loop_reactive",
            "primary_knob": "loop.loop_detect_enabled",
        })
        if second_candidate_config_path is not None:
            writer.writerow({
                "detector_family": "loop_churn",
                "online_signal_id": "repeat_loop_without_new_evidence",
                "lookup_status": "ready_live_lookup",
                "reactive_lookup_allowed": "true",
                "live_eligible": "true",
                "ready_for_live": "true",
                "executor_status": "implemented",
                "runtime_executor_id": "toml_overlay.apply",
                "rank_within_ladder": "2",
                "rank_within_family": "2",
                "rank_within_hurdle": "2",
                "intervention_id": "toml_overlay.apply::output.compound_selective_trace_test_anchor_lines",
                "candidate_config_path": str(second_candidate_config_path),
                "timing_class": "loop_reactive",
                "primary_knob": "output.compound_selective_trace_test_anchor_lines",
            })
        if third_candidate_config_path is not None:
            writer.writerow({
                "detector_family": "loop_churn",
                "online_signal_id": "repeat_loop_without_new_evidence",
                "lookup_status": "ready_live_lookup",
                "reactive_lookup_allowed": "true",
                "live_eligible": "true",
                "ready_for_live": "true",
                "executor_status": "implemented",
                "runtime_executor_id": "toml_overlay.apply",
                "rank_within_ladder": "3",
                "rank_within_family": "3",
                "rank_within_hurdle": "3",
                "intervention_id": "toml_overlay.apply::output.compound_selective_trace_source_anchor_lines",
                "candidate_config_path": str(third_candidate_config_path),
                "timing_class": "loop_reactive",
                "primary_knob": "output.compound_selective_trace_source_anchor_lines",
            })
    return path


def _live_detector_cfg(
    tmp_path: Path,
    atlas_path: Path,
    lookup_path: Path,
    baseline_path: Path,
    *,
    max_interventions: int = 1,
    max_same_signal_interventions: int = 1,
    cooldown_after_apply_slots: int = 0,
    watch_window_turns: int = 5,
    cadence: int = 1,
    backend: str = "llm",
    ledger_path: str | None = None,
    detector_log_path: str | None = None,
) -> object:
    detector_log = detector_log_path or str(tmp_path / "llm_detector.jsonl")
    control_ledger = ledger_path or str(tmp_path / "adaptive_control_ledger.jsonl")
    baseline_path.write_text(f"""
[loop]
loop_detect_enabled = false

[output]
compound_selective_trace_test_anchor_lines = 0
compound_selective_trace_source_anchor_lines = 0

[adaptive_control]
enabled = true
ledger_path = "{control_ledger}"
model = "llm_detector_oscillation"
detector_backend = "{backend}"
lookup_table_path = "{lookup_path}"
baseline_config_paths = ["{baseline_path}"]
max_interventions = {max_interventions}
max_same_signal_interventions = {max_same_signal_interventions}
max_interventions_per_attempt = {max_interventions}
max_interventions_per_hurdle_episode = {max_same_signal_interventions}
max_distinct_hurdle_episodes_per_attempt = {max_interventions}
disallow_repeat_intervention = true
cooldown_after_apply_slots = {cooldown_after_apply_slots}
watch_window_turns = {watch_window_turns}

[llm_hurdle_detector]
enabled = true
cadence_turns = {cadence}
atlas_dictionary_path = "{atlas_path}"
input_contract_path = "contract.tsv"
log_path = "{detector_log}"
max_trace_events = 4
max_field_chars = 200
max_state_snapshots = 4
prompt_version = "test_prompt_v1"
""".strip() + "\n", encoding="utf-8")
    return load_config(user_config=baseline_path)


def test_live_hook_calls_detector_with_no_tools_and_logs_verdict(tmp_path: Path) -> None:
    atlas_path = _atlas(tmp_path / "hurdle_dictionary.v3.tsv")
    client = _FakeDetectorClient(_response(
        hurdle_present="yes",
        hurdle_family="permission_wall",
        confidence="high",
        evidence_refs=["host_task/.trace.jsonl:T1:result_summary"],
        recommended_config="configs/interventions/permission_wall.toml",
        decision_summary="write wall is visible",
        why_now="T1 contains the permission denial",
        new_facts_still_appearing=False,
        rejected_families=["env_discovery"],
        rejected_reason="the evidence is a source write denial, not interpreter discovery",
        timing_basis="T1 result_summary",
        uncertainty="none",
    ))
    session = SimpleNamespace(
        cfg=_detector_cfg(tmp_path, atlas_path),
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=[
            _event(0),
            _event(1, result_summary="Permission denied"),
            _event(2, result_summary="future"),
        ],
    )

    row = maybe_run_llm_hurdle_detector(session, turn=1)

    assert row is not None
    assert client.calls and client.calls[0]["tools"] == []
    assert client.calls[0]["turn"] == 1
    assert row["verdict"]["hurdle_present"] == "yes"
    assert row["verdict"]["rejected_families"] == ["env_discovery"]
    assert row["packet"]["prompt_version"] == "test_prompt_v1"
    assert [item["turn_number"] for item in row["packet"]["trace_prefix"]] == [0, 1]
    saved = json.loads(Path(session.cfg.llm_hurdle_detector_log_path).read_text(encoding="utf-8"))
    assert saved["verdict"]["recommended_config"] == "configs/interventions/permission_wall.toml"
    assert saved["verdict"]["why_now"] == "T1 contains the permission denial"


def test_live_hook_applies_family_lookup_intervention_when_enabled(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("""
[loop]
loop_detect_enabled = true
""".strip() + "\n", encoding="utf-8")
    lookup_path = _write_family_lookup(tmp_path / "family_lookup.tsv", candidate_config_path=candidate)
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(tmp_path, atlas_path, lookup_path, baseline_path)
    assert cfg.loop_detect_enabled is False

    client = _FakeDetectorClient(_response(
        hurdle_present="yes",
        hurdle_family="loop_churn",
        confidence="high",
        evidence_refs=["host_task/.trace.jsonl:T1:result_summary"],
        decision_summary="loop churn is visible",
        why_now="T1 repeats the same dead frame",
        new_facts_still_appearing=False,
        timing_basis="T1 result_summary",
        uncertainty="none",
    ))
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=[_event(0), _event(1, result_summary="same probe again")],
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    row = maybe_run_llm_hurdle_detector(session, turn=1)

    assert row["intervention_selection"]["selection_status"] == "selected"
    assert row["intervention_selection"]["selected_intervention_id"] == "toml_overlay.apply::loop.loop_detect_enabled"
    assert row["intervention_apply"]["apply_status"] == "applied"
    assert session.cfg.loop_detect_enabled is True

    saved = json.loads(Path(cfg.llm_hurdle_detector_log_path).read_text(encoding="utf-8"))
    assert saved["intervention_selection"]["detector_family"] == "loop_churn"
    assert saved["intervention_apply"]["apply_status"] == "applied"

    ledger_rows = [
        json.loads(line)
        for line in Path(cfg.adaptive_control_ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_rows) == 1
    assert ledger_rows[0]["active_hurdle_mode"] == "loop_churn"
    assert ledger_rows[0]["apply_status"] == "applied"
    assert ledger_rows[0]["candidate_config_path"] == str(candidate)


def test_relative_control_logs_resolve_beside_trace(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("[loop]\nloop_detect_enabled = true\n", encoding="utf-8")
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        ledger_path="adaptive_control_ledger.jsonl",
        detector_log_path="llm_hurdle_detector.jsonl",
    )
    trace_dir = tmp_path / "cell" / "host_task"
    trace_dir.mkdir(parents=True)
    client = _FakeDetectorClient(_response(
        hurdle_present="yes",
        hurdle_family="loop_churn",
        confidence="high",
        evidence_refs=["T1"],
    ))
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=trace_dir / ".trace.jsonl",
        _trace_events=[_event(0), _event(1)],
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="repo__task-1:session1",
        instance_id="repo__task-1",
    )

    maybe_run_llm_hurdle_detector(session, turn=1)

    detector_log = trace_dir / "llm_hurdle_detector.jsonl"
    control_ledger = trace_dir / "adaptive_control_ledger.jsonl"
    assert detector_log.is_file()
    assert control_ledger.is_file()
    ledger_row = json.loads(control_ledger.read_text(encoding="utf-8"))
    assert ledger_row["instance_id"] == "repo__task-1"
    assert ledger_row["attempt_id"] == "repo__task-1:session1"


def test_live_hook_restores_baseline_as_soon_as_hurdle_clears(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("""
[loop]
loop_detect_enabled = true
""".strip() + "\n", encoding="utf-8")
    unused_candidate = tmp_path / "unused_candidate.toml"
    unused_candidate.write_text(
        "[output]\ncompound_selective_trace_test_anchor_lines = 1\n",
        encoding="utf-8",
    )
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate,
        second_candidate_config_path=unused_candidate,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        max_interventions=2,
        max_same_signal_interventions=2,
        cadence=25,
    )
    client = _FakeDetectorClient([
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["host_task/.trace.jsonl:T1:result_summary"],
            decision_summary="loop churn is visible",
            why_now="T1 repeats the same dead frame",
            new_facts_still_appearing=False,
            timing_basis="T1 result_summary",
            uncertainty="none",
        ),
        _response(
            hurdle_present="no",
            confidence="high",
            abstain_reason="the loop cleared after intervention",
            decision_summary="no active hurdle remains",
            why_now="the post-intervention prefix shows progress",
            new_facts_still_appearing=True,
            timing_basis="T25 detector packet",
            uncertainty="none",
        ),
    ])
    events = [_event(i, result_summary="same probe again") for i in range(30)]
    events[25].update(
        write_like=True,
        source_write_like=True,
        source_write_paths=["src/module.py"],
    )
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=events,
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    first = maybe_run_llm_hurdle_detector(session, turn=24)
    assert first["intervention_apply"]["apply_status"] == "applied"
    assert session.cfg.loop_detect_enabled is True

    cleared = maybe_run_llm_hurdle_detector(session, turn=25)

    assert cleared["watch_transition"]["episode_transition"] == "cleared_to_progress"
    assert cleared["watch_transition"]["budget_exhausted"] is False
    assert cleared["watch_transition"]["watch_status"] == "closed"
    assert [call["turn"] for call in client.calls] == [24, 25]
    assert cleared["baseline_restore"]["apply_status"] == "applied"
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.compound_selective_trace_test_anchor_lines == 0
    assert getattr(session, "_llm_detector_pending_watch") is None
    assert session._adaptive_control_episode_machine.interventions_total == 1

    ledger_rows = [
        json.loads(line)
        for line in Path(cfg.adaptive_control_ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_rows) == 2
    assert ledger_rows[0]["active_config_basis"] == "baseline_plus_candidate"
    assert ledger_rows[1]["intervention_id"] == "toml_overlay.restore_baseline"
    assert ledger_rows[1]["active_config_basis"] == "baseline"
    assert ledger_rows[1]["episode_transition"] == "cleared_to_progress"


def test_live_hook_advances_at_budget_limit_after_silence_without_progress(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate1 = tmp_path / "candidate1.toml"
    candidate1.write_text("[loop]\nloop_detect_enabled = true\n", encoding="utf-8")
    candidate2 = tmp_path / "candidate2.toml"
    candidate2.write_text(
        "[output]\ncompound_selective_trace_test_anchor_lines = 1\n",
        encoding="utf-8",
    )
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate1,
        second_candidate_config_path=candidate2,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        max_interventions=1,
        max_same_signal_interventions=1,
        cadence=25,
    )
    client = _FakeDetectorClient([
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["T24"],
        ),
        _response(
            hurdle_present="no",
            confidence="high",
            abstain_reason="the narrow symptom is quiet",
        ),
    ])
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=[_event(i, result_summary=f"probe {i}") for i in range(55)],
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    first = maybe_run_llm_hurdle_detector(session, turn=24)
    early = maybe_run_llm_hurdle_detector(session, turn=25)
    quiet = maybe_run_llm_hurdle_detector(session, turn=29)

    assert first["intervention_selection"]["rank_within_ladder"] == "1"
    assert early["watch_transition"]["watch_status"] == "continuing"
    assert early["watch_transition"]["budget_exhausted"] is False
    assert quiet["watch_transition"]["episode_transition"] == "dislodged_no_progress"
    assert quiet["watch_transition"]["budget_exhausted"] is True
    assert "baseline_restore" not in quiet
    assert quiet["intervention_selection"]["rank_within_ladder"] == "2"
    assert quiet["intervention_selection"]["excluded_intervention_ids"]
    assert quiet["intervention_apply"]["apply_status"] == "applied"
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.compound_selective_trace_test_anchor_lines == 1
    machine = session._adaptive_control_episode_machine
    assert machine.episodes_opened == 1
    assert machine.current.attempt_index == 2
    ledger_rows = [
        json.loads(line)
        for line in Path(cfg.adaptive_control_ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["intervention_id"] for row in ledger_rows] == [
        "toml_overlay.apply::loop.loop_detect_enabled",
        "toml_overlay.apply::output.compound_selective_trace_test_anchor_lines",
    ]
    assert all(row["active_config_basis"] == "baseline_plus_candidate" for row in ledger_rows)


def test_live_hook_walks_each_rank_once_then_exhausts_on_baseline(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate1 = tmp_path / "candidate1.toml"
    candidate1.write_text("[loop]\nloop_detect_enabled = true\n", encoding="utf-8")
    candidate2 = tmp_path / "candidate2.toml"
    candidate2.write_text(
        "[output]\ncompound_selective_trace_test_anchor_lines = 1\n",
        encoding="utf-8",
    )
    candidate3 = tmp_path / "candidate3.toml"
    candidate3.write_text(
        "[output]\ncompound_selective_trace_source_anchor_lines = 1\n",
        encoding="utf-8",
    )
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate1,
        second_candidate_config_path=candidate2,
        third_candidate_config_path=candidate3,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        max_interventions=1,
        max_same_signal_interventions=1,
        cadence=25,
    )
    client = _FakeDetectorClient([
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["T24"],
        ),
        _response(hurdle_present="no", confidence="high", abstain_reason="quiet"),
        _response(hurdle_present="no", confidence="high", abstain_reason="quiet"),
        _response(hurdle_present="no", confidence="high", abstain_reason="quiet"),
    ])
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=[_event(i, result_summary=f"probe {i}") for i in range(40)],
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    first = maybe_run_llm_hurdle_detector(session, turn=24)
    second = maybe_run_llm_hurdle_detector(session, turn=29)
    third = maybe_run_llm_hurdle_detector(session, turn=34)
    exhausted = maybe_run_llm_hurdle_detector(session, turn=39)

    assert first["intervention_selection"]["rank_within_ladder"] == "1"
    assert second["intervention_selection"]["rank_within_ladder"] == "2"
    assert third["intervention_selection"]["rank_within_ladder"] == "3"
    assert exhausted["intervention_selection"]["selection_blocked_reason"] == "candidate_exhausted"
    assert exhausted["intervention_apply"]["blocked_reason"] == "candidate_exhausted"
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.compound_selective_trace_test_anchor_lines == 0
    assert session.cfg.compound_selective_trace_source_anchor_lines == 0
    machine = session._adaptive_control_episode_machine
    assert machine.episodes_opened == 1
    assert machine.interventions_total == 3
    assert machine.current is None
    assert machine.episodes[0].status == "exhausted"
    assert machine.exhausted_slot == 39

    ledger_rows = [
        json.loads(line)
        for line in Path(cfg.adaptive_control_ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    applied = [row for row in ledger_rows if row["apply_status"] == "applied"]
    ladder_applies = [
        row for row in applied
        if row["intervention_id"] != "toml_overlay.restore_baseline"
    ]
    assert [row["candidate_rank_at_selection"] for row in ladder_applies] == ["1", "2", "3"]
    assert all(row["active_config_basis"] == "baseline_plus_candidate" for row in ladder_applies)
    restore_rows = [
        row for row in ledger_rows
        if row["intervention_id"] == "toml_overlay.restore_baseline"
    ]
    assert len(restore_rows) == 1
    assert restore_rows[0]["active_config_basis"] == "baseline"
    assert restore_rows[0]["immediate_effect"] == "baseline_restored_after_exhaustion"


def test_live_hook_records_clean_exhaustion_with_multiple_episode_allowance(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("[loop]\nloop_detect_enabled = true\n", encoding="utf-8")
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        max_interventions=3,
        max_same_signal_interventions=1,
        cadence=1,
    )
    client = _FakeDetectorClient([
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["T24"],
        ),
        _response(hurdle_present="no", confidence="high", abstain_reason="quiet"),
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["T30"],
        ),
        _response(
            hurdle_present="no",
            confidence="high",
            abstain_reason="task progress without a new hurdle",
        ),
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["T32"],
        ),
    ])
    events = [_event(i, result_summary=f"probe {i}") for i in range(33)]
    events[31].update(
        write_like=True,
        source_write_like=True,
        source_write_paths=["src/module.py"],
    )
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=events,
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    maybe_run_llm_hurdle_detector(session, turn=24)
    exhausted = maybe_run_llm_hurdle_detector(session, turn=29)
    later = maybe_run_llm_hurdle_detector(session, turn=30)
    progress = maybe_run_llm_hurdle_detector(session, turn=31)
    resumed = maybe_run_llm_hurdle_detector(session, turn=32)

    assert exhausted["intervention_apply"]["blocked_reason"] == "candidate_exhausted"
    assert "selected_intervention_id" not in exhausted["intervention_selection"]
    assert later["intervention_selection"] == {
        "selection_status": "not_attempted",
        "selection_blocked_reason": "candidate_exhausted",
    }
    assert "intervention_apply" not in later
    assert progress["verdict"]["hurdle_present"] == "no"
    assert "intervention_selection" not in progress
    assert resumed["episode_resume_after_progress"] == {
        "prior_exhausted_slot": 29,
        "material_progress_refs": ["turn=31"],
    }
    assert resumed["intervention_selection"]["rank_within_ladder"] == "1"
    assert resumed["intervention_apply"]["apply_status"] == "applied"
    assert session._adaptive_control_episode_machine.episodes_opened == 2


def test_live_hook_treats_family_flip_without_progress_as_same_episode(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate1 = tmp_path / "candidate1.toml"
    candidate1.write_text("[loop]\nloop_detect_enabled = true\n", encoding="utf-8")
    candidate2 = tmp_path / "candidate2.toml"
    candidate2.write_text(
        "[output]\ncompound_selective_trace_test_anchor_lines = 1\n",
        encoding="utf-8",
    )
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate1,
        second_candidate_config_path=candidate2,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        max_interventions=2,
        max_same_signal_interventions=2,
        cadence=25,
    )
    client = _FakeDetectorClient([
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["T24"],
        ),
        _response(
            hurdle_present="yes",
            hurdle_family="reread_slump",
            confidence="high",
            evidence_refs=["T29"],
        ),
    ])
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=[_event(i, result_summary=f"probe {i}") for i in range(30)],
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    maybe_run_llm_hurdle_detector(session, turn=24)
    flipped = maybe_run_llm_hurdle_detector(session, turn=29)

    assert flipped["watch_transition"]["episode_transition"] == "unchanged"
    assert flipped["watch_transition"]["verdict_family"] == "reread_slump"
    assert flipped["intervention_selection"]["detector_family"] == "loop_churn"
    assert flipped["intervention_selection"]["rank_within_family"] == "2"
    assert session._adaptive_control_episode_machine.episodes_opened == 1


def test_live_hook_advances_when_unlock_is_unproven_at_budget_limit(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate1 = tmp_path / "candidate1.toml"
    candidate1.write_text("[loop]\nloop_detect_enabled = true\n", encoding="utf-8")
    candidate2 = tmp_path / "candidate2.toml"
    candidate2.write_text(
        "[output]\ncompound_selective_trace_test_anchor_lines = 1\n",
        encoding="utf-8",
    )
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate1,
        second_candidate_config_path=candidate2,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        max_interventions=2,
        max_same_signal_interventions=2,
        cadence=25,
    )
    client = _FakeDetectorClient([
        _response(
            hurdle_present="yes",
            hurdle_family="loop_churn",
            confidence="high",
            evidence_refs=["T24"],
        ),
        _response(
            hurdle_present="uncertain",
            confidence="low",
            abstain_reason="cannot prove clearance",
        ),
    ])
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=[_event(i) for i in range(30)],
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    maybe_run_llm_hurdle_detector(session, turn=24)
    advanced = maybe_run_llm_hurdle_detector(session, turn=29)

    assert advanced["watch_transition"]["budget_exhausted"] is True
    assert advanced["watch_transition"]["episode_transition"] == "unchanged"
    assert "extended_watch_window_end" not in advanced["watch_transition"]
    assert advanced["intervention_selection"]["rank_within_ladder"] == "2"
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.compound_selective_trace_test_anchor_lines == 1


def test_live_hook_escalates_same_hurdle_after_watch_window(tmp_path: Path) -> None:
    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate1 = tmp_path / "candidate1.toml"
    candidate1.write_text("""
[loop]
loop_detect_enabled = true
""".strip() + "\n", encoding="utf-8")
    candidate2 = tmp_path / "candidate2.toml"
    candidate2.write_text("""
[output]
compound_selective_trace_test_anchor_lines = 1
""".strip() + "\n", encoding="utf-8")
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate1,
        second_candidate_config_path=candidate2,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        max_interventions=2,
        max_same_signal_interventions=2,
        cooldown_after_apply_slots=5,
        cadence=25,
    )
    client = _FakeDetectorClient(_response(
        hurdle_present="yes",
        hurdle_family="loop_churn",
        confidence="high",
        evidence_refs=["host_task/.trace.jsonl:T1:result_summary"],
        decision_summary="loop churn persists",
        why_now="same family is still active",
        new_facts_still_appearing=False,
        timing_basis="test timing",
        uncertainty="none",
    ))
    session = SimpleNamespace(
        cfg=cfg,
        client=client,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=[_event(i, result_summary="same probe again") for i in range(30)],
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    first = maybe_run_llm_hurdle_detector(session, turn=24)
    assert session.cfg.loop_detect_enabled is True
    second = maybe_run_llm_hurdle_detector(session, turn=25)
    assert session.cfg.loop_detect_enabled is True
    third = maybe_run_llm_hurdle_detector(session, turn=29)

    assert first["intervention_apply"]["apply_status"] == "applied"
    assert first["intervention_selection"]["rank_within_family"] == "1"
    assert second["watch_transition"]["watch_status"] == "continuing"
    assert second["watch_transition"]["episode_transition"] == ""
    assert "intervention_apply" not in second
    assert [call["turn"] for call in client.calls] == [24, 25, 29]
    assert third["watch_transition"]["episode_transition"] == "unchanged"
    assert third["intervention_apply"]["apply_status"] == "applied"
    assert third["intervention_selection"]["rank_within_family"] == "2"
    # Runtime TOML overlays are baseline_plus_candidate, not stacked.
    assert session.cfg.loop_detect_enabled is False
    assert session.cfg.compound_selective_trace_test_anchor_lines == 1

    ledger_rows = [
        json.loads(line)
        for line in Path(cfg.adaptive_control_ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_rows) == 2
    assert ledger_rows[0]["hurdle_episode_id"] == "attempt-1#ep1"
    assert ledger_rows[0]["watch_window_start"] == "25"
    assert ledger_rows[0]["watch_window_end"] == "29"
    assert ledger_rows[1]["episode_transition"] == "unchanged"
    assert ledger_rows[1]["same_hurdle_escalation"] == "true"
    assert ledger_rows[1]["previous_intervention_id"] == "toml_overlay.apply::loop.loop_detect_enabled"
    assert all(row["intervention_id"] != "toml_overlay.restore_baseline" for row in ledger_rows)


def test_live_hook_respects_cadence(tmp_path: Path) -> None:
    atlas_path = _atlas(tmp_path / "hurdle_dictionary.v3.tsv")
    client = _FakeDetectorClient(_response(
        hurdle_present="no",
        confidence="low",
        abstain_reason="cadence test",
        new_facts_still_appearing=True,
    ))
    cfg = _detector_cfg(tmp_path, atlas_path, cadence=2)
    session = SimpleNamespace(cfg=cfg, client=client, _trace_path=tmp_path / ".trace.jsonl", _trace_events=[_event(0)])

    assert maybe_run_llm_hurdle_detector(session, turn=0) is None
    assert client.calls == []
    assert not Path(cfg.llm_hurdle_detector_log_path).exists()


def test_live_hook_setup_error_is_fatal(tmp_path: Path) -> None:
    """A setup error stops the session instead of disabling detection."""
    missing_atlas = tmp_path / "missing.tsv"
    client = _FakeDetectorClient(_response(
        hurdle_present="no",
        confidence="low",
        abstain_reason="unused",
    ))
    cfg = _detector_cfg(tmp_path, missing_atlas)
    session = SimpleNamespace(cfg=cfg, client=client, _trace_path=tmp_path / ".trace.jsonl", _trace_events=[_event(0)])

    with pytest.raises(RuntimeError, match="unreadable"):
        maybe_run_llm_hurdle_detector(session, turn=0)
    assert client.calls == []
