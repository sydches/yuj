"""Runtime entrypoint for invoking the LLM hurdle detector during a session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import resolve_project_path
from .._loop.model_role_runtime import consumer_role_client, record_role_usage
from .llm_detector_apply import _maybe_apply_detector_intervention, _pending_watch
from .llm_detector_core import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    LLMDetectorPacket,
    LLMDetectorVerdict,
    append_detector_log,
    build_detector_log_row,
    build_detector_packet,
    parse_detector_verdict,
    render_detector_messages,
)

def maybe_run_llm_hurdle_detector(session: Any, turn: int) -> dict[str, Any] | None:
    """Run the configured no-tool LLM detector after a solver turn.

    This is fail-open: a detector error is logged when possible, but it must not
    interrupt the solver loop.
    """
    cfg = getattr(session, "cfg", None)
    if not bool(getattr(cfg, "llm_hurdle_detector_enabled", False)):
        return None
    pending = _pending_watch(session)
    cadence = max(1, int(getattr(cfg, "llm_hurdle_detector_cadence_turns", 1) or 1))
    if not pending and (int(turn) + 1) % cadence != 0:
        return None

    atlas_source = str(
        getattr(cfg, "llm_hurdle_detector_atlas_dictionary_path", "") or ""
    )
    # Stop on setup errors. A detector that cannot load its dictionary
    # must not let the session continue without detection.
    if not atlas_source:
        _log_detector_setup_error(session, int(turn), "missing_atlas_dictionary_path")
        raise RuntimeError(
            "llm_hurdle_detector: atlas_dictionary_path is empty but the "
            "detector is enabled — refusing to run a silently dead arm")
    atlas_path = resolve_project_path(atlas_source)
    if not atlas_path.is_file():
        _log_detector_setup_error(session, int(turn), f"atlas_dictionary_not_found:{atlas_source}")
        raise RuntimeError(
            f"llm_hurdle_detector: atlas dictionary unreadable: {atlas_source}")

    log_path = _detector_log_path(session)
    if not log_path:
        return None

    packet = build_detector_packet(
        atlas_dictionary_path=atlas_source,
        input_contract_path=str(getattr(cfg, "llm_hurdle_detector_input_contract_path", "") or ""),
        trace_events=list(getattr(session, "_trace_events", []) or []),
        observation_turn=int(turn),
        evidence_regime=str(getattr(cfg, "adaptive_control_evidence_regime", "causal_live") or "causal_live"),
        config_state=_config_state(cfg),
        max_trace_events=int(getattr(cfg, "llm_hurdle_detector_max_trace_events", 80) or 80),
        max_field_chars=int(getattr(cfg, "llm_hurdle_detector_max_field_chars", 800) or 800),
        max_state_snapshots=int(getattr(cfg, "llm_hurdle_detector_max_state_snapshots", 24) or 24),
        prompt_version=str(getattr(cfg, "llm_hurdle_detector_prompt_version", PROMPT_VERSION) or PROMPT_VERSION),
    )

    messages = render_detector_messages(packet)
    raw_response = ""
    routed = None
    backend = str(getattr(cfg, "llm_hurdle_detector_backend", "llm") or "llm")
    try:
        if backend == "trace_nets":
            # mechanical soft-tier nets, in-process: no model call at all
            from .trace_nets_detector import evaluate_trace_nets
            verdict = evaluate_trace_nets(session, int(turn))
            raw_response = "trace_nets_backend"
        else:
            routed = consumer_role_client(session, "weak")
            if routed.resolution is None:
                # Preserve the injectable legacy test/replay client contract.
                result = routed.client.chat(messages, [], turn=int(turn))
                if getattr(result, "tool_calls", None):
                    raise ValueError(
                        "detector returned tool calls despite no tools being provided"
                    )
                raw_response = str(getattr(result, "content", "") or "")
            else:
                role_cfg = getattr(routed.client, "cfg", cfg)
                response = routed.client.complete_side_request({
                    "model": getattr(role_cfg, "model", cfg.model),
                    "messages": messages,
                    "max_tokens": max(
                        1, int(getattr(role_cfg, "max_tokens", 1024) or 1024)
                    ),
                })
                record_role_usage(session, routed, response.usage)
                raw_response = response.content
            verdict = parse_detector_verdict(raw_response)
        row = build_detector_log_row(
            packet=packet,
            messages=messages,
            raw_response=raw_response,
            verdict=verdict,
        )
    except Exception as exc:  # noqa: BLE001 - detector must not break solver
        verdict = LLMDetectorVerdict(
            hurdle_present="uncertain",
            hurdle_family="",
            confidence="low",
            evidence_refs=[],
            recommended_config="",
            abstain_reason=f"detector_error:{type(exc).__name__}: {exc}",
            decision_summary="Detector call or parsing failed.",
            why_now="No valid detector verdict was available for this turn.",
            new_facts_still_appearing=None,
            rejected_families=[],
            rejected_reason="none",
            timing_basis="detector_error",
            uncertainty=f"{type(exc).__name__}: {exc}",
        )
        row = build_detector_log_row(
            packet=packet,
            messages=messages,
            raw_response=raw_response,
            verdict=verdict,
        )
        row["detector_error"] = f"{type(exc).__name__}: {exc}"
    if routed is not None:
        row.update(routed.trace_fields())
    _maybe_apply_detector_intervention(session, int(turn), verdict, row)
    append_detector_log(log_path, row)
    return row




def _detector_log_path(session: Any) -> str:
    cfg = getattr(session, "cfg", None)
    configured = str(getattr(cfg, "llm_hurdle_detector_log_path", "") or "")
    if configured:
        path = Path(configured)
        if path.is_absolute():
            return str(path)
        trace_path = getattr(session, "_trace_path", None)
        if trace_path:
            return str(Path(trace_path).parent / path)
        return configured
    trace_path = getattr(session, "_trace_path", None)
    if trace_path:
        return str(Path(trace_path).parent / "llm_hurdle_detector.jsonl")
    return ""


def _config_state(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    keys = (
        "model",
        "profile_name",
        "context_size",
        "adaptive_control_detector_mode",
        "adaptive_control_detector_version",
        "adaptive_control_enabled",
        "adaptive_control_lookup_table_path",
        "adaptive_control_candidate_config_path",
        "adaptive_control_intervention_target",
        "adaptive_control_max_interventions",
        "adaptive_control_max_same_signal_interventions",
        "adaptive_control_disallow_repeat_intervention",
        "adaptive_control_cooldown_after_apply_slots",
        "llm_hurdle_detector_prompt_version",
        "llm_hurdle_detector_cadence_turns",
        "llm_hurdle_detector_max_trace_events",
        "llm_hurdle_detector_max_field_chars",
        "llm_hurdle_detector_max_state_snapshots",
    )
    return {key: getattr(cfg, key) for key in keys if hasattr(cfg, key)}


def _log_detector_setup_error(session: Any, turn: int, reason: str) -> dict[str, Any] | None:
    log_path = _detector_log_path(session)
    if not log_path:
        return None
    cfg = getattr(session, "cfg", None)
    packet = LLMDetectorPacket(
        schema_version=SCHEMA_VERSION,
        prompt_version=str(getattr(cfg, "llm_hurdle_detector_prompt_version", PROMPT_VERSION) or PROMPT_VERSION),
        observation_turn=int(turn),
        evidence_regime=str(getattr(cfg, "adaptive_control_evidence_regime", "causal_live") or "causal_live"),
        atlas_source=str(getattr(cfg, "llm_hurdle_detector_atlas_dictionary_path", "") or ""),
        input_contract_source=str(getattr(cfg, "llm_hurdle_detector_input_contract_path", "") or ""),
        atlas_families=[],
        trace_prefix=[],
        raw_state_snapshots=[],
        config_state=_config_state(cfg),
        omissions={
            "raw_transcript_included": False,
            "future_turns_included": False,
            "post_run_artifacts_included": False,
            "setup_error": reason,
        },
    )
    verdict = LLMDetectorVerdict(
        hurdle_present="uncertain",
        hurdle_family="",
        confidence="low",
        evidence_refs=[],
        recommended_config="",
        abstain_reason=reason,
        decision_summary="Detector setup failed.",
        why_now="Detector setup failed before a live verdict could be produced.",
        new_facts_still_appearing=None,
        rejected_families=[],
        rejected_reason="none",
        timing_basis="setup_error",
        uncertainty=reason,
    )
    row = build_detector_log_row(
        packet=packet,
        messages=[],
        raw_response="",
        verdict=verdict,
    )
    row["detector_error"] = reason
    append_detector_log(log_path, row)
    return row
