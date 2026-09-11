"""Causal completed-observation facts for the local detector.

Repeated requests or equal results do not establish stalled progress. The
runtime delivers qualified observation notices without selecting a hurdle,
changing configuration or opening an intervention episode. Historical nets
remain available in trace_net_facts for offline analysis.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .llm_detector_core import LLMDetectorVerdict
from .observation_notice import repeated_observation


def evaluate_trace_nets(session: Any, turn: int) -> LLMDetectorVerdict:
    """Observe first, then apply the declared quiet-period delivery policy."""
    cfg = getattr(session, "cfg", None)
    arm_after = int(getattr(cfg, "guardrails_arm_after_turn", 0) or 0)
    active_episode = (
        getattr(cfg, "adaptive_control_delivery", "in_place") == "stop_resume"
        and getattr(session, "_llm_detector_pending_watch", None) is not None
    )
    fact = repeated_observation(session, turn)
    observed = LLMDetectorVerdict(
        hurdle_present="uncertain",
        hurdle_family="",
        confidence="high",
        evidence_refs=([f"T{fact['current_turn']}:same completed observation "
                        f"as T{fact['prior_turn']}"] if fact else []),
        abstain_reason="stalled_progress_not_established",
        decision_summary=("matching completed observation; task progress unknown"
                          if fact else "no matching completed-observation evidence"),
        new_facts_still_appearing=None,
        uncertainty="observation equality does not establish task progress",
        timing_basis="native receipts available at the current turn",
    )
    if turn <= arm_after and not active_episode:
        return replace(
            observed, abstain_reason="warmup",
            decision_summary=f"Observation during declared quiet period: {observed.decision_summary}",
            timing_basis="intervention withheld by declared quiet period",
            rejected_reason=observed.abstain_reason,
        )
    return observed
