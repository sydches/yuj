"""Packet, prompt, verdict, and log helpers for the LLM hurdle detector."""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...config import resolve_project_path
from ..raw_trace_state_ledger import replay_events
from .llm_detector_utils import (
    _json_object_candidates,
    _optional_bool,
    _sha256,
    _string_list,
    _strip_json_fence,
    _tail,
    _trace_prefix_rows,
)

SCHEMA_VERSION = "llm_hurdle_detector_v1"
PROMPT_VERSION = "llm_hurdle_detector_prompt_v4"
SIMPLE_PROMPT_VERSION = "simple_live_hurdle_prompt_v1"
EVENT_TYPE = "llm_hurdle_detector_verdict"

HURDLE_PRESENT_VALUES = {"yes", "no", "uncertain"}
CONFIDENCE_VALUES = {"low", "medium", "high"}

class AtlasFamily:
    family: str
    cell_count: int
    description: str
    covered_by_prior: str = ""
    uncovered: str = ""
    partial_or_late: str = ""


ATLAS_BASE_FIELDS = (
    "family",
    "cell_count",
    "description",
    "covered_by_prior",
    "uncovered",
    "partial_or_late",
)


@dataclass(frozen=True)
class LLMDetectorPacket:
    schema_version: str
    prompt_version: str
    observation_turn: int
    evidence_regime: str
    atlas_source: str
    input_contract_source: str
    atlas_families: list[dict[str, Any]]
    trace_prefix: list[dict[str, Any]]
    raw_state_snapshots: list[dict[str, str]]
    config_state: dict[str, Any] = field(default_factory=dict)
    omissions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LLMDetectorVerdict:
    hurdle_present: str
    hurdle_family: str
    confidence: str
    evidence_refs: list[str]
    recommended_config: str = ""
    abstain_reason: str = ""
    decision_summary: str = ""
    why_now: str = ""
    new_facts_still_appearing: bool | None = None
    rejected_families: list[str] = field(default_factory=list)
    rejected_reason: str = ""
    timing_basis: str = ""
    uncertainty: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LLMDetectorVerdict":
        data = _normalize_detector_mapping(data)
        hurdle_present = str(data.get("hurdle_present", "")).strip().lower()
        if hurdle_present not in HURDLE_PRESENT_VALUES:
            raise ValueError(f"invalid hurdle_present: {hurdle_present!r}")

        confidence = str(data.get("confidence", "")).strip().lower()
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(f"invalid confidence: {confidence!r}")

        refs = data.get("evidence_refs", [])
        if isinstance(refs, str):
            refs = [refs] if refs else []
        if not isinstance(refs, list):
            raise ValueError("evidence_refs must be a list or string")
        evidence_refs = [str(item).strip() for item in refs if str(item).strip()]

        hurdle_family = str(data.get("hurdle_family", "") or "").strip()
        if hurdle_present == "yes" and not hurdle_family:
            raise ValueError("hurdle_family is required when hurdle_present=yes")
        if hurdle_present == "yes" and not evidence_refs:
            raise ValueError("evidence_refs are required when hurdle_present=yes")
        if hurdle_present != "yes" and not str(data.get("abstain_reason", "") or "").strip():
            raise ValueError("abstain_reason is required when hurdle_present is no/uncertain")

        decision_summary = str(data.get("decision_summary", "") or "").strip()
        why_now = str(data.get("why_now", "") or "").strip()
        rejected_families = _string_list(data.get("rejected_families", []), field_name="rejected_families")
        rejected_reason = str(data.get("rejected_reason", "") or "").strip()
        timing_basis = str(data.get("timing_basis", "") or "").strip()
        uncertainty = str(data.get("uncertainty", "") or "").strip()
        missing_rationale = [
            name for name, value in (
                ("decision_summary", decision_summary),
                ("why_now", why_now),
                ("timing_basis", timing_basis),
                ("uncertainty", uncertainty),
            )
            if not value
        ]
        if missing_rationale:
            raise ValueError("missing detector rationale fields: " + ",".join(missing_rationale))
        if rejected_families and not rejected_reason:
            raise ValueError("rejected_reason is required when rejected_families is non-empty")

        return cls(
            hurdle_present=hurdle_present,
            hurdle_family=hurdle_family,
            confidence=confidence,
            evidence_refs=evidence_refs,
            recommended_config=str(data.get("recommended_config", "") or "").strip(),
            abstain_reason=str(data.get("abstain_reason", "") or "").strip(),
            decision_summary=decision_summary,
            why_now=why_now,
            new_facts_still_appearing=_optional_bool(data.get("new_facts_still_appearing")),
            rejected_families=rejected_families,
            rejected_reason=rejected_reason,
            timing_basis=timing_basis,
            uncertainty=uncertainty,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _normalize_detector_mapping(data: dict[str, Any]) -> dict[str, Any]:
    if "live_hurdle" not in data:
        return data
    normalized = dict(data)
    live = str(normalized.get("live_hurdle", "") or "").strip().lower()
    hurdle_present = "uncertain" if live == "needs_review" else live
    evidence_note = str(normalized.get("evidence_note", "") or "").strip()
    if "hurdle_present" not in normalized:
        normalized["hurdle_present"] = hurdle_present
    if "hurdle_family" not in normalized:
        normalized["hurdle_family"] = normalized.get("simple_hurdle_name") or ""
    if "confidence" not in normalized:
        normalized["confidence"] = "medium"
    if "evidence_refs" not in normalized:
        normalized["evidence_refs"] = _string_list(normalized.get("evidence_turns", []), field_name="evidence_turns")
    if hurdle_present != "yes" and "abstain_reason" not in normalized:
        normalized["abstain_reason"] = evidence_note or "no live hurdle"
    normalized.setdefault("decision_summary", evidence_note or "simple live-hurdle detector response")
    normalized.setdefault("why_now", evidence_note or "see evidence_turns")
    normalized.setdefault("new_facts_still_appearing", None)
    normalized.setdefault("rejected_families", [])
    normalized.setdefault("rejected_reason", "none")
    normalized.setdefault("timing_basis", ";".join(normalized.get("evidence_refs", [])) or "simple live-hurdle schema")
    normalized.setdefault("uncertainty", "needs_review" if live == "needs_review" else "none")
    return normalized


def load_atlas_families(path: str | Path) -> list[dict[str, Any]]:
    with resolve_project_path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    families: list[dict[str, Any]] = []
    for row in rows:
        if row.get("simple_hurdle_name"):
            family = _simple_hurdle_family(row)
        else:
            family = {
                "family": row.get("family", ""),
                "cell_count": int(row.get("cell_count", 0) or 0),
                "description": row.get("description", ""),
                "covered_by_prior": row.get("covered_by_prior", ""),
                "uncovered": row.get("uncovered", ""),
                "partial_or_late": row.get("partial_or_late", ""),
            }
            for key, value in row.items():
                if key in ATLAS_BASE_FIELDS or key in ("dictionary_version", "cells"):
                    continue
                if value:
                    family[key] = value
        families.append(family)
    return families


def _simple_hurdle_family(row: dict[str, str]) -> dict[str, Any]:
    family = {
        "family": row.get("simple_hurdle_name", ""),
        "cell_count": int(row.get("gold_yes_count", 0) or 0),
        "description": row.get("plain_rule", ""),
        "decision_order": row.get("decision_order", ""),
        "positive_live_signs": row.get("positive_live_signs", ""),
        "do_not_call_when": row.get("do_not_call_when", ""),
        "source_specific_labels": row.get("source_specific_labels", ""),
    }
    return {key: value for key, value in family.items() if value != ""}


def build_detector_packet(
    *,
    atlas_dictionary_path: str | Path,
    trace_events: list[dict[str, Any]],
    observation_turn: int,
    input_contract_path: str | Path = "",
    config_state: dict[str, Any] | None = None,
    evidence_regime: str = "causal_live",
    max_trace_events: int = 80,
    max_field_chars: int = 800,
    max_state_snapshots: int = 24,
    prompt_version: str = PROMPT_VERSION,
) -> LLMDetectorPacket:
    """Build a detector evidence packet from already-loaded live artifacts."""
    observation_turn = int(observation_turn)
    trace_prefix_all = _trace_prefix_rows(
        trace_events,
        observation_turn=observation_turn,
        max_field_chars=max_field_chars,
    )
    trace_prefix = _tail(trace_prefix_all, max_trace_events)
    snapshots_all = [item.to_row() for item in replay_events(trace_events, through_turn=observation_turn)]
    snapshots = _tail(snapshots_all, max_state_snapshots)
    families = load_atlas_families(atlas_dictionary_path)

    return LLMDetectorPacket(
        schema_version=SCHEMA_VERSION,
        prompt_version=prompt_version or PROMPT_VERSION,
        observation_turn=observation_turn,
        evidence_regime=evidence_regime,
        atlas_source=str(atlas_dictionary_path),
        input_contract_source=str(input_contract_path) if input_contract_path else "",
        atlas_families=families,
        trace_prefix=trace_prefix,
        raw_state_snapshots=snapshots,
        config_state=config_state or {},
        omissions={
            "raw_transcript_included": False,
            "future_turns_included": False,
            "post_run_artifacts_included": False,
            "trace_prefix_tool_calls_total": len(trace_prefix_all),
            "trace_prefix_tool_calls_included": len(trace_prefix),
            "trace_prefix_older_tool_calls_omitted": max(0, len(trace_prefix_all) - len(trace_prefix)),
            "raw_state_snapshots_total": len(snapshots_all),
            "raw_state_snapshots_included": len(snapshots),
            "raw_state_snapshots_older_omitted": max(0, len(snapshots_all) - len(snapshots)),
            "max_trace_events": max_trace_events,
            "max_field_chars": max_field_chars,
            "max_state_snapshots": max_state_snapshots,
        },
    )


def render_detector_messages(packet: LLMDetectorPacket) -> list[dict[str, str]]:
    """Render the prompt/messages for a no-tool detector LLM call."""
    if packet.prompt_version.startswith("simple_live_hurdle"):
        return _render_simple_detector_messages(packet)
    packet_json = json.dumps(packet.to_dict(), indent=2, sort_keys=True)
    system = (
        "You are a live hurdle detector, not a task solver. "
        "Use only the evidence packet. Do not use future turns, raw transcript, "
        "post-run artifacts, or outside knowledge. Name only atlas families. "
        "Return one JSON object only. Do not include markdown, prose, or "
        "reasoning tags such as <think>. The first character of your response "
        "must be {."
    )
    user = f"""\
Detector task:
- Decide whether a hurdle is present at observation_turn.
- If yes, name the atlas hurdle family and cite exact artifact/turn/field evidence.
- If no or uncertain, abstain and explain why.
- Do not propose task fixes.
- The harness routes interventions through its lookup table. Leave
  recommended_config empty unless the evidence packet already contains an exact
  configured id.

Required JSON schema:
{{
  "hurdle_present": "yes|no|uncertain",
  "hurdle_family": "<atlas family or empty>",
  "confidence": "low|medium|high",
  "evidence_refs": ["host_task/.trace.jsonl:T<turn>:<field>", "..."],
  "recommended_config": "<TOML intervention/config id or empty>",
  "abstain_reason": "<required for no/uncertain>",
  "decision_summary": "<one short sentence explaining the verdict>",
  "why_now": "<why this exact turn is, or is not, the hurdle point>",
  "new_facts_still_appearing": true|false|null,
  "rejected_families": ["<atlas family considered but rejected>", "..."],
  "rejected_reason": "<why rejected families do not fit; use 'none' if none>",
  "timing_basis": "<first visible turn and artifact basis, or why no timing basis exists>",
  "uncertainty": "<remaining doubt, or 'none'>"
}}

Decision discipline:
- A red signal is not automatically a hurdle. Hurdle status begins when the
  model mishandles the red, repeats a dead frame, crosses a bad finish boundary,
  or otherwise meets a family definition.
- Before saying "yes", ask whether the model is still gaining new useful facts.
  If useful new facts are still appearing, prefer "no" or "uncertain".
- The "new useful facts" guard has a limit. Tiny additional observations do not
  excuse sustained no-edit diagnosis after the relevant test, target, or
  reproduction is already visible. If the state snapshots show zero source
  writes, repeated same-purpose reads/probes, and rising prompt/context pressure,
  actively consider `loop_churn` or `budget_mechanics`.
- Initial orientation is not unlimited. Repeating the same empty or missing
  introspection target, such as `.solver/state.json`, for many turns produces no
  new task fact and should be treated as a `loop_churn` candidate even before a
  source file has been found.
- A passing existing test can be a diagnostic fact, but it is not a clearance if
  the trace also shows the live issue behavior remains reproduced and no tracked
  source edit exists.
- Explain rejected families explicitly. For example, a /tmp helper script is not
  patch_assembly when it writes tracked /testbed source and the final diff keeps
  that source change.
- If you cannot fill these rationale fields clearly, return "uncertain".

Evidence packet:
```json
{packet_json}
```
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _render_simple_detector_messages(packet: LLMDetectorPacket) -> list[dict[str, str]]:
    packet_json = json.dumps(packet.to_dict(), indent=2, sort_keys=True)
    system = (
        "You are a live simple-hurdle detector, not a task solver. "
        "Use only the evidence packet. Do not use future turns, raw transcript, "
        "post-run artifacts, branch buckets, manual verdicts, or outside knowledge. "
        "Return one JSON object only. The first character of your response must be {."
    )
    user = f"""\
Detector task:
1. Decide whether a live-visible hurdle exists at observation_turn.
2. If yes, choose exactly one simple hurdle name from atlas_families.
3. If no, set simple_hurdle_name to null.
4. If the evidence is insufficient, return live_hurdle = "needs_review".

Use the first matching card by decision_order. Do not invent names.

Required JSON schema:
{{
  "live_hurdle": "yes|no|needs_review",
  "simple_hurdle_name": "<one atlas_families.family value or null>",
  "evidence_turns": ["T12", "T18"],
  "evidence_note": "<one short sentence using only prefix-visible evidence>",
  "used_forbidden_evidence": false
}}

Evidence packet:
```json
{packet_json}
```
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_detector_verdict(text: str) -> LLMDetectorVerdict:
    """Parse a detector LLM JSON response into a closed verdict object."""
    stripped = _strip_json_fence(text)
    candidate_errors: list[str] = []
    valid: list[LLMDetectorVerdict] = []
    for data in _json_object_candidates(stripped):
        try:
            valid.append(LLMDetectorVerdict.from_mapping(data))
        except ValueError as exc:
            candidate_errors.append(str(exc))
    if valid:
        return valid[-1]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        if candidate_errors:
            raise ValueError(f"no schema-valid detector JSON found; last candidate error: {candidate_errors[-1]}")
        raise
    if not isinstance(data, dict):
        raise ValueError("detector verdict must be a JSON object")
    return LLMDetectorVerdict.from_mapping(data)


def run_detector_call(
    packet: LLMDetectorPacket,
    call_model: Callable[[list[dict[str, str]]], str],
) -> tuple[list[dict[str, str]], str, LLMDetectorVerdict]:
    """Run an injected no-tool model call and parse its verdict."""
    messages = render_detector_messages(packet)
    raw_response = call_model(messages)
    verdict = parse_detector_verdict(raw_response)
    return messages, raw_response, verdict


def build_detector_log_row(
    *,
    packet: LLMDetectorPacket,
    messages: list[dict[str, str]],
    raw_response: str,
    verdict: LLMDetectorVerdict,
) -> dict[str, Any]:
    packet_json = json.dumps(packet.to_dict(), sort_keys=True)
    messages_json = json.dumps(messages, sort_keys=True)
    return {
        "event_type": EVENT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": packet.prompt_version,
        "observation_turn": packet.observation_turn,
        "evidence_regime": packet.evidence_regime,
        "packet_sha256": _sha256(packet_json),
        "messages_sha256": _sha256(messages_json),
        "packet": packet.to_dict(),
        "verdict": verdict.to_dict(),
        "raw_response": raw_response,
    }


def append_detector_log(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
