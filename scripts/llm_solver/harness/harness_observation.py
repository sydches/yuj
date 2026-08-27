"""Live harness observations for bounded mechanical concern packets.

This layer is deliberately narrower than the retired adaptive-control
detector/watch path. It tracks content-blind mechanical concern state from the
live prefix and, for the first pass, can append one compact user-role packet to
``halflife`` contexts only. The solver LLM remains responsible for semantic
interpretation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .._shared.classification import (
    classify_outcome,
    derive_envelope_status,
    is_gate_blocked,
)
from ._guardrails.extractors import (
    MUTATION_TOOLS,
    _is_bash_write_like,
    _is_test_command,
)
from .context_contract import context_mode_for_class


CONCERN_TYPE_OPEN_RED = "open_red_not_cleared"
OBSERVATION_HEADER = "HARNESS OBSERVATION"
DEFAULT_PACKET_CHAR_BUDGET = 1200
DEFAULT_EXCERPT_CHARS = 180
DEFAULT_EVIDENCE_LINES = 3

_READ_SEARCH_TOOLS = frozenset({
    "read",
    "grep",
    "glob",
    "list_files",
    "list_definitions",
    "structural_search",
})
_TEST_RESULTS_STATUS_RE = re.compile(r'<test_results\b[^>]*\bstatus="([^"]+)"')
_TOOL_RESULT_STATUS_RE = re.compile(r'<tool_result\b[^>]*\bstatus="([^"]+)"')


@dataclass
class ObservationFact:
    turn: int
    text: str
    kind: str


@dataclass
class ConcernState:
    concern_id: str
    concern_type: str
    opened_at_turn: int
    last_evidence_turn: int
    status: str = "open"
    facts: list[ObservationFact] = field(default_factory=list)
    later_activity_count: int = 0
    last_emitted_at_turn: int | None = None
    last_emitted_evidence_turn: int | None = None
    emit_count: int = 0


@dataclass
class ObservationState:
    next_concern_index: int = 1
    active: ConcernState | None = None


def observe_tool_result(
    session: Any,
    *,
    turn: int,
    tool_name: str,
    tool_args: dict | None,
    result: str,
    gate_blocked: bool = False,
) -> None:
    """Update prefix-only observation state from an executed tool result."""
    if not _enabled(getattr(session, "cfg", None)):
        return
    if gate_blocked or is_gate_blocked(result):
        return

    state = _state(session)
    outcome = classify_outcome(result)
    if _is_red(tool_name, tool_args, outcome):
        _record_red(state, int(turn), tool_name, result)
        return

    if state.active is None or state.active.status != "open":
        return

    if _is_clear_or_supersede(tool_name, tool_args, outcome):
        state.active.status = "cleared"
        state.active = None
        return

    if _is_mechanical_activity(tool_name, tool_args):
        state.active.later_activity_count += 1
        state.active.last_evidence_turn = int(turn)
        _append_fact(
            state.active,
            ObservationFact(
                turn=int(turn),
                kind="activity",
                text=f"{tool_name} ran after the red; no clear marker observed",
            ),
        )


def maybe_emit_observation(session: Any, *, turn: int) -> str | None:
    """Append a halflife user-role observation packet when policy fires."""
    cfg = getattr(session, "cfg", None)
    if not _enabled(cfg):
        return None
    state = getattr(session, "_harness_observation_state", None)
    if state is None or state.active is None or state.active.status != "open":
        return None
    concern = state.active
    reason = _emission_reason(cfg, concern, int(turn))
    if not reason:
        return None
    context_mode = context_mode_for_class(type(getattr(session, "context", object())))
    if context_mode != "halflife":
        return None

    packet = _format_packet(cfg, concern)
    if not packet:
        return None
    session.context.add_user(packet)
    concern.emit_count += 1
    concern.last_emitted_at_turn = int(turn)
    concern.last_emitted_evidence_turn = concern.last_evidence_turn
    _emit_trace(session, int(turn), concern, reason, packet, context_mode)
    return packet


def _enabled(cfg: Any) -> bool:
    return bool(getattr(cfg, "harness_observation_enabled", False))


def _state(session: Any) -> ObservationState:
    state = getattr(session, "_harness_observation_state", None)
    if state is None:
        state = ObservationState()
        setattr(session, "_harness_observation_state", state)
    return state


def _is_red(tool_name: str, tool_args: dict | None, outcome: str) -> bool:
    if outcome != "FAIL":
        return False
    if tool_name in MUTATION_TOOLS or _is_bash_write_like(tool_name, tool_args):
        return False
    return tool_name == "bash" or _is_test_command(tool_name, tool_args)


def _is_clear_or_supersede(tool_name: str, tool_args: dict | None, outcome: str) -> bool:
    if outcome != "OK":
        return False
    if tool_name in MUTATION_TOOLS or _is_bash_write_like(tool_name, tool_args):
        return True
    return _is_test_command(tool_name, tool_args)


def _is_mechanical_activity(tool_name: str, tool_args: dict | None) -> bool:
    if tool_name in _READ_SEARCH_TOOLS:
        return True
    if tool_name in MUTATION_TOOLS or _is_bash_write_like(tool_name, tool_args):
        return True
    if _is_test_command(tool_name, tool_args):
        return True
    return tool_name in {"bash", "done"}


def _record_red(
    state: ObservationState,
    turn: int,
    tool_name: str,
    result: str,
) -> None:
    if state.active is None or state.active.status != "open":
        concern = ConcernState(
            concern_id=f"hobs-{state.next_concern_index}",
            concern_type=CONCERN_TYPE_OPEN_RED,
            opened_at_turn=turn,
            last_evidence_turn=turn,
        )
        state.next_concern_index += 1
        state.active = concern
    else:
        concern = state.active
        if turn != concern.opened_at_turn:
            concern.later_activity_count += 1
        concern.last_evidence_turn = turn

    _append_fact(
        concern,
        ObservationFact(
            turn=turn,
            kind="red",
            text=f"{tool_name} failure marker: {_failure_excerpt(result)}",
        ),
    )


def _append_fact(concern: ConcernState, fact: ObservationFact) -> None:
    if (
        concern.facts
        and concern.facts[-1].turn == fact.turn
        and concern.facts[-1].text == fact.text
    ):
        return
    concern.facts.append(fact)
    # Keep enough history for packet selection without growing indefinitely.
    if len(concern.facts) > 12:
        del concern.facts[:-12]


def _emission_reason(cfg: Any, concern: ConcernState, turn: int) -> str:
    grace = max(
        0,
        int(getattr(cfg, "harness_observation_grace_activity_turns", 2) or 0),
    )
    cadence = max(
        1,
        int(getattr(cfg, "harness_observation_cadence_turns", 10) or 10),
    )
    if concern.later_activity_count < grace and (turn - concern.opened_at_turn) < cadence:
        return ""
    if concern.last_emitted_at_turn is None:
        if (turn - concern.opened_at_turn) >= cadence:
            return "cadence_watchdog"
        return "grace_expired"
    if (turn - concern.last_emitted_at_turn) < cadence:
        return ""
    if concern.last_evidence_turn > (concern.last_emitted_evidence_turn or -1):
        return "material_new_evidence"
    return "cadence_watchdog"


def _format_packet(cfg: Any, concern: ConcernState) -> str:
    budget = max(
        300,
        int(
            getattr(
                cfg,
                "harness_observation_packet_char_budget",
                DEFAULT_PACKET_CHAR_BUDGET,
            )
            or DEFAULT_PACKET_CHAR_BUDGET
        ),
    )
    max_lines = max(
        1,
        int(
            getattr(
                cfg,
                "harness_observation_evidence_lines",
                DEFAULT_EVIDENCE_LINES,
            )
            or DEFAULT_EVIDENCE_LINES
        ),
    )
    facts = _select_facts(concern, max_lines)
    lines = [
        OBSERVATION_HEADER,
        f"Type: {concern.concern_type}",
        "Status: open",
        "Evidence:",
    ]
    lines.extend(f"- T{fact.turn}: {fact.text}" for fact in facts)
    lines.extend([
        "Open question:",
        (
            "Before continuing, decide whether this is expected/out-of-scope "
            "or whether the patch needs repair. If expected, cite task "
            "evidence or a clean control. If not, repair or run a "
            "discriminating check."
        ),
    ])
    packet = "\n".join(lines)
    if len(packet) <= budget:
        return packet
    return packet[: max(0, budget - 3)].rstrip() + "..."


def _select_facts(concern: ConcernState, max_lines: int) -> list[ObservationFact]:
    if len(concern.facts) <= max_lines:
        return list(concern.facts)
    first_red = next(
        (fact for fact in concern.facts if fact.kind == "red"),
        concern.facts[0],
    )
    tail: list[ObservationFact] = []
    for fact in reversed(concern.facts):
        if fact is first_red:
            continue
        tail.append(fact)
        if len(tail) >= max_lines - 1:
            break
    return [first_red, *reversed(tail)]


def _failure_excerpt(result: str) -> str:
    max_chars = DEFAULT_EXCERPT_CHARS
    status = _structured_status(result)
    if status:
        return status[:max_chars]
    for raw in str(result or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("</tool_result"):
            continue
        if line.startswith("<tool_result"):
            continue
        if "[exit code:" in line or line.startswith("ERROR:"):
            return line[:max_chars]
    for raw in str(result or "").splitlines():
        line = raw.strip()
        if (
            line
            and not line.startswith("<tool_result")
            and not line.startswith("</tool_result")
        ):
            return line[:max_chars]
    status_name, error_kind = derive_envelope_status(str(result or ""))
    suffix = f"/{error_kind}" if error_kind else ""
    return f"{status_name}{suffix}"[:max_chars]


def _structured_status(result: str) -> str:
    text = str(result or "")
    test_match = _TEST_RESULTS_STATUS_RE.search(text)
    if test_match:
        return f'test_results status="{test_match.group(1)}"'
    tool_match = _TOOL_RESULT_STATUS_RE.search(text)
    if tool_match and tool_match.group(1) != "ok":
        status, error_kind = derive_envelope_status(text)
        if error_kind:
            return f'tool_result status="{status}" error_kind="{error_kind}"'
        return f'tool_result status="{status}"'
    return ""


def _emit_trace(
    session: Any,
    turn: int,
    concern: ConcernState,
    reason: str,
    packet: str,
    context_mode: str,
) -> None:
    emit = getattr(session, "_emit", None)
    if emit is None:
        return
    emit(
        "harness_observation",
        session_number=getattr(session, "_session_number", 0),
        turn_number=turn,
        concern_id=concern.concern_id,
        concern_type=concern.concern_type,
        reason=reason,
        opened_at_turn=concern.opened_at_turn,
        last_evidence_turn=concern.last_evidence_turn,
        emit_count=concern.emit_count,
        packet_chars=len(packet),
        context_mode=context_mode,
        evidence_turns=[
            fact.turn for fact in _select_facts(concern, DEFAULT_EVIDENCE_LINES)
        ],
        future_evidence_used=False,
    )


__all__ = [
    "CONCERN_TYPE_OPEN_RED",
    "OBSERVATION_HEADER",
    "ConcernState",
    "ObservationState",
    "maybe_emit_observation",
    "observe_tool_result",
]
