from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.context import FullTranscript
from llm_solver.harness.context_strategies import HalfLifeContext
from llm_solver.harness.harness_observation import (
    OBSERVATION_HEADER,
    maybe_emit_observation,
    observe_tool_result,
)
from llm_solver.harness._loop.trace_schema import KNOWN_TRACE_EVENT_TYPES


def _session(context, **overrides):
    cfg_kwargs = {
        "harness_observation_enabled": True,
        "harness_observation_grace_activity_turns": 2,
        "harness_observation_cadence_turns": 10,
        "harness_observation_packet_char_budget": 1200,
        "harness_observation_evidence_lines": 3,
    }
    cfg_kwargs.update(overrides)
    cfg = make_config(**cfg_kwargs)
    context.add_system("SYSTEM")
    context.add_user("TASK")
    emitted = []

    def _emit(event_type, **fields):
        emitted.append({"event": event_type, **fields})

    return SimpleNamespace(
        cfg=cfg,
        context=context,
        _session_number=0,
        _emit=_emit,
        emitted=emitted,
    )


def _observe(session, turn, tool_name, result, args=None):
    observe_tool_result(
        session,
        turn=turn,
        tool_name=tool_name,
        tool_args=args or {},
        result=result,
        gate_blocked=False,
    )
    return maybe_emit_observation(session, turn=turn)


def _bash_fail() -> str:
    return "pytest reported failure\n[exit code: 1]"


def test_halflife_emits_packet_after_grace_activity():
    session = _session(HalfLifeContext())

    assert _observe(session, 1, "bash", _bash_fail(), {"cmd": "pytest"}) is None
    assert _observe(session, 2, "read", "file body", {"path": "x.py"}) is None
    packet = _observe(session, 3, "grep", "match", {"pattern": "foo", "path": "."})

    assert packet is not None
    assert packet.startswith(OBSERVATION_HEADER)
    assert "Type: open_red_not_cleared" in packet
    assert "Status: open" in packet
    assert "- T1: bash failure marker:" in packet
    assert "- T2: read ran after the red; no clear marker observed" in packet
    assert "- T3: grep ran after the red; no clear marker observed" in packet
    assert session.context.get_messages()[-1] == {"role": "user", "content": packet}
    assert session.emitted[-1]["event"] == "harness_observation"
    assert session.emitted[-1]["reason"] == "grace_expired"
    assert session.emitted[-1]["future_evidence_used"] is False


def test_successful_mutation_supersedes_red_before_packet():
    session = _session(HalfLifeContext())

    assert _observe(session, 1, "bash", _bash_fail(), {"cmd": "pytest"}) is None
    assert _observe(
        session,
        2,
        "edit",
        "OK: replacement made",
        {"path": "x.py", "old_str": "a", "new_str": "b"},
    ) is None
    assert _observe(session, 3, "read", "file body", {"path": "x.py"}) is None

    assert not any(
        str(message.get("content", "")).startswith(OBSERVATION_HEADER)
        for message in session.context.get_messages()
    )
    assert session.emitted == []


def test_halflife_rate_limits_unchanged_concern_until_cadence():
    session = _session(HalfLifeContext())

    _observe(session, 1, "bash", _bash_fail(), {"cmd": "pytest"})
    _observe(session, 2, "read", "file body", {"path": "x.py"})
    first = _observe(session, 3, "grep", "match", {"pattern": "foo", "path": "."})
    assert first is not None

    assert _observe(session, 4, "read", "file body", {"path": "y.py"}) is None
    assert _observe(session, 9, "read", "file body", {"path": "z.py"}) is None
    second = _observe(session, 13, "read", "file body", {"path": "q.py"})

    assert second is not None
    assert len([
        m for m in session.context.get_messages()
        if m.get("content", "").startswith(OBSERVATION_HEADER)
    ]) == 2
    assert [event["reason"] for event in session.emitted] == [
        "grace_expired",
        "material_new_evidence",
    ]


def test_non_halflife_context_updates_state_but_does_not_append_packet():
    session = _session(FullTranscript())

    _observe(session, 1, "bash", _bash_fail(), {"cmd": "pytest"})
    _observe(session, 2, "read", "file body", {"path": "x.py"})
    packet = _observe(session, 3, "grep", "match", {"pattern": "foo", "path": "."})

    assert packet is None
    assert session._harness_observation_state.active is not None
    assert not any(
        str(message.get("content", "")).startswith(OBSERVATION_HEADER)
        for message in session.context.get_messages()
    )
    assert session.emitted == []


def test_packet_is_bounded_and_turn_cited():
    session = _session(
        HalfLifeContext(),
        harness_observation_packet_char_budget=420,
        harness_observation_evidence_lines=3,
    )

    _observe(session, 1, "bash", "x" * 1000 + "\n[exit code: 1]", {"cmd": "pytest"})
    _observe(session, 2, "read", "file body", {"path": "x.py"})
    packet = _observe(session, 3, "read", "file body", {"path": "y.py"})

    assert packet is not None
    assert len(packet) <= 420
    assert "T1:" in packet
    assert "T2:" in packet
    assert "T3:" in packet


def test_trace_schema_knows_harness_observation_event():
    assert "harness_observation" in KNOWN_TRACE_EVENT_TYPES
