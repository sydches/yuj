"""Declared quiet periods withhold interventions while retaining observations."""
from types import SimpleNamespace

import pytest

from scripts.llm_solver.config import load_config


def test_default_is_armed_from_turn_one():
    cfg = load_config()
    assert cfg.guardrails_arm_after_turn == 0


def test_key_loads_from_toml(tmp_path):
    f = tmp_path / "o.toml"
    f.write_text("[loop]\nguardrails_arm_after_turn = 10\n")
    cfg = load_config(user_config=[str(f)])
    assert cfg.guardrails_arm_after_turn == 10


def test_duplicate_abort_zero_is_declared_disabled():
    from types import SimpleNamespace
    from scripts.llm_solver.harness._guardrails.checks_pre import duplicate_guard
    from scripts.llm_solver.harness._guardrails.state import GuardrailState
    st = GuardrailState()
    cfg = SimpleNamespace(duplicate_guard_enabled=True, duplicate_abort=0,
                          duplicate_warn_count=1,
                          duplicate_warn="[{count} identical; ends at {abort}]")
    d = None
    for _ in range(6):  # far past any deque length
        d = duplicate_guard(st, cfg, tool_calls_sig=("same",),
                            observations=({"kind": "read_observation", "pending": False, "sha256": "a"*64},))
    assert d.action.name != "END"
    assert "ends at disabled" in (d.text or "")


def test_quiet_intent_observations_do_not_count_as_rejections():
    from scripts.llm_solver.harness.guardrails import intent_gate, Action, init_guardrail_state
    from _config_helpers import make_config
    cfg = make_config(require_intent=True, intent_grace_turns=0, intent_abort_threshold=2)
    state = init_guardrail_state(cfg)
    for turn in range(4):
        decision = intent_gate(state, cfg, turn=turn, content="", tool_calls=[object()],
                               allow_intervention=False)
        assert decision.action == Action.PASS
        assert state.intent_evidence == {"required": True, "tool_calls": 1,
                                         "content_present": False}
        assert state.intent_block_count == state.consecutive_intent_rejections == 0
        assert state.intent_first_block_turn is None
    assert intent_gate(state, cfg, turn=4, content="", tool_calls=[object()]).action == Action.BLOCK
    assert state.consecutive_intent_rejections == 1


def test_quiet_loop_observations_do_not_consume_recovery_warning():
    from scripts.llm_solver.harness.guardrails import loop_detect, Action, init_guardrail_state
    from _config_helpers import make_config
    cfg = make_config(loop_detect_enabled=True, loop_detect_threshold=2)
    state = init_guardrail_state(cfg)
    for count in range(1, 5):
        assert loop_detect(state, cfg, tool_calls_sig=("same",),
                           allow_intervention=False).action == Action.PASS
        assert state.loop_detect_streak == count
        assert state.loop_detect_warned is False
    assert loop_detect(state, cfg, tool_calls_sig=("same",)).action == Action.PASS
    assert loop_detect(state, cfg, tool_calls_sig=("same",)).action == Action.PASS


def test_quiet_duplicate_history_retains_known_results_and_breaks_on_unknown():
    from scripts.llm_solver.harness.guardrails import duplicate_guard, Action, init_guardrail_state
    from scripts.llm_solver.harness.repeated_observations import read_observation
    from _config_helpers import make_config
    cfg = make_config(duplicate_guard_enabled=True, duplicate_abort=2, duplicate_warn_count=1)
    state = init_guardrail_state(cfg)
    args = dict(tool_calls_sig=("read",), observations=(read_observation("same"),))
    for _ in range(4):
        assert duplicate_guard(state, cfg, **args, allow_intervention=False).action == Action.PASS
    assert state.duplicate_evidence["count"] == 2
    assert duplicate_guard(state, cfg, tool_calls_sig=("read",), observations=None,
                           allow_intervention=False).action == Action.PASS
    assert not state.duplicate_evidence["eligible"]
    assert duplicate_guard(state, cfg, **args).action != Action.END
    assert duplicate_guard(state, cfg, **args).action == Action.END


@pytest.mark.parametrize("guard", ["intent_gate", "loop_detect", "duplicate_guard"])
def test_session_quiet_policy_also_withholds_override_actions(tmp_path, guard):
    from unittest.mock import MagicMock
    from _config_helpers import make_config
    from scripts.llm_solver.harness.guardrails import build_guardrail_registry, Decision
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage
    observed = []
    def replacement(state, cfg, **kwargs):
        observed.append(kwargs["allow_intervention"])
        return Decision.end("fixture_guard")
    phase = "turn_post_overrides" if guard == "duplicate_guard" else "turn_pre_overrides"
    registry = build_guardrail_registry(**{phase: {guard: replacement}})
    cfg = make_config(max_turns=4, guardrails_arm_after_turn=0)
    client = MagicMock()
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall(id="r", name="read", arguments={"path": "file.txt"})],
        finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    (tmp_path / "file.txt").write_text("fixture")
    session = Session(cfg, client, "system", "task", str(tmp_path), guardrail_registry=registry)
    assert session.run().finish_reason == "fixture_guard"
    assert observed == [False, True]
    assert client.chat.call_count == 2
