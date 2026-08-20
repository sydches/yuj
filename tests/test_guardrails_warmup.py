"""guardrails_arm_after_turn: pre-dispatch guardrails dormant through the
warmup, active after."""
from types import SimpleNamespace

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
        d = duplicate_guard(st, cfg, tool_calls_sig=("same",))
    assert d.action.name != "END"
    assert "ends at disabled" in (d.text or "")
