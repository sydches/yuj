from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver.harness._loop._session_setup import build_context_manager
from llm_solver.harness.context_strategies import HalfLifeContext


def _fixed_tokens(messages: list[dict]) -> int:
    return sum(len(str(message)) for message in messages) // 4


def _tool_content(messages: list[dict]) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "tool"
    ]


def test_halflife_keeps_full_transcript_below_activation_threshold():
    ctx = HalfLifeContext(
        context_size=100_000,
        activation_ratio=0.50,
        token_estimator=_fixed_tokens,
    )
    ctx.add_system("SYSTEM")
    ctx.add_user("TASK")
    ctx.add_assistant({"role": "assistant", "content": "read", "tool_calls": []})
    ctx.add_tool_result("call-1", "A" * 1000)

    messages = ctx.get_messages()

    assert messages is ctx._messages
    assert _tool_content(messages) == ["A" * 1000]


def test_standard_halflife_does_not_inject_stateful_suffix(tmp_path: Path):
    cfg = make_config(
        context_size=100_000,
        state_context_suffix="Continue. Progress tracked in .solver/state.json.",
    )

    ctx = build_context_manager(
        HalfLifeContext, cfg, tmp_path, "TASK", 1, token_estimator=_fixed_tokens,
    )
    assert ctx is not None
    ctx.add_system("SYSTEM")
    ctx.add_user("TASK")

    assert ctx.get_messages() == ctx._messages
    assert all(".solver/state.json" not in str(message) for message in ctx.get_messages())


def test_halflife_decays_old_tool_results_after_activation():
    ctx = HalfLifeContext(
        context_size=1000,
        activation_ratio=0.10,
        verbatim_tool_results=2,
        cap_7_chars=200,
        cap_15_chars=60,
        cap_31_chars=40,
        cap_63_chars=30,
        cap_older_chars=20,
        token_estimator=_fixed_tokens,
    )
    ctx.add_system("SYSTEM")
    ctx.add_user("TASK")
    for index in range(6):
        ctx.add_assistant({
            "role": "assistant",
            "content": f"turn {index}",
            "tool_calls": [{"id": f"call-{index}", "type": "function"}],
        })
        ctx.add_tool_result(f"call-{index}", f"RESULT-{index}-" + ("x" * 240))

    messages = ctx.get_messages()
    contents = _tool_content(messages)

    assert [message["role"] for message in messages] == [
        "system", "user",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "tool",
    ]
    assert len(contents[-1]) > 200
    assert len(contents[-2]) > 200
    assert len(contents[0]) <= 200
    assert "[halflife: omitted" in contents[0]
    assert "full output remains in trace/transcript artifacts" in contents[0]


def test_halflife_replace_all_messages_rebases_append_log():
    ctx = HalfLifeContext(context_size=10, activation_ratio=0.0)
    replacement = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "DIGEST"},
    ]

    assert ctx.replace_all_messages(replacement) is True
    assert ctx.get_messages() == replacement
