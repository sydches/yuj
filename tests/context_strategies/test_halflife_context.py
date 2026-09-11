from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver import config as config_module
from llm_solver.harness._loop._session_setup import build_context_manager
from llm_solver.harness.context_strategies import HalfLifeContext
from llm_solver.harness import savings


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
    assert "retrieval unverified" in contents[0]


def test_halflife_replace_all_messages_rebases_append_log():
    ctx = HalfLifeContext(context_size=10, activation_ratio=0.0)
    replacement = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "DIGEST"},
    ]

    assert ctx.replace_all_messages(replacement) is True
    assert ctx.get_messages() == replacement


def test_halflife_overlay_controls_activation_and_all_age_bands(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "_LOCAL_CONFIG", tmp_path / "absent.toml")
    overlay = tmp_path / "halflife.toml"
    overlay.write_text(
        "[context]\n"
        "halflife_context_limit_tokens = 400\n"
        "halflife_no_decay_ratio = 0.25\n"
        "halflife_verbatim_tool_results = 1\n"
        "halflife_cap_7_chars = 256\n"
        "halflife_cap_15_chars = 240\n"
        "halflife_cap_31_chars = 224\n"
        "halflife_cap_63_chars = 208\n"
        "halflife_cap_older_chars = 192\n"
    )
    cfg = config_module.load_config(user_config=overlay)
    for count in (99, 100):
        ctx = build_context_manager(
            HalfLifeContext, cfg, tmp_path, "TASK", 1,
            token_estimator=lambda messages: count,
        )
        ctx.add_system("SYSTEM")
        ctx.add_user("TASK")
        for index in range(66):
            ctx.add_assistant({"role": "assistant", "content": f"read {index}"})
            ctx.add_tool_result(f"call-{index}", "HEAD" + "x" * 392 + "TAIL")

        raw = ctx.snapshot_messages()
        rendered = ctx.get_messages()
        contents = list(reversed(_tool_content(rendered)))
        assert _tool_content(raw) == ["HEAD" + "x" * 392 + "TAIL"] * 66
        assert [m["role"] for m in rendered] == [m["role"] for m in raw]
        if count == 99:
            assert rendered == raw
            continue
        assert contents[0] == "HEAD" + "x" * 392 + "TAIL"
        for age, cap in (
            (1, 256), (7, 256), (8, 240), (15, 240), (16, 224),
            (31, 224), (32, 208), (63, 208), (64, 192), (65, 192),
        ):
            assert len(contents[age]) == cap, age
            assert contents[age].startswith("HEAD")
            assert contents[age].endswith("TAIL")
        assert ctx.snapshot_messages() == raw


def test_halflife_logs_only_the_first_active_render(tmp_path: Path):
    ledger_path = tmp_path / "halflife.jsonl"
    ledger = savings.open_ledger(ledger_path, task="halflife-task")
    try:
        ledger.set_turn(1, 4)
        ctx = HalfLifeContext(
            context_size=400,
            activation_ratio=0.50,
            verbatim_tool_results=1,
            cap_7_chars=80,
            token_estimator=_fixed_tokens,
        )
        ctx.add_system("SYSTEM")
        ctx.add_user("TASK")
        ctx.add_assistant({"role": "assistant", "content": "one"})
        ctx.add_tool_result("call-1", "A" * 100)

        # Still below the 50% gate: no half-life mutation record.
        assert ctx.get_messages() is ctx._messages
        assert ledger_path.read_text() == ""

        ctx.add_assistant({"role": "assistant", "content": "two"})
        ctx.add_tool_result("call-2", "B" * 700)
        rendered = ctx.get_messages()
        # Cached reads do not count the same request render twice.
        assert ctx.get_messages() is rendered
    finally:
        savings.close_ledger()

    records = [
        json.loads(line)
        for line in ledger_path.read_text().splitlines()
        if line
    ]
    assert records
    assert all(record["mechanism"] == "halflife_decay" for record in records)
    assert all(record["ctx"]["activation_threshold_tokens"] == 200
               for record in records)
    assert all(record["ctx"]["full_tokens_est"] >= 200 for record in records)
    assert all(record["chain_step"] == 1 for record in records)
    assert len({record["chain_id"] for record in records}) == len(records)
