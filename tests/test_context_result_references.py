"""Compression must retain a real inline source and preserve raw history."""
import json

import pytest
from _context_helpers import _make_solver_state


def context(tmp_path, *, budget=30000):
    (tmp_path / ".solver").mkdir(exist_ok=True)
    (tmp_path / ".solver/state.json").write_text("{}")
    ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task",
                             min_turns=0, recent_tool_results_chars=budget)
    ctx.add_system("system")
    ctx.add_user("task")
    ctx.add_assistant({"role": "assistant", "content": "inspect"})
    return ctx


@pytest.mark.parametrize("command", ["", '{"cmd":"cat x"}', '{"cmd":"pytest"}'])
def test_same_turn_repetition_preserves_source_and_uses_actual_call_id(tmp_path, command):
    ctx = context(tmp_path)
    source = "diagnostic " + "x" * 500
    for call in ("first", "second", "latest"):
        ctx.add_tool_result(call, source, tool_name="bash", cmd_signature=command)
    raw = ctx.snapshot_messages()
    assert [m["content"] for m in raw if m["role"] == "tool"] == [source] * 3
    view = ctx.get_messages()[1]["content"]
    assert view.count(source) == 1
    assert view.count('Same supplied text as tool call "latest"') == 2
    assert 'Tool call "latest"' in view
    assert "BLOCKED" not in view and "WARNING" not in view and "turn -" not in view
    assert ctx.snapshot_messages() == raw


def test_evicted_source_cannot_leave_dangling_reference(tmp_path):
    ctx = context(tmp_path, budget=650)
    source = "source " + "x" * 500
    ctx.add_tool_result("old", source, tool_name="bash")
    ctx.add_tool_result("latest", source, tool_name="bash")
    view = ctx.get_messages()[1]["content"]
    assert source in view
    assert 'Tool call "latest"' in view
    ctx.add_tool_result("different", "new " + "z" * 500, tool_name="bash")
    view = ctx.get_messages()[1]["content"]
    assert "Same supplied text" not in view
    assert "new " + "z" * 500 in view


@pytest.mark.parametrize("length", [50, 199, 200, 201])
def test_compression_depends_on_reference_cost_not_fixed_cutoff(tmp_path, length):
    ctx = context(tmp_path)
    source = "x" * length
    ctx.add_tool_result("first", source, tool_name="bash")
    ctx.add_tool_result("last", source, tool_name="bash")
    view = ctx.get_messages()[1]["content"]
    assert ("Same supplied text" in view) is (length >= 199)
    assert view.count(source) == (1 if length >= 199 else 2)


@pytest.mark.parametrize("ids", [("same", "same"), ("", ""), ("a", "b" * 1000)])
def test_ambiguous_or_more_expensive_reference_preserves_both_results(tmp_path, ids):
    ctx = context(tmp_path)
    for call in ids:
        ctx.add_tool_result(call, "x" * 300, tool_name="bash")
    assert ctx.get_messages()[1]["content"].count("x" * 300) == 2


@pytest.mark.parametrize("blocked", ["first", "last"])
def test_blocked_current_or_source_result_cannot_be_compressed(tmp_path, blocked):
    ctx = context(tmp_path)
    for call in ("first", "last"):
        ctx.add_tool_result(call, "x" * 300, tool_name="bash", gate_blocked=call == blocked)
    assert ctx.get_messages()[1]["content"].count("x" * 300) == 2


def test_reset_and_changed_output_do_not_reuse_a_source(tmp_path):
    ctx = context(tmp_path)
    ctx.add_tool_result("a", "x" * 300, tool_name="bash", cmd_signature="same command")
    ctx.reset_dedup_counts()
    ctx.add_tool_result("b", "x" * 300, tool_name="bash", cmd_signature="same command")
    ctx.add_tool_result("c", "y" * 300, tool_name="bash", cmd_signature="same command")
    view = ctx.get_messages()[1]["content"]
    assert view.count("x" * 300) == 2
    assert "y" * 300 in view


def test_non_ascii_text_and_escaped_call_ids_keep_exact_source(tmp_path):
    ctx = context(tmp_path)
    source = "α漢字" * 100
    anchor = 'last\n"quoted"'
    ctx.add_tool_result("first", source, tool_name="bash")
    ctx.add_tool_result(anchor, source, tool_name="bash")
    view = ctx.get_messages()[1]["content"]
    assert view.count(source) == 1
    assert "Same supplied text as tool call " + json.dumps(anchor) in view
    assert "Tool call " + json.dumps(anchor) in view


def test_raw_fallback_and_rewind_keep_full_supplied_results(tmp_path):
    ctx = context(tmp_path)
    for call in ("a", "b"):
        ctx.add_tool_result(call, "x" * 300, tool_name="bash")
    raw = ctx.snapshot_messages()
    ctx.get_messages()
    ctx.rewind_messages(raw)
    assert ctx.snapshot_messages() == raw
    (tmp_path / ".solver/state.json").unlink()
    ctx.rewind_messages(raw)
    assert [m["content"] for m in ctx.get_messages() if m["role"] == "tool"] == ["x" * 300] * 2


@pytest.mark.parametrize("signature,tier", [("", "tier2_byte_identical"),
                                            ('{"cmd":"inspect"}', "tier1_cmd_signature")])
def test_ledger_records_only_rendered_replacements_and_their_sources(tmp_path, signature, tier):
    from llm_solver.harness import savings
    ctx = context(tmp_path)
    path = tmp_path / "savings.jsonl"
    savings.open_ledger(path)
    try:
        for call in ("a", "b", "c"):
            ctx.add_tool_result(call, "x" * 300, tool_name="bash", cmd_signature=signature)
        assert path.read_text() == ""
        view = ctx.get_messages()
        assert ctx.get_messages() is view
    finally:
        savings.close_ledger()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1  # component reductions must not be counted twice
    projection = records[0]
    assert projection["bucket"] == "context_projection"
    assert projection["surface"] == "context_render"
    replacements = projection["ctx"]["result_references"]
    assert len(replacements) == 2
    assert {r["tool_call_id"] for r in replacements} == {"a", "b"}
    for record in replacements:
        assert record["mechanism"] == tier
        assert record["reference_tool_call_id"] == "c"
        assert record["reference_scope"] == "same_render_full_result"
        assert record["output_chars"] < record["input_chars"]


def test_local_session_delivers_reference_and_full_native_search_result(tmp_path):
    from copy import deepcopy
    from unittest.mock import MagicMock
    from _config_helpers import make_config
    from llm_solver.harness.loop import Session
    from llm_solver.server.types import ToolCall, TurnResult, Usage

    (tmp_path / ".solver").mkdir()
    (tmp_path / ".solver/state.json").write_text("{}")
    source = "needle " + "x" * 500
    (tmp_path / "input.txt").write_text(source + "\n")
    ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Inspect",
                             min_turns=0, suffix="")
    cfg = make_config(max_turns=2, reply_mode="conversation", require_intent=False,
                      tools_output_dedup_enabled=False, duplicate_guard_enabled=False,
                      loop_detect_enabled=False, duplicate_abort=0)
    client = MagicMock()
    requests = []

    def chat(messages, tools, **kwargs):
        requests.append(deepcopy(messages))
        if len(requests) == 1:
            return TurnResult(content=None, tool_calls=[
                ToolCall(id=call, name="grep", arguments={"pattern": "needle", "path": "input.txt"})
                for call in ("a", "b", "c")], finish_reason="tool_calls",
                usage=Usage(prompt_tokens=10, completion_tokens=5))
        return TurnResult(content="finished", tool_calls=[], finish_reason="stop",
                          usage=Usage(prompt_tokens=10, completion_tokens=5))

    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    current = Session(cfg, client, "system", "Inspect", str(tmp_path), context_manager=ctx)
    current.run()
    assert len(requests) == 2
    rendered = requests[-1][-1]["content"]
    assert rendered.count(source) == 1
    assert rendered.count('Same supplied text as tool call "c"') == 2
    results = [m for m in ctx.snapshot_messages() if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["a", "b", "c"]
    assert all(source in m["content"] for m in results)
