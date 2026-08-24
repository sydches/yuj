"""Unit proofs for validated mid-stream rule semantics."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness.stream_rules import (
    StreamRuleError,
    StreamRuleRuntime,
    load_stream_rules,
    parse_stream_rule,
)
from scripts.llm_solver.server._streaming import StreamRuleInterrupt
from scripts.llm_solver.server.types import TurnResult, Usage


def _rule(frontmatter: str, body: str = "Correct the response."):
    return parse_stream_rule(
        f"+++\n{frontmatter.strip()}\n+++\n{body}\n",
        source_path=".harness/stream_rules/test.md",
        default_name="test",
    )


def _text_delta(text: str):
    return SimpleNamespace(source="text", delta=text)


def _finish(runtime: StreamRuleRuntime):
    return runtime.accept_response(
        TurnResult(None, [], "stop", Usage(0, 0)),
        turn=0,
        streamed=True,
        replay=False,
    )


def test_defaults_are_text_and_tool_always_once():
    rule = _rule('condition = "forbidden"')
    assert [scope.label for scope in rule.scopes] == ["text", "tool"]
    assert rule.interrupt_mode == "always"
    assert rule.repeat_mode == "once"


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ('condition = "["', "invalid condition regex"),
        ('condition = "x"\nscope = "answer"', "invalid scope token"),
        ('condition = "x"\ninterruptMode = "sometimes"', "interruptMode"),
        ('condition = "x"\nrepeatMode = "often"', "repeatMode"),
        ('condition = "x"\nrepeatGap = 0', "repeatGap"),
        ('scope = "text"', "at least one of condition or astCondition"),
    ],
)
def test_frontmatter_errors_name_the_file_and_invalid_field(frontmatter, message):
    with pytest.raises(StreamRuleError, match=message) as caught:
        _rule(frontmatter)
    assert ".harness/stream_rules/test.md" in str(caught.value)


def test_loader_uses_filename_order_metadata_and_rejects_duplicate_names(tmp_path):
    rule_dir = tmp_path / ".harness" / "stream_rules"
    rule_dir.mkdir(parents=True)
    (rule_dir / "20-second.md").write_text(
        '+++\nname = "same"\ncondition = "two"\n+++\nSecond.\n'
    )
    (rule_dir / "10-first.md").write_text(
        '+++\nname = "first"\ncondition = "one"\n+++\nFirst.\n'
    )
    loaded = load_stream_rules(rule_dir, display_dir=".harness/stream_rules")
    assert [rule.name for rule in loaded.rules] == ["first", "same"]
    assert loaded.files[0]["path"] == ".harness/stream_rules/10-first.md"
    assert len(str(loaded.files[0]["sha256"])) == 64

    (rule_dir / "30-duplicate.md").write_text(
        '+++\nname = "same"\ncondition = "three"\n+++\nThird.\n'
    )
    with pytest.raises(StreamRuleError, match="duplicate stream-rule name"):
        load_stream_rules(rule_dir, display_dir=".harness/stream_rules")


def test_once_rule_fires_only_once_per_session(tmp_path):
    rule = _rule('condition = "forbidden"\nrepeatMode = "once"')
    runtime = StreamRuleRuntime([rule], repeat_gap=10, cwd=tmp_path)

    runtime.begin_attempt()
    with pytest.raises(StreamRuleInterrupt) as caught:
        runtime.observe(_text_delta("forbidden"), turn=0)
    runtime.mark_injected(caught.value.matches, turn=0)

    runtime.begin_attempt()
    runtime.observe(_text_delta("forbidden"), turn=100)
    assert _finish(runtime) == ()


def test_after_gap_uses_logical_turns_not_stream_chunks(tmp_path):
    rule = _rule(
        'condition = "forbidden"\nrepeatMode = "after-gap"\nrepeatGap = 3'
    )
    runtime = StreamRuleRuntime([rule], repeat_gap=10, cwd=tmp_path)

    runtime.begin_attempt()
    with pytest.raises(StreamRuleInterrupt) as caught:
        runtime.observe(_text_delta("forbidden"), turn=4)
    runtime.mark_injected(caught.value.matches, turn=4)

    for _ in range(5):
        runtime.begin_attempt()
        runtime.observe(_text_delta("forbidden"), turn=6)
        assert _finish(runtime) == ()

    runtime.begin_attempt()
    with pytest.raises(StreamRuleInterrupt):
        runtime.observe(_text_delta("forbidden"), turn=7)


def test_after_gap_uses_config_default_when_rule_omits_repeat_gap(tmp_path):
    rule = _rule('condition = "forbidden"\nrepeatMode = "after-gap"')
    runtime = StreamRuleRuntime([rule], repeat_gap=2, cwd=tmp_path)

    runtime.begin_attempt()
    with pytest.raises(StreamRuleInterrupt) as caught:
        runtime.observe(_text_delta("forbidden"), turn=1)
    runtime.mark_injected(caught.value.matches, turn=1)

    runtime.begin_attempt()
    runtime.observe(_text_delta("forbidden"), turn=2)
    assert _finish(runtime) == ()

    runtime.begin_attempt()
    with pytest.raises(StreamRuleInterrupt):
        runtime.observe(_text_delta("forbidden"), turn=3)


def test_structural_tool_rule_honors_scope_and_repository_glob(tmp_path):
    rule = _rule(
        'astCondition = "eval($ARG)"\n'
        'scope = "tool:write(**/*.py)"\n'
        'interruptMode = "never"'
    )
    runtime = StreamRuleRuntime([rule], repeat_gap=10, cwd=tmp_path)
    runtime.begin_attempt()
    runtime.observe(
        SimpleNamespace(
            source="tool",
            delta="",
            tool_index=0,
            tool_name="write",
            tool_arguments=(
                '{"path":"root.py","content":"safe = 1\\neval(user_data)\\n"}'
            ),
        ),
        turn=0,
    )
    records = _finish(runtime)
    assert len(records) == 1
    assert records[0]["scope"] == "tool:write(**/*.py)"
    assert records[0]["path"] == "root.py"
    assert records[0]["interrupt"] is False
