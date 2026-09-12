"""Unit proofs for validated mid-stream rule semantics."""
from __future__ import annotations

from types import SimpleNamespace
import json

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


@pytest.mark.parametrize('suffix', ['', '\nunfinished = ('])
def test_ast_rules_share_source_parse_preserving_order_and_character_offsets(tmp_path, monkeypatch, suffix):
    from scripts.llm_solver.harness import _stream_rule_ast as module

    rules = [
        _rule('name = "later"\nastCondition = "exec($ARG)"\ninterruptMode = "never"'),
        _rule('name = "earlier"\nastCondition = "eval($ARG)"'),
    ]
    text = '# café\neval(value)\nexec(other)' + suffix
    original = module._parser_for
    source_parses = []

    def parser_for(language):
        parser = original(language)
        def parse(raw):
            if raw == text.encode():
                source_parses.append(language)
            return parser.parse(raw)
        return SimpleNamespace(parse=parse)

    monkeypatch.setattr(module, '_parser_for', parser_for)
    runtime = StreamRuleRuntime(rules, repeat_gap=10, cwd=tmp_path)
    delta = SimpleNamespace(source='tool', delta='', tool_index=0, tool_name='write',
                            tool_arguments=json.dumps({'path': 'root.py', 'content': text}))
    with pytest.raises(StreamRuleInterrupt) as caught:
        runtime.observe(delta, turn=0)
    assert source_parses == ['python']
    assert [(r['rule'], r['offset'], r['interrupt']) for r in caught.value.matches] == [
        ('later', text.index('exec'), False), ('earlier', text.index('eval'), True),
    ]


def test_ast_source_parse_is_fresh_for_each_delta_and_language(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import _stream_rule_ast as module

    rules = [_rule(f'name = "{language}"\nastCondition = "forbidden($ARG)"\n'
                   f'scope = "tool:write(*.{extension})"')
             for language, extension in [('python', 'py'), ('javascript', 'js')]]
    runtime = StreamRuleRuntime(rules, repeat_gap=10, cwd=tmp_path)
    original = module._parser_for
    observed = []

    def parser_for(language):
        parser = original(language)
        def parse(raw):
            if raw in (b'safe(', b'forbidden(value)'):
                observed.append((language, raw))
            return parser.parse(raw)
        return SimpleNamespace(parse=parse)

    monkeypatch.setattr(module, '_parser_for', parser_for)
    for text in ['safe(', 'forbidden(value)']:
        delta = SimpleNamespace(source='tool', delta='', tool_index=0, tool_name='write',
            tool_arguments=json.dumps({'path': 'root.py', 'file': 'root.js', 'content': text}))
        if text.startswith('safe'):
            runtime.observe(delta, turn=0)
        else:
            with pytest.raises(StreamRuleInterrupt) as caught:
                runtime.observe(delta, turn=0)
            assert [r['rule'] for r in caught.value.matches] == ['python', 'javascript']
    assert observed == [('python', b'safe('), ('javascript', b'safe('),
                        ('python', b'forbidden(value)'), ('javascript', b'forbidden(value)')]


def test_ast_preparation_stays_lazy_for_regex_match_and_unmatched_scope(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import _stream_rule_ast as module

    rules = [_rule('name = "regex"\ncondition = "safe"\nastCondition = "eval($ARG)"'),
             _rule('name = "other"\nscope = "tool:edit"\nastCondition = "eval($ARG)"')]
    def unexpected_parse(language):
        pytest.fail(f'unneeded parser preparation: {language}')
    monkeypatch.setattr(module, '_parser_for', unexpected_parse)
    runtime = StreamRuleRuntime(rules, repeat_gap=10, cwd=tmp_path)
    with pytest.raises(StreamRuleInterrupt) as caught:
        runtime.observe(SimpleNamespace(source='tool', delta='', tool_index=0, tool_name='write',
            tool_arguments=json.dumps({'path': 'root.py', 'content': 'safe()'})), turn=0)
    assert [r['rule'] for r in caught.value.matches] == ['regex']
