"""Focused tests for compound-aware bash command redirects."""
from __future__ import annotations

import re
from io import StringIO
import json
from unittest.mock import MagicMock

import pytest

from scripts.llm_solver._shared.classification import is_error_result
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.command_redirect import (
    RedirectRule,
    find_redirect,
    load_redirect_rules,
    parse_redirect_rules,
    render_redirect_error,
    split_shell_fragments,
    strip_leading_assignments,
)
from scripts.llm_solver.harness.tools import ToolRegistry, dispatch

from _config_helpers import make_config


def rule(name, pattern, tool, message="use the dedicated tool", **kwargs):
    return RedirectRule(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE | re.DOTALL),
        tool=tool,
        message=message,
        **kwargs,
    )


READ_RULES = [
    rule("cat_read", r"^(?:cat|head|tail|less|more)\b", "read"),
    rule("grep_read", r"^(?:grep|rg|ag|ack)\b", "grep"),
    rule("find_glob", r"^(?:find|fd|locate)\b", "glob"),
]
WRITE_RULES = [
    rule("sed_edit", r"^(?:sed\s+-i|perl\s+-i|awk\s+-i\s+inplace)\b", "edit"),
    rule("redirect_write", r"^(?:echo|printf)\b[^\n]*>>?", "write"),
]


def test_splitter_handles_quoted_escaped_and_top_level_operators():
    command = (
        "echo 'a && b | c'; FOO=x grep 'a|b' file.py "
        "|& sed -n \"1;2p\" file.py && echo a\\&b"
    )

    fragments = split_shell_fragments(command)

    assert [fragment.text for fragment in fragments] == [
        "echo 'a && b | c'",
        "FOO=x grep 'a|b' file.py",
        'sed -n "1;2p" file.py',
        "echo a\\&b",
    ]
    assert [fragment.operator_after for fragment in fragments] == [
        ";", "|&", "&&", "",
    ]
    assert fragments[2].stdin_from_pipe is True


def test_splitter_keeps_command_substitution_and_subshell_together():
    fragments = split_shell_fragments(
        "echo $(printf 'a;b|c') && (printf x | wc -l); cat file.py"
    )

    assert [fragment.text for fragment in fragments] == [
        "echo $(printf 'a;b|c')",
        "(printf x | wc -l)",
        "cat file.py",
    ]


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ("FOO=bar LC_ALL='C UTF-8' rg x file", "rg x file"),
        ("/usr/bin/env FOO=bar rg x file", "/usr/bin/env FOO=bar rg x file"),
        ("FOO=bar", ""),
        ("rg x file", "rg x file"),
    ],
)
def test_strip_leading_assignments(fragment, expected):
    assert strip_leading_assignments(fragment) == expected


def test_fragment_match_finds_command_after_semicolon_and_assignment():
    decision = find_redirect(
        "echo ready; LC_ALL=C cat src/app.py",
        READ_RULES,
        active_tools={"bash", "read", "grep", "glob"},
        read_side_enabled=True,
    )

    assert decision is not None
    assert decision.rule_name == "cat_read"
    assert decision.tool == "read"
    assert decision.fragment == "cat src/app.py"
    assert decision.fragment_index == 1


def test_quoted_operator_does_not_hide_read_redirect():
    decision = find_redirect(
        "grep 'left|right && center' src/app.py",
        READ_RULES,
        active_tools={"grep"},
        read_side_enabled=True,
    )

    assert decision is not None
    assert decision.fragment_index is None


def test_read_rules_are_off_by_default_gate():
    assert find_redirect(
        "cat src/app.py",
        READ_RULES,
        active_tools={"read"},
        read_side_enabled=False,
    ) is None


def test_rule_is_inert_when_target_tool_is_disabled():
    assert find_redirect(
        "find src -name '*.py'",
        READ_RULES,
        active_tools={"bash", "read", "grep"},
        read_side_enabled=True,
    ) is None


def test_write_side_rule_remains_active_when_read_side_is_off():
    decision = find_redirect(
        "sed -i 's/a/b/' src/app.py",
        WRITE_RULES,
        active_tools={"edit"},
        read_side_enabled=False,
    )

    assert decision is not None
    assert decision.tool == "edit"


def test_pipe_stage_that_consumes_stdin_is_not_redirected():
    decision = find_redirect(
        "printf 'needle' | grep needle",
        READ_RULES,
        active_tools={"grep"},
        read_side_enabled=True,
    )

    assert decision is None


@pytest.mark.parametrize(
    "command",
    [
        "grep -c needle src/app.py",
        "rg --count needle src/app.py",
        "rg -l needle src/app.py",
        "cat src/app.py | wc -l",
        "cat src/app.py | grep -c needle",
    ],
)
def test_aggregate_reads_are_not_blocked(command):
    assert find_redirect(
        command,
        READ_RULES,
        active_tools={"read", "grep", "glob"},
        read_side_enabled=True,
    ) is None


def test_nonaggregate_pipeline_producer_is_redirected():
    decision = find_redirect(
        "cat src/app.py | grep needle",
        READ_RULES,
        active_tools={"read", "grep"},
        read_side_enabled=True,
    )

    assert decision is not None
    assert decision.tool == "read"


def test_fragment_aware_false_only_checks_full_command():
    rules = [rule(
        "full_only", r"^cat\b", "read", fragment_aware=False
    )]

    assert find_redirect(
        "echo ok; cat src.py",
        rules,
        active_tools={"read"},
        read_side_enabled=True,
    ) is None


def test_first_rule_wins_deterministically():
    rules = [
        rule("first", r"^cat\b", "read", "first message"),
        rule("second", r"^cat\b", "read", "second message"),
    ]

    decision = find_redirect(
        "cat src.py",
        rules,
        active_tools={"read"},
        read_side_enabled=True,
    )

    assert decision is not None
    assert decision.rule_name == "first"


def test_unified_error_envelope_is_ladder_countable_and_escaped():
    decision = find_redirect(
        "cat src.py",
        [rule("cat", r"^cat\b", "read", "use read <now>")],
        active_tools={"read"},
        read_side_enabled=True,
    )
    assert decision is not None

    result = render_redirect_error(decision)

    assert result.startswith(
        '<tool_result tool_name="bash" status="error" '
        'error_kind="redirect_rule" redirect_tool="read" v="1">'
    )
    assert "Blocked: use read &lt;now&gt;" in result
    assert is_error_result(result) is True
    assert decision.trace_fields() == {
        "rule": "cat", "tool": "read", "fragment_index": None,
    }


def test_parse_and_load_redirect_table(tmp_path):
    path = tmp_path / "forbidden.toml"
    path.write_text(
        """
[[redirect]]
name = "cat_read"
pattern = '''^cat\\b'''
flags = "i"
tool = "read"
message = "use read"
fragment_aware = true

[[redirect]]
name = "case_sensitive"
pattern = '''^RG\\b'''
flags = ""
tool = "grep"
message = "use grep"
fragment_aware = false
""".strip()
    )

    rules = load_redirect_rules(path)

    assert [item.name for item in rules] == ["cat_read", "case_sensitive"]
    assert rules[0].pattern.search("CAT file")
    assert not rules[1].pattern.search("rg pattern file")
    assert rules[1].fragment_aware is False


def test_malformed_rule_is_skipped(caplog):
    rules = parse_redirect_rules([
        {"pattern": "[", "tool": "read", "message": "bad"},
        {"pattern": "^cat", "tool": "read", "message": "good"},
    ])

    assert len(rules) == 1
    assert "redirect rule 1 skipped" in caplog.text


def test_canonical_redirect_rules_cover_public_default_families():
    rules = load_redirect_rules()

    assert [(item.name, item.tool) for item in rules] == [
        ("read_file", "read"),
        ("search_text", "grep"),
        ("find_paths", "glob"),
    ]
    active = {"read", "grep", "glob"}
    cases = {
        "cat src.py": "read",
        "rg needle src": "grep",
        "find src -name '*.py'": "glob",
    }
    for command, target in cases.items():
        decision = find_redirect(
            command, rules, active_tools=active, read_side_enabled=True
        )
        assert decision is not None
        assert decision.tool == target


@pytest.mark.parametrize(
    "command",
    [
        "echo value > out.txt",
        "cat source.txt > out.txt",
        "cat source.txt >> out.txt",
        "cat <<EOF > out.txt\nvalue\nEOF",
        "sed -i 's/a/b/' src.py",
        "echo value | tee out.txt",
    ],
)
def test_canonical_redirect_rules_do_not_block_sandboxed_writes(command):
    assert find_redirect(
        command,
        load_redirect_rules(),
        active_tools={"read", "write", "edit", "grep", "glob"},
        read_side_enabled=True,
    ) is None


def test_session_transform_loader_includes_canonical_redirect_rules():
    from scripts.llm_solver.harness._loop.session_io import _load_bash_transforms

    loaded = _load_bash_transforms(make_config())

    assert len(loaded) == 6
    assert [item.name for item in loaded[3]] == [
        "read_file", "search_text", "find_paths",
    ]


def test_dispatch_redirect_prevents_handler_and_emits_event(tmp_path):
    calls = []
    events = []
    metadata = {}

    def handler(arguments, cwd, cfg):
        calls.append((arguments, cwd, cfg))
        return "handler ran"

    result = dispatch(
        "bash",
        {"cmd": "cat src.py"},
        cwd=str(tmp_path),
        cfg=make_config(tools_bash_redirect_read_side=True),
        redirect_rules=load_redirect_rules(),
        active_tools={"bash", "read"},
        redirect_event_sink=events.append,
        execution_metadata=metadata,
        tool_registry=ToolRegistry(handlers={"bash": handler}),
    )

    assert calls == []
    assert metadata["executed"] is False
    assert result.count("<tool_result") == 1
    assert 'status="error" error_kind="redirect_rule"' in result
    assert "Blocked: use the read tool" in result
    assert events == [{
        "event": "redirect_rule",
        "rule": "read_file",
        "tool": "read",
        "fragment_index": None,
    }]


def test_dispatch_read_gate_and_active_tool_gate_are_runtime_effective(tmp_path):
    calls = []

    def handler(arguments, cwd, cfg):
        calls.append(arguments["cmd"])
        return "shell output"

    registry = ToolRegistry(handlers={"bash": handler})
    rules = load_redirect_rules()
    disabled_by_knob = dispatch(
        "bash", {"cmd": "cat src.py"}, cwd=str(tmp_path),
        cfg=make_config(
            tools_bash_redirect_read_side=False,
            tools_unified_envelope_enabled=False,
        ),
        redirect_rules=rules, active_tools={"read"}, tool_registry=registry,
    )
    disabled_by_surface = dispatch(
        "bash", {"cmd": "cat src.py"}, cwd=str(tmp_path),
        cfg=make_config(
            tools_bash_redirect_read_side=True,
            tools_unified_envelope_enabled=False,
        ),
        redirect_rules=rules, active_tools={"bash"}, tool_registry=registry,
    )
    blocked = dispatch(
        "bash", {"cmd": "cat src.py"}, cwd=str(tmp_path),
        cfg=make_config(tools_bash_redirect_read_side=True),
        redirect_rules=rules, active_tools={"read"}, tool_registry=registry,
    )

    assert disabled_by_knob == "shell output"
    assert disabled_by_surface == "shell output"
    assert calls == ["cat src.py", "cat src.py"]
    assert 'error_kind="redirect_rule"' in blocked


def test_redirect_result_honors_output_cap(tmp_path):
    long_rule = rule("long", r"^cat\b", "read", "x" * 2000)
    result = dispatch(
        "bash", {"cmd": "cat src.py"}, cwd=str(tmp_path),
        cfg=make_config(
            tools_bash_redirect_read_side=True,
            max_output_chars=240,
        ),
        redirect_rules=[long_rule], active_tools={"read"},
        tool_registry=ToolRegistry(
            handlers={"bash": lambda _args, _cwd, _cfg: "unexpected"}
        ),
    )

    assert len(result) <= 240
    assert "redirect message truncated" in result
    assert result.endswith("</tool_result>")


def test_redirect_config_default_overlay_and_validation(tmp_path):
    assert load_config().tools_bash_redirect_read_side is False

    overlay = tmp_path / "redirect.toml"
    overlay.write_text("[tools]\nbash_redirect_read_side = true\n")
    assert load_config(user_config=overlay).tools_bash_redirect_read_side is True

    overlay.write_text('[tools]\nbash_redirect_read_side = "yes"\n')
    with pytest.raises(ValueError, match="tools.bash_redirect_read_side"):
        load_config(user_config=overlay)


def test_redirect_envelope_counts_as_one_stable_guardrail_error_class():
    from scripts.llm_solver.harness._guardrails.checks_post import error_ladder
    from scripts.llm_solver.harness._guardrails.extractors import _error_signature
    from scripts.llm_solver.harness._guardrails.state import GuardrailState

    result = render_redirect_error(
        find_redirect(
            "cat src.py", READ_RULES, active_tools={"read"},
            read_side_enabled=True,
        )
    )
    state = GuardrailState()
    cfg = make_config(error_nudge_threshold=99)

    error_ladder(state, cfg, tc_name="bash", result=result)

    assert state.consecutive_errors == {"bash": 1}
    assert state.same_class_error_signature == "redirect_rule"
    assert _error_signature(result) == "redirect_rule"


def test_session_uses_profile_filtered_tools_and_traces_redirect(tmp_path):
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage

    trace = StringIO()
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(
            content=None,
            tool_calls=[ToolCall(
                id="call-1", name="bash", arguments={"cmd": "cat src.py"}
            )],
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        ),
        TurnResult(
            content="done", tool_calls=[], finish_reason="stop",
            usage=Usage(prompt_tokens=12, completion_tokens=2),
        ),
    ]
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": "done"},
    ]
    session = Session(
        make_config(
            max_turns=3,
            tools_bash_redirect_read_side=True,
            rumination_nudge_threshold=999,
        ),
        client,
        "system",
        "task",
        str(tmp_path),
        trace_file=trace,
        session_number=7,
        redirect_rules=load_redirect_rules(),
    )

    session.run()
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    redirect = next(event for event in events if event["event"] == "redirect_rule")
    call = next(event for event in events if event["event"] == "tool_call")

    assert redirect["session_number"] == 7
    assert redirect["turn_number"] == 0
    assert redirect["rule"] == "read_file"
    assert redirect["tool"] == "read"
    assert call["outcome"] == "error"
    assert call["error_class"] == "redirect_rule"

    from scripts.llm_solver.harness._loop.trace_schema import (
        TRACE_EVENT_REQUIRED_FIELDS,
    )
    assert TRACE_EVENT_REQUIRED_FIELDS["redirect_rule"] == frozenset({
        "session_number", "turn_number", "rule", "tool", "fragment_index",
    })
