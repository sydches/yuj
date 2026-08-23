"""Focused tests for compound-aware bash command redirects."""
from __future__ import annotations

import re

import pytest

from scripts.llm_solver._shared.classification import is_error_result
from scripts.llm_solver.harness.command_redirect import (
    RedirectRule,
    find_redirect,
    load_redirect_rules,
    parse_redirect_rules,
    render_redirect_error,
    split_shell_fragments,
    strip_leading_assignments,
)


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
