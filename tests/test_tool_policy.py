"""Focused tests for declarative per-tool permission policy."""
from __future__ import annotations

import json

import pytest

from scripts.llm_solver.harness.tool_specs import ACTIVE_TOOL_NAMES
from scripts.llm_solver.harness.tool_policy import (
    DEFAULT_PERMISSION_RULE,
    PERMISSION_ASK_FALLBACKS,
    PERMISSION_DECISIONS,
    PERMISSION_MATCH_FIELDS,
    PermissionApprovalRequired,
    PermissionDenied,
    PermissionPolicy,
    PermissionPolicyError,
    normalize_ask_fallback,
    permission_guarded_dispatch,
    permission_match_argument,
    require_tool_permission,
)


def test_public_decision_and_fallback_contracts():
    assert PERMISSION_DECISIONS == ("allow", "ask", "deny")
    assert PERMISSION_ASK_FALLBACKS == ("deny", "allow")


def test_last_matching_rule_wins_for_bash_command():
    policy = PermissionPolicy.from_rule_tables(
        {
            "bash": {
                "*": "ask",
                "git *": "allow",
                "rm *": "deny",
            }
        }
    )

    git = policy.evaluate(
        tool_name="bash",
        arguments={"cmd": "git status --short"},
        runtime_mode="assistant",
    )
    remove = policy.evaluate(
        tool_name="bash",
        arguments={"cmd": "rm build.log"},
        runtime_mode="assistant",
    )
    other = policy.evaluate(
        tool_name="bash",
        arguments={"cmd": "python -m pytest"},
        runtime_mode="assistant",
    )

    assert (git.rule, git.decision, git.rule_ordinal) == ("git *", "allow", 1)
    assert (remove.rule, remove.decision, remove.rule_ordinal) == (
        "rm *",
        "deny",
        2,
    )
    assert (other.rule, other.decision, other.rule_ordinal) == ("*", "ask", 0)


def test_later_catch_all_can_override_an_earlier_specific_rule():
    policy = PermissionPolicy.from_rule_tables(
        {"bash": {"git *": "allow", "*": "ask"}}
    )
    result = policy.evaluate(
        tool_name="bash",
        arguments={"cmd": "git status"},
        runtime_mode="assistant",
    )
    assert result.rule == "*"
    assert result.decision == "ask"


def test_star_and_question_are_the_only_wildcards():
    # Put the specific rules last in a second layer so they win when matched.
    policy = PermissionPolicy.from_rule_tables(
        {"read": {"*": "ask"}},
        {"read": {"docs/?.md": "allow", "docs/[ab].md": "deny"}},
    )
    assert policy.evaluate(
        tool_name="read", arguments={"path": "docs/a.md"}, runtime_mode="assistant"
    ).decision == "allow"
    # Brackets are literals, not fnmatch's undocumented character-class tier.
    assert policy.evaluate(
        tool_name="read",
        arguments={"path": "docs/[ab].md"},
        runtime_mode="assistant",
    ).decision == "deny"
    assert policy.evaluate(
        tool_name="read", arguments={"path": "docs/ab.md"}, runtime_mode="assistant"
    ).decision == "ask"


@pytest.mark.parametrize("tool_name", ("read", "write", "edit"))
def test_file_tool_rules_match_path_not_other_arguments(tool_name):
    policy = PermissionPolicy.from_rule_tables(
        {tool_name: {"*": "deny", "docs/*.md": "allow"}}
    )
    arguments = {
        "path": "docs/guide.md",
        "content": "outside/*.py",
        "old_str": "outside/*.py",
        "new_str": "outside/*.py",
    }

    resolution = policy.evaluate(
        tool_name=tool_name, arguments=arguments, runtime_mode="assistant"
    )

    assert resolution.argument_field == "path"
    assert resolution.decision == "allow"
    assert resolution.rule == "docs/*.md"


def test_bash_rule_matches_command_not_a_path_named_argument():
    policy = PermissionPolicy.from_rule_tables(
        {"bash": {"*": "deny", "git *": "allow"}}
    )
    resolution = policy.evaluate(
        tool_name="bash",
        arguments={"cmd": "git diff", "path": "blocked/secret"},
        runtime_mode="assistant",
    )
    assert resolution.argument_field == "cmd"
    assert resolution.decision == "allow"


def test_search_path_defaults_match_the_handler_contract():
    assert permission_match_argument("glob", {"pattern": "*.py"}) == ("path", ".")
    assert permission_match_argument("grep", {"pattern": "value"}) == ("path", ".")


def test_global_tool_rule_composes_with_exact_tool_override():
    policy = PermissionPolicy.from_rule_tables(
        {"*": "deny", "read": {"docs/*": "allow"}}
    )

    assert policy.evaluate(
        tool_name="write",
        arguments={"path": "docs/a.md", "content": "x"},
        runtime_mode="assistant",
    ).decision == "deny"
    assert policy.evaluate(
        tool_name="read",
        arguments={"path": "docs/a.md"},
        runtime_mode="assistant",
    ).decision == "allow"


def test_later_rule_table_is_an_agent_or_session_override_layer():
    base = {"edit": {"*": "deny"}}
    override = {"edit": {"docs/*.mdx": "allow"}}
    policy = PermissionPolicy.from_rule_tables(base, override)

    allowed = policy.evaluate(
        tool_name="edit",
        arguments={"path": "docs/index.mdx"},
        runtime_mode="assistant",
    )
    denied = policy.evaluate(
        tool_name="edit",
        arguments={"path": "src/app.py"},
        runtime_mode="assistant",
    )

    assert (allowed.decision, allowed.rule_ordinal) == ("allow", 1)
    assert (denied.decision, denied.rule_ordinal) == ("deny", 0)


def test_whole_table_and_per_tool_string_shorthands():
    deny_all = PermissionPolicy.from_rule_tables("deny")
    allow_reads = PermissionPolicy.from_rule_tables(
        "deny", {"read": "allow"}
    )

    assert deny_all.evaluate(
        tool_name="read", arguments={"path": "x"}, runtime_mode="assistant"
    ).denied
    assert allow_reads.evaluate(
        tool_name="read", arguments={"path": "x"}, runtime_mode="assistant"
    ).allowed


def test_no_matching_rule_preserves_legacy_allow_default_and_is_traceable():
    policy = PermissionPolicy.from_rule_tables({"bash": {"rm *": "deny"}})
    resolution = policy.evaluate(
        tool_name="read",
        arguments={"path": "README.md"},
        runtime_mode="measurement",
    )
    assert resolution.allowed
    assert resolution.rule == DEFAULT_PERMISSION_RULE
    assert resolution.rule_ordinal is None
    assert resolution.trace_fields() == {
        "tool": "read",
        "rule": DEFAULT_PERMISSION_RULE,
        "decision": "allow",
    }


def test_ask_raises_approval_request_in_assistant_mode_before_handler():
    policy = PermissionPolicy.from_rule_tables({"bash": {"*": "ask"}})
    calls = []

    with pytest.raises(PermissionApprovalRequired) as caught:
        permission_guarded_dispatch(
            policy=policy,
            tool_name="bash",
            arguments={"cmd": "python -m pytest"},
            runtime_mode="assistant",
            handler=lambda: calls.append(True) or "unexpected",
        )

    assert calls == []
    resolution = caught.value.resolution
    assert resolution.approval_required
    assert resolution.approval_reason() == (
        "permission rule '*' requires operator approval for tool 'bash'"
    )
    assert resolution.trace_fields() == {
        "tool": "bash",
        "rule": "*",
        "decision": "ask",
    }


@pytest.mark.parametrize("ask_fallback", ("deny", "allow"))
def test_ask_is_always_denied_in_measurement_mode(ask_fallback):
    policy = PermissionPolicy.from_rule_tables({"bash": {"*": "ask"}})

    resolution = policy.evaluate(
        tool_name="bash",
        arguments={"cmd": "git status"},
        runtime_mode="measurement",
        ask_fallback=ask_fallback,
    )

    assert resolution.configured_decision == "ask"
    assert resolution.decision == "deny"
    assert resolution.fallback_reason == "measurement mode"
    with pytest.raises(PermissionDenied):
        require_tool_permission(
            policy=policy,
            tool_name="bash",
            arguments={"cmd": "git status"},
            runtime_mode="measurement",
            ask_fallback=ask_fallback,
        )


@pytest.mark.parametrize(
    ("fallback", "expected"), (("deny", "deny"), ("allow", "allow"))
)
def test_ask_fallback_is_used_only_when_assistant_approval_is_unavailable(
    fallback, expected
):
    policy = PermissionPolicy.from_rule_tables({"read": {"*": "ask"}})
    resolution = policy.evaluate(
        tool_name="read",
        arguments={"path": "README.md"},
        runtime_mode="assistant",
        ask_fallback=fallback,
        approval_available=False,
    )
    assert resolution.decision == expected
    assert resolution.fallback_reason == "approval transport unavailable"


def test_explicit_deny_never_calls_handler_and_returns_value_free_envelope():
    private_path = "private/customer-secret.txt"
    policy = PermissionPolicy.from_rule_tables({"read": {"private/*": "deny"}})
    calls = []

    with pytest.raises(PermissionDenied) as caught:
        permission_guarded_dispatch(
            policy=policy,
            tool_name="read",
            arguments={"path": private_path},
            runtime_mode="assistant",
            handler=lambda: calls.append(True) or "unexpected",
        )

    assert calls == []
    envelope = caught.value.error_envelope()
    assert envelope.startswith("ERROR: ")
    assert private_path not in envelope
    payload = json.loads(envelope.removeprefix("ERROR: "))
    assert payload["error"] == {
        "decision": "deny",
        "message": "permission rule 'private/*' denies tool 'read'",
        "rule": "private/*",
        "tool": "read",
        "type": "permission_denied",
        "version": 1,
    }


def test_allow_dispatches_handler_once_and_pairs_trace_decision():
    policy = PermissionPolicy.from_rule_tables(
        {"bash": {"*": "deny", "git *": "allow"}}
    )
    calls = []
    dispatched = permission_guarded_dispatch(
        policy=policy,
        tool_name="bash",
        arguments={"cmd": "git status"},
        runtime_mode="assistant",
        handler=lambda: calls.append(True) or "clean",
    )

    assert dispatched.result == "clean"
    assert calls == [True]
    assert dispatched.resolution.trace_fields() == {
        "tool": "bash",
        "rule": "git *",
        "decision": "allow",
    }


def test_unknown_tool_rules_match_a_stable_canonical_arguments_object():
    field, rendered = permission_match_argument(
        "task", {"z": 1, "a": "two"}
    )
    assert field == "arguments"
    assert rendered == '{"a":"two","z":1}'


def test_match_field_registry_covers_current_shipped_tools():
    assert set(PERMISSION_MATCH_FIELDS) == set(ACTIVE_TOOL_NAMES)


@pytest.mark.parametrize(
    ("tables", "match"),
    [
        (({"bash": {"*": "prompt"}},), "must be one of"),
        (({"bash": ["allow"]},), "must be a table"),
        (({1: {"*": "allow"}},), "tool key"),
        (([],), "must be a table"),
    ],
)
def test_malformed_rule_tables_fail_closed(tables, match):
    with pytest.raises(PermissionPolicyError, match=match):
        PermissionPolicy.from_rule_tables(*tables)


@pytest.mark.parametrize("value", ("ask", "prompt", "", None))
def test_ask_fallback_accepts_only_allow_or_deny(value):
    with pytest.raises(PermissionPolicyError, match="ask_fallback"):
        normalize_ask_fallback(value)
