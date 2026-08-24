"""Focused tests for tool-schema validation and constrained decoding."""
from __future__ import annotations

import json

import pytest

from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.tool_specs import SCHEMA_TOOL_NAMES
from scripts.llm_solver.harness.tool_validation import (
    CONSTRAINED_DECODING_MODES,
    SCHEMA_VALIDATION_MODES,
    ToolSchemaDefinitionError,
    ToolSchemaSet,
    attach_constrained_decoding,
    guarded_tool_dispatch,
    normalize_constrained_decoding_mode,
    normalize_schema_validation_mode,
    resolve_constrained_decoding,
    validate_json_instance,
)


VALID_ARGUMENTS = {
    "bash": {"cmd": "git status --short"},
    "bash_poll": {"proc_id": "p0001", "timeout_s": 2.5},
    "bash_kill": {"proc_id": "p0001"},
    "read": {"path": "src/app.py", "offset": 2, "limit": 20},
    "write": {"path": "src/app.py", "content": "value = 1\n"},
    "edit": {
        "path": "src/app.py",
        "old_str": "value = 1",
        "new_str": "value = 2",
    },
    "glob": {"pattern": "**/*.py", "path": "src", "page": 1},
    "grep": {"pattern": "value", "path": "src", "glob": "*.py", "page": 2},
    "checkpoint": {"goal": "Inspect the implementation."},
    "rewind": {"report": "The implementation uses a safe boundary."},
    "lsp": {
        "kind": "definition", "path": "src/app.py",
        "line": 4, "character": 2,
    },
    "run_tests": {"path": "tests", "k": "unit", "last_failed": False},
    "list_definitions": {"path": "src/app.py"},
    "apply_patch": {"patch": "*** Begin Patch\n*** End Patch"},
    "load_tools": {"names": ["write", "run_tests"]},
    "done": {"message": "All checks pass."},
}

INVALID_ARGUMENTS = (
    ("bash", {}, "$.cmd", "required"),
    ("bash", {"cmd": 7}, "$.cmd", "type"),
    ("bash_poll", {}, "$.proc_id", "required"),
    ("bash_kill", {"proc_id": 7}, "$.proc_id", "type"),
    ("read", {}, "$.path", "required"),
    ("read", {"path": 7}, "$.path", "type"),
    ("write", {"path": "x.py"}, "$.content", "required"),
    ("write", {"path": "x.py", "content": False}, "$.content", "type"),
    (
        "edit",
        {"path": "x.py", "old_str": "before"},
        "$.new_str",
        "required",
    ),
    (
        "edit",
        {"path": "x.py", "old_str": "before", "new_str": []},
        "$.new_str",
        "type",
    ),
    ("glob", {}, "$.pattern", "required"),
    ("glob", {"pattern": "*.py", "page": "one"}, "$.page", "type"),
    ("grep", {}, "$.pattern", "required"),
    ("grep", {"pattern": "x", "path": 3}, "$.path", "type"),
    ("checkpoint", {}, "$.goal", "required"),
    ("checkpoint", {"goal": 7}, "$.goal", "type"),
    ("rewind", {}, "$.report", "required"),
    ("rewind", {"report": 7}, "$.report", "type"),
    ("run_tests", {"last_failed": "false"}, "$.last_failed", "type"),
    ("list_definitions", {}, "$.path", "required"),
    ("list_definitions", {"path": None}, "$.path", "type"),
    ("apply_patch", {}, "$.patch", "required"),
    ("apply_patch", {"patch": {"text": "patch"}}, "$.patch", "type"),
    ("load_tools", {}, "$.names", "required"),
    ("load_tools", {"names": "write"}, "$.names", "type"),
    ("done", {"message": 1}, "$.message", "type"),
)


@pytest.fixture(scope="module")
def schemas() -> ToolSchemaSet:
    return ToolSchemaSet.from_openai_tools(get_tool_schemas())


def test_mode_names_are_the_public_contract():
    assert SCHEMA_VALIDATION_MODES == ("off", "reject")
    assert CONSTRAINED_DECODING_MODES == ("off", "json_schema", "grammar")


def test_schema_set_uses_the_active_model_facing_order(schemas):
    assert schemas.names == SCHEMA_TOOL_NAMES
    assert set(VALID_ARGUMENTS) == set(SCHEMA_TOOL_NAMES)


@pytest.mark.parametrize("tool_name", SCHEMA_TOOL_NAMES)
def test_every_shipped_tool_accepts_representative_valid_arguments(
    schemas, tool_name
):
    validation = schemas.validate(tool_name, VALID_ARGUMENTS[tool_name])
    assert validation.valid, [error.as_dict() for error in validation.errors]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "path", "keyword"), INVALID_ARGUMENTS
)
def test_invalid_arguments_are_repairable_and_never_reach_handler(
    schemas, tool_name, arguments, path, keyword
):
    calls = []

    guarded = guarded_tool_dispatch(
        mode="reject",
        schemas=schemas,
        tool_name=tool_name,
        arguments=arguments,
        handler=lambda: calls.append(tool_name) or "HANDLER RESULT",
    )

    assert guarded.dispatched is False
    assert calls == []
    assert guarded.validation is not None
    assert guarded.validation.valid is False
    assert guarded.result.startswith("ERROR: ")
    payload = json.loads(guarded.result.removeprefix("ERROR: "))
    error = payload["error"]
    assert error["type"] == "tool_schema_reject"
    assert error["tool"] == tool_name
    assert any(
        item["path"] == path and item["keyword"] == keyword
        for item in error["errors"]
    )
    assert guarded.validation.trace_fields() == {
        "tool": tool_name,
        "errors": [item.as_dict() for item in guarded.validation.errors],
    }


def test_rejection_does_not_echo_invalid_argument_values(schemas):
    secret_value = "private-value-that-must-not-enter-trace-metadata"
    validation = schemas.validate("bash", {"cmd": [secret_value]})

    assert validation.valid is False
    assert secret_value not in validation.error_envelope()
    assert secret_value not in json.dumps(validation.trace_fields())
    assert validation.errors[0].actual == "array"


def test_validated_dispatch_calls_handler_once_for_a_valid_call(schemas):
    calls = []
    guarded = guarded_tool_dispatch(
        mode="reject",
        schemas=schemas,
        tool_name="read",
        arguments={"path": "README.md"},
        handler=lambda: calls.append("called") or "contents",
    )

    assert guarded.result == "contents"
    assert guarded.dispatched is True
    assert guarded.validation is not None and guarded.validation.valid
    assert calls == ["called"]


def test_validation_off_preserves_legacy_dispatch(schemas):
    guarded = guarded_tool_dispatch(
        mode="off",
        schemas=schemas,
        tool_name="read",
        arguments={},
        handler=lambda: "legacy result",
    )

    assert guarded == guarded.__class__("legacy result", True, None)


def test_unknown_tool_is_a_schema_reject_not_a_handler_call(schemas):
    calls = []
    guarded = guarded_tool_dispatch(
        mode="reject",
        schemas=schemas,
        tool_name="invented_tool",
        arguments={},
        handler=lambda: calls.append(True) or "unexpected",
    )

    assert guarded.dispatched is False
    assert calls == []
    assert guarded.validation is not None
    assert guarded.validation.errors[0].keyword == "tool"


def test_json_integer_does_not_accept_boolean(schemas):
    validation = schemas.validate("read", {"path": "x.py", "offset": True})
    assert validation.errors[0].path == "$.offset"
    assert validation.errors[0].expected == "integer"
    assert validation.errors[0].actual == "boolean"


def test_supported_json_schema_features_report_stable_nested_paths():
    schema = {
        "type": "object",
        "properties": {
            "mode": {"enum": ["fast", "safe"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"count": {"type": "integer", "minimum": 1}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["mode", "items"],
        "additionalProperties": False,
    }

    errors = validate_json_instance(
        {"mode": "turbo", "items": [{"count": 0}, {}], "extra": True}, schema
    )

    assert [(error.path, error.keyword) for error in errors] == [
        ("$.mode", "enum"),
        ("$.items[0].count", "minimum"),
        ("$.items[1].count", "required"),
        ("$.extra", "additionalProperties"),
    ]


def test_schema_compilation_rejects_duplicate_tools_and_unknown_keywords():
    one = get_tool_schemas()[0]
    with pytest.raises(ToolSchemaDefinitionError, match="duplicate tool"):
        ToolSchemaSet.from_openai_tools([one, one])

    malformed = json.loads(json.dumps(one))
    malformed["function"]["parameters"]["madeUpConstraint"] = True
    with pytest.raises(ToolSchemaDefinitionError, match="unsupported keyword"):
        ToolSchemaSet.from_openai_tools([malformed])

    missing_object_type = json.loads(json.dumps(one))
    missing_object_type["function"]["parameters"].pop("type")
    with pytest.raises(ToolSchemaDefinitionError, match="declare type='object'"):
        ToolSchemaSet.from_openai_tools([missing_object_type])


def test_schema_compilation_rejects_circular_local_references():
    circular = [
        {
            "type": "function",
            "function": {
                "name": "recursive",
                "parameters": {
                    "type": "object",
                    "$defs": {"node": {"$ref": "#/$defs/node"}},
                    "properties": {"node": {"$ref": "#/$defs/node"}},
                },
            },
        }
    ]
    with pytest.raises(ToolSchemaDefinitionError, match="circular"):
        ToolSchemaSet.from_openai_tools(circular)


def test_local_schema_refs_survive_embedding_in_constrained_wrapper():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "$defs": {"path": {"type": "string"}},
                    "properties": {"path": {"$ref": "#/$defs/path"}},
                    "required": ["path"],
                },
            },
        }
    ]
    schema_set = ToolSchemaSet.from_openai_tools(tools)

    assert schema_set.validate("lookup", {"path": "README.md"}).valid
    wrapper = schema_set.constrained_json_schema()
    assert validate_json_instance(
        {"name": "lookup", "arguments": {"path": "README.md"}}, wrapper
    ) == ()
    assert "args-lookup ::=" in schema_set.constrained_grammar()


def test_non_string_mapping_key_is_not_treated_as_valid_json_object(schemas):
    validation = schemas.validate("done", {1: "not a JSON field"})
    assert validation.valid is False
    assert validation.errors[0].expected == "string"


def test_constrained_json_schema_has_one_strict_branch_per_active_tool(schemas):
    wrapper = schemas.constrained_json_schema()
    assert len(wrapper["oneOf"]) == len(SCHEMA_TOOL_NAMES)

    branch_names = [
        branch["properties"]["name"]["const"] for branch in wrapper["oneOf"]
    ]
    assert tuple(branch_names) == SCHEMA_TOOL_NAMES
    assert all(branch["additionalProperties"] is False for branch in wrapper["oneOf"])

    for tool_name in SCHEMA_TOOL_NAMES:
        call = {"name": tool_name, "arguments": VALID_ARGUMENTS[tool_name]}
        assert validate_json_instance(call, wrapper) == ()


def test_runtime_and_constrained_schemas_share_a_closed_argument_surface(schemas):
    runtime = schemas.validate("read", {"path": "x.py", "future": 1})
    assert runtime.valid is False
    assert runtime.errors[0].keyword == "additionalProperties"

    wrapper = schemas.constrained_json_schema()
    errors = validate_json_instance(
        {"name": "read", "arguments": {"path": "x.py", "future": 1}},
        wrapper,
    )
    assert any(error.keyword == "additionalProperties" for error in errors)


def test_generated_grammar_covers_every_valid_shipped_tool_call(schemas):
    grammar = schemas.constrained_grammar()

    root = grammar.splitlines()[0]
    for tool_name in SCHEMA_TOOL_NAMES:
        slug = tool_name.replace("_", "-")
        assert f"call-{slug}" in root
        assert f"args-{slug} ::=" in grammar
        # The corresponding canonical wrapper is schema-valid; the grammar
        # branch is generated from this same strict branch and includes both
        # wrapper-member orders and every allowed argument-member ordering.
        call = {"name": tool_name, "arguments": VALID_ARGUMENTS[tool_name]}
        assert validate_json_instance(call, schemas.constrained_json_schema()) == ()

    assert 'json-string ::= "\\\"" json-char* "\\\""' in grammar
    assert "ws ::= [ \\t\\n\\r]*" in grammar


@pytest.mark.parametrize("mode", ("json_schema", "grammar"))
def test_enabled_constrained_mode_builds_llama_request_extra(schemas, mode):
    resolution = resolve_constrained_decoding(
        mode=mode,
        schemas=schemas,
        supports_constrained_tools=True,
    )

    assert resolution.enabled is True
    assert resolution.effective_mode == mode
    if mode == "json_schema":
        assert set(resolution.request_extra) == {"json_schema"}
        assert resolution.request_extra["json_schema"]["oneOf"]
    else:
        assert set(resolution.request_extra) == {"grammar", "grammar_type"}
        assert resolution.request_extra["grammar_type"] == "tool_calls"
        assert str(resolution.request_extra["grammar"]).startswith("root ::=")


def test_profile_capability_gate_falls_back_to_runtime_validation(schemas):
    resolution = resolve_constrained_decoding(
        mode="grammar",
        schemas=schemas,
        supports_constrained_tools=False,
    )

    assert resolution.enabled is False
    assert resolution.effective_mode == "off"
    assert resolution.request_extra == {}
    assert resolution.fallback_reason == "profile_unsupported"


def test_attach_constrained_decoding_is_defensive_and_policy_owned(schemas):
    resolution = resolve_constrained_decoding(
        mode="json_schema",
        schemas=schemas,
        supports_constrained_tools=True,
    )
    payload = {
        "model": "served-model",
        "messages": [],
        "extra_body": {"cache_prompt": True, "grammar": "stale"},
    }

    attached = attach_constrained_decoding(payload, resolution)
    attached["extra_body"]["json_schema"]["title"] = "mutated copy"

    assert payload["extra_body"] == {"cache_prompt": True, "grammar": "stale"}
    assert attached["extra_body"]["cache_prompt"] is True
    assert "grammar" not in attached["extra_body"]
    assert resolution.request_extra["json_schema"]["title"] == "Yuj tool call"


def test_off_resolution_does_not_rewrite_manual_request_fields(schemas):
    resolution = resolve_constrained_decoding(
        mode="off", schemas=schemas, supports_constrained_tools=True
    )
    payload = {"extra_body": {"grammar": "manual"}}
    assert attach_constrained_decoding(payload, resolution) == payload


@pytest.mark.parametrize(
    ("normalizer", "value", "setting"),
    [
        (normalize_schema_validation_mode, "repair", "schema_validation"),
        (normalize_constrained_decoding_mode, "regex", "constrained_decoding"),
        (normalize_schema_validation_mode, None, "schema_validation"),
    ],
)
def test_mode_validation_fails_closed(normalizer, value, setting):
    with pytest.raises(ValueError, match=setting):
        normalizer(value)
