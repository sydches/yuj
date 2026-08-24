"""Composability tests — verify the SoD "add a file, it works" contract.

Each quirk absorber and plugin set in the SoD claims that a new entry
can be dropped in without touching other layers. These tests make
those claims mechanically enforceable: an agent or contributor who
breaks the plugin shape sees a red test, not a silent regression at
run time.

Covered plugin sets:
  - Language quirks (task_formats): synthetic TOML → load_output_control + load_output_parser
  - Context strategies: custom ContextManager subclass → conforms to the protocol + works with Session
  - Context mode registry: CLI modes resolve from strategy-module registration
  - Config toggles: multi-overlay layering via load_config(user_config=[...])
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.llm_solver.bash_quirks import (
    OutputControl,
    OutputParser,
    load_output_control,
    load_output_parser,
    parse_structured,
    render_digest,
)
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


# ─── Context mode registry ───────────────────────────────────────────────

def test_context_mode_registry_has_expected_modes():
    """CLI-visible context modes come from the context-strategy registry."""
    from scripts.llm_solver.harness.context_strategies import list_context_modes

    assert list_context_modes() == (
        "full",
        "compact",
        "concise",
        "slot",
        "yuj",
        "yconcise",
        "yslot",
        "stateful",
        "compound",
        "focused_compound",
        "compound_selective",
        "salience",
        "halflife",
    )


def test_context_mode_registry_resolves_stateful():
    """stateful mode must resolve explicitly to SolverStateContext."""
    from scripts.llm_solver.harness.context_strategies import (
        SolverStateContext,
        resolve_context_class,
    )

    assert resolve_context_class("stateful") is SolverStateContext


def test_context_mode_registry_exposes_first_class_records():
    """Each CLI-visible context mode has class and metadata in one record."""
    from scripts.llm_solver.harness.context_strategies import (
        ContextMode,
        list_context_modes,
        resolve_context_class,
        resolve_context_mode,
    )

    for mode in list_context_modes():
        record = resolve_context_mode(mode)
        assert isinstance(record, ContextMode)
        assert record.name == mode
        assert record.cls is resolve_context_class(mode)
        assert record.metadata.cli_order >= 0
        assert record.metadata.file_freshness != "strategy-defined"
        assert record.metadata.injection_support != "strategy-defined"


def test_full_context_mode_is_registered_uniformly():
    """full is a discovered mode record, not a registry-only special case."""
    from scripts.llm_solver.harness.context import FullTranscript
    from scripts.llm_solver.harness.context_strategies import resolve_context_mode

    record = resolve_context_mode("full")
    assert record.cls is FullTranscript
    assert record.metadata.message_shape == "append-only transcript"
    assert record.metadata.section_order == ("append_only_chronological_messages",)
    assert record.metadata.constructor_config_attrs == {}


def test_context_modes_own_constructor_config_metadata():
    """Context construction kwargs live on the mode record."""
    from scripts.llm_solver.harness.context_strategies import (
        CompoundSelectiveContext,
        resolve_context_mode,
        resolve_context_mode_for_class,
    )

    compact = resolve_context_mode("compact")
    assert (
        compact.metadata.constructor_config_attrs["recent_results_chars"]
        == "recent_tool_results_chars"
    )

    selective = resolve_context_mode("compound_selective")
    assert (
        selective.metadata.constructor_config_attrs["selective_trace_lines"]
        == "compound_selective_trace_lines"
    )
    assert resolve_context_mode_for_class(CompoundSelectiveContext) is selective


def test_context_modes_own_budget_metadata():
    """Context contract budget fields live on the mode record."""
    from scripts.llm_solver.harness.context_strategies import resolve_context_mode

    full = resolve_context_mode("full")
    focused = resolve_context_mode("focused_compound")
    selective = resolve_context_mode("compound_selective")

    assert "solver_trace_lines" in full.metadata.budget_config_attrs
    assert (
        "focused_compound_trace_lines"
        in focused.metadata.budget_config_attrs
    )
    assert (
        "compound_selective_trace_lines"
        in selective.metadata.budget_config_attrs
    )
    assert (
        "compound_selective_trace_lines"
        not in focused.metadata.budget_config_attrs
    )


# ─── Tool surface registry ──────────────────────────────────────────────

def test_tool_specs_drive_surface_metadata():
    """Cross-cutting tool facts come from first-class specs."""
    from scripts.llm_solver.harness.schemas import get_tool_schemas
    from scripts.llm_solver.harness.tool_specs import (
        ACTIVE_TOOL_NAMES,
        CODE_MODE_SCHEMA_TOOL_NAMES,
        GUARDRAIL_MUTATION_TOOL_NAMES,
        NATIVE_ENVELOPE_PREFIXES,
        PARALLEL_READ_SAFE_TOOL_NAMES,
        PROFILE_GATE_ATTRS,
        SCHEMA_TOOL_NAMES,
        is_native_envelope,
    )
    from scripts.llm_solver.harness.tools import build_tool_registry

    schema_names = {
        schema["function"]["name"]
        for schema in get_tool_schemas("minimal")
    }
    code_schema_names = {
        schema["function"]["name"]
        for schema in get_tool_schemas("minimal", code_mode=True)
    }
    assert set(ACTIVE_TOOL_NAMES) == schema_names | code_schema_names
    assert tuple(
        schema["function"]["name"] for schema in get_tool_schemas("minimal")
    ) == SCHEMA_TOOL_NAMES
    assert tuple(
        schema["function"]["name"]
        for schema in get_tool_schemas("minimal", code_mode=True)
    ) == CODE_MODE_SCHEMA_TOOL_NAMES
    assert tuple(build_tool_registry().handlers) == ACTIVE_TOOL_NAMES
    assert PARALLEL_READ_SAFE_TOOL_NAMES == frozenset({"read", "glob", "grep"})
    assert GUARDRAIL_MUTATION_TOOL_NAMES == frozenset(
        {"write", "edit", "apply_patch", "udiff", "str_replace"}
    )
    assert PROFILE_GATE_ATTRS == {
        "run_tests": "tools_run_tests_enabled",
        "list_definitions": "tools_list_definitions_enabled",
        "lsp": "lsp_tool_enabled",
        "bash_poll": "tools_background_enabled",
        "bash_kill": "tools_background_enabled",
        "task": "tools_task_enabled",
        "think": "tools_think_enabled",
        "write_todos": "tools_todos_enabled",
        "list_functions": "tools_exec_cell_enabled",
        "get_function_details": "tools_exec_cell_enabled",
        "exec_cell": "tools_exec_cell_enabled",
        "load_tools": "tools_lazy_loading_enabled",
        "checkpoint": "tools_checkpoint_enabled",
        "rewind": "tools_checkpoint_enabled",
    }
    assert NATIVE_ENVELOPE_PREFIXES == (
        "<test_results",
        "<list_definitions",
        "<apply_patch",
    )
    assert is_native_envelope("<test_results status=\"passed\">ok</test_results>")


# ─── Trace event schema registry ────────────────────────────────────────

def test_trace_event_specs_drive_compatibility_views():
    """Trace event sets and required-field maps derive from event specs."""
    from scripts.llm_solver.harness._loop.trace_schema import (
        KNOWN_TRACE_EVENT_TYPES,
        TRACE_EVENT_REQUIRED_FIELDS,
        TRACE_EVENT_SPECS,
    )

    assert KNOWN_TRACE_EVENT_TYPES == frozenset(
        spec.event_type for spec in TRACE_EVENT_SPECS
    )
    assert TRACE_EVENT_REQUIRED_FIELDS == {
        spec.event_type: spec.required_fields
        for spec in TRACE_EVENT_SPECS
    }
    assert TRACE_EVENT_REQUIRED_FIELDS["tool_call"] == frozenset(
        {"session_number", "turn_number", "tool_name"}
    )
    assert TRACE_EVENT_REQUIRED_FIELDS["todos"] == frozenset(
        {"session_number", "turn_number", "todos"}
    )


# ─── Language-quirks file-drop ──────────────────────────────────────────

_SYNTHETIC_TOML = r'''
name = "fixture"
description = "Synthetic runner for composability test"

verification_patterns = [
    "(?:^|[\\s/\"])fixture-runner\\b",
]

[output_control]
failure_only_flag = "--fail-fast"
passed_marker = "ok:"
failed_marker = "fail:"

[output_parser.summary]
passed = "(\\d+)\\s+ok"
failed = "(\\d+)\\s+fail"

[output_parser.per_test]
regex = "^(?P<verdict>OK|FAIL)\\s+(?P<test_id>\\S+)"
'''.strip()


def test_language_quirks_drop_in_output_control(tmp_path: Path):
    """A synthetic language_quirks TOML loads into a valid OutputControl."""
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(_SYNTHETIC_TOML)

    oc = load_output_control(fixture)
    assert oc is not None
    assert isinstance(oc, OutputControl)
    assert oc.failure_only_flag == "--fail-fast"
    assert oc.passed_marker == "ok:"
    assert oc.failed_marker == "fail:"
    assert len(oc.verification_patterns) == 1


def test_language_quirks_drop_in_output_parser(tmp_path: Path):
    """A synthetic language_quirks TOML loads into a working OutputParser."""
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(_SYNTHETIC_TOML)

    parser = load_output_parser(fixture)
    assert parser is not None
    assert isinstance(parser, OutputParser)
    # summary regexes parse the fixture's output format
    parsed = parse_structured("5 ok, 2 fail in 0.3s", parser)
    assert parsed["summary"] == {"passed": 5, "failed": 2}
    # per_test regex captures individual verdicts; verdicts are normalized
    # to canonical PASSED/FAILED so downstream consumers don't need to
    # know each runner's vocabulary.
    parsed = parse_structured("OK tests/a.py::test_x\nFAIL tests/a.py::test_y", parser)
    assert parsed["tests"] == {
        "tests/a.py::test_x": "PASSED",
        "tests/a.py::test_y": "FAILED",
    }


def test_language_quirks_render_digest_on_synthetic_output(tmp_path: Path):
    """Digest rendering works for a synthetic runner's parse results."""
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(_SYNTHETIC_TOML)
    parser = load_output_parser(fixture)
    parsed = parse_structured("3 ok, 1 fail in 0.1s\nFAIL tests/z::t", parser)
    digest = render_digest(parsed)
    assert "3 passed" in digest
    assert "1 failed" in digest
    assert "tests/z::t" in digest


def test_language_quirks_missing_parser_block_returns_none(tmp_path: Path):
    """A TOML without [output_parser] yields None — no crash."""
    minimal = tmp_path / "minimal.toml"
    minimal.write_text(textwrap.dedent('''
        name = "minimal"
        verification_patterns = [ "foo" ]
    ''').strip())
    assert load_output_parser(minimal) is None


# ─── Context-strategy plugin shape ──────────────────────────────────────

def test_custom_context_strategy_conforms_to_protocol():
    """A user-defined ContextManager subclass implements every abstract method.

    The SoD declares context strategies are swappable — Session consumes
    any ContextManager. Prove it by building a minimal subclass and
    confirming Python accepts it (ABC enforcement).
    """
    from scripts.llm_solver.harness.context import ContextManager, chars_div_4

    class MyStrategy(ContextManager):
        def __init__(self):
            super().__init__(chars_div_4)
            self._messages: list[dict] = []

        def add_system(self, content: str) -> None:
            self._messages.append({"role": "system", "content": content})

        def add_user(self, content: str) -> None:
            self._messages.append({"role": "user", "content": content})

        def add_assistant(self, message: dict) -> None:
            self._messages.append(message)

        def add_tool_result(self, tool_call_id, content, *, tool_name="",
                            cmd_signature="", gate_blocked=False) -> None:
            self._messages.append({"role": "tool", "tool_call_id": tool_call_id,
                                   "content": content})

        def get_messages(self) -> list[dict]:
            return self._messages

        def estimate_tokens(self) -> int:
            return self._token_estimator(self._messages)

        def message_count(self) -> int:
            return len(self._messages)

    # Instantiation would fail if any abstract method were missing.
    ctx = MyStrategy()
    ctx.add_system("sys")
    ctx.add_user("hi")
    assert ctx.get_messages()[0] == {"role": "system", "content": "sys"}
    assert ctx.estimate_tokens() > 0


# ─── Config toggle layering ─────────────────────────────────────────────

def test_multi_config_layers_later_wins(tmp_path: Path):
    """--config can be passed multiple times; later overlays override earlier.

    This makes arbitrary flag combinations composable without pre-baking
    every subset of toggles.
    """
    from scripts.llm_solver.config import load_config

    a = tmp_path / "a.toml"
    a.write_text('[loop]\nduplicate_guard_enabled = true\nrumination_enabled = true\n')
    b = tmp_path / "b.toml"
    b.write_text('[loop]\nrumination_enabled = false\n')

    # Load single overlay — rumination stays on.
    cfg_a = load_config(user_config=[a])
    assert cfg_a.duplicate_guard_enabled is True
    assert cfg_a.rumination_enabled is True

    # Layer b on top — rumination flips off, duplicate carries through.
    cfg_ab = load_config(user_config=[a, b])
    assert cfg_ab.duplicate_guard_enabled is True
    assert cfg_ab.rumination_enabled is False

    # Reverse order — a wins over b for rumination.
    cfg_ba = load_config(user_config=[b, a])
    assert cfg_ba.rumination_enabled is True


def test_single_config_path_accepted(tmp_path: Path):
    """load_config still accepts a single Path for backward compatibility."""
    from scripts.llm_solver.config import load_config

    a = tmp_path / "a.toml"
    a.write_text('[loop]\nduplicate_warn_count = 7\n')
    cfg = load_config(user_config=a)  # not a list
    assert cfg.duplicate_warn_count == 7


# ─── Guardrail registry composability ───────────────────────────────────

def test_guardrail_registry_override_applies_without_session_edits():
    """Session consumes injected guardrail registry callables by name."""
    from scripts.llm_solver.harness.guardrails import (
        Action,
        Decision,
        build_guardrail_registry,
    )
    from scripts.llm_solver.config import load_config
    from scripts.llm_solver.harness.loop import Session

    cfg = load_config(overrides={"max_turns": 2, "tokenizer_id": ""})
    client = MagicMock()
    client.chat.return_value = TurnResult(
        content="",
        tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "x"})],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=3, completion_tokens=1),
    )
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}

    def hard_stop(*_args, **_kwargs):
        return Decision(Action.END, reason="forced_guard_end")

    registry = build_guardrail_registry(
        turn_pre_overrides={"intent_gate": hard_stop},
    )
    session = Session(
        cfg,
        client,
        "sys",
        "prompt",
        "/tmp",
        guardrail_registry=registry,
    )
    result = session.run()
    assert result.finish_reason == "forced_guard_end"


def test_guardrail_registry_validation_fails_when_required_name_missing():
    """Missing required call-site names fail fast at validation time."""
    from scripts.llm_solver.harness.guardrails import (
        build_guardrail_registry,
        validate_guardrail_registry,
    )

    registry = build_guardrail_registry()
    registry.tool_post_dispatch.pop("error_ladder")
    with pytest.raises(ValueError):
        validate_guardrail_registry(registry)
