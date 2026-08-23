"""Focused contract tests for model-written fresh-session handoffs."""
from __future__ import annotations

import json
from typing import Any

from scripts.llm_solver.harness._loop.handoff_summary import (
    HANDOFF_FALLBACK,
    HandoffResult,
    build_handoff_request,
    generate_handoff,
    insert_handoff_into_resume_prompt,
    serialize_trace_for_handoff,
    validate_handoff_summary,
)


class CharTokenizer:
    def count(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> int:
        content = sum(len(str(message.get("content") or "")) + 5 for message in messages)
        return content + (len(json.dumps(tools)) if tools else 0)


def _valid_handoff(*, path: str = "src/app.py") -> str:
    return f"""\
## Goal
Fix the context-continuity behavior.
## Done
Updated {path} and ran the focused test.
## In progress
Checking the fresh-session integration boundary.
## Blocked
Protected loop seams are not editable in this leaf.
## Key decisions
Keep the current mechanical resume text as the fallback floor.
## Critical paths/errors
The critical path is {path}; test_handoff_resume previously failed.
## Next step
Wire the validated handoff into the session rollover call site."""


def _trace_events() -> list[dict[str, Any]]:
    return [
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 4,
            "tool_name": "read",
            "args_summary": "path='old-session.py'",
            "output_snippet": "OLD SESSION SENTINEL",
        },
        {
            "event": "tool_call",
            "session_number": 2,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='src/app.py'",
            "action_summary": "read(path='src/app.py')",
            "reasoning": "Inspect the owning file before editing.",
            "output_snippet": "HEAD" + ("x" * 5_000) + "TAIL",
        },
        {
            "event": "tool_call",
            "session_number": 2,
            "turn_number": 1,
            "tool_name": "edit",
            "args_summary": "path='src/app.py'",
            "action_summary": "edit(path='src/app.py')",
            "write_like": True,
            "source_write_paths": ["src/app.py"],
            "outcome": "ok",
            "pass_fail": "pass",
            "output_snippet": "OK",
        },
        {
            "event": "session_end",
            "session_number": 2,
            "finish_reason": "context_full",
            "turns": 2,
        },
    ]


def test_trace_serialization_is_session_scoped_and_clips_results() -> None:
    rendered = serialize_trace_for_handoff(
        _trace_events(),
        session_number=2,
        tool_result_chars=160,
    )

    assert "OLD SESSION SENTINEL" not in rendered
    assert "Inspect the owning file before editing." in rendered
    assert "[Tool call]\nread(path='src/app.py')" in rendered
    assert "chars omitted by compaction overflow guard" in rendered
    assert "HEAD" in rendered and "TAIL" in rendered
    assert "finish_reason=context_full; turns=2" in rendered


def test_trace_serialization_bounds_whole_event_blocks() -> None:
    rendered = serialize_trace_for_handoff(
        _trace_events(),
        session_number=2,
        max_chars=220,
        tool_result_chars=80,
    )

    assert len(rendered) <= 220
    assert "older trace event(s) omitted" in rendered
    assert "[Session end]" in rendered


def test_handoff_request_has_no_tools_and_disables_thinking() -> None:
    request = build_handoff_request(
        model="local-model",
        task="Fix it.",
        serialized_trace="[Turn 1]\n[Tool call]\nread(path='src/app.py')",
        modified_files=("src/app.py",),
        max_tokens=700,
        previous_handoff="Prior decision.",
    )

    assert request["model"] == "local-model"
    assert request["max_tokens"] == 700
    assert "tools" not in request
    assert "tool_choice" not in request
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "another software-engineering agent" in request["messages"][0]["content"]
    assert "Prior decision." in request["messages"][1]["content"]
    assert '["src/app.py"]' in request["messages"][1]["content"]


def test_handoff_validation_rejects_missing_header_path_and_oversize() -> None:
    tokenizer = CharTokenizer()
    missing_header = validate_handoff_summary(
        _valid_handoff().replace("## Key decisions\n", ""),
        modified_files=("src/app.py",),
        tokenizer=tokenizer,
        max_tokens=2_000,
    )
    assert missing_header.valid is False
    assert "required headers" in missing_header.reason

    missing_path = validate_handoff_summary(
        _valid_handoff(path="src/other.py"),
        modified_files=("src/app.py",),
        tokenizer=tokenizer,
        max_tokens=2_000,
    )
    assert missing_path.valid is False
    assert missing_path.reason == "handoff omitted modified file: src/app.py"

    oversize = validate_handoff_summary(
        _valid_handoff(),
        modified_files=("src/app.py",),
        tokenizer=tokenizer,
        max_tokens=20,
    )
    assert oversize.valid is False
    assert "exceeds max tokens" in oversize.reason
    assert oversize.tokens > 20


def test_generate_handoff_calls_model_once_and_validates_modified_files() -> None:
    captured: list[dict[str, Any]] = []

    def call_model(request: dict[str, Any]) -> str:
        captured.append(request)
        return _valid_handoff()

    result = generate_handoff(
        model="local-model",
        task="Fix the context-continuity behavior.",
        trace_events=_trace_events(),
        tokenizer=CharTokenizer(),
        max_tokens=2_000,
        session_number=2,
        max_history_chars=2_000,
        previous_handoff="Earlier work remains valid.",
        call_model=call_model,
    )

    assert len(captured) == 1
    assert result.valid is True
    assert result.fallback is None
    assert result.reason == "ok"
    assert result.modified_files == ("src/app.py",)
    assert result.tokens > 0
    assert "OLD SESSION SENTINEL" not in captured[0]["messages"][1]["content"]
    assert "Earlier work remains valid." in captured[0]["messages"][1]["content"]


def test_invalid_handoff_leaves_mechanical_resume_prompt_byte_identical() -> None:
    calls = 0

    def call_model(_request: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return "missing every required header"

    result = generate_handoff(
        model="local-model",
        task="Fix it.",
        trace_events=_trace_events(),
        tokenizer=CharTokenizer(),
        max_tokens=2_000,
        call_model=call_model,
    )
    mechanical = (
        "Task:\nFix it.\n\n"
        "Previous session ended after 2 turns: context_full.\n\n"
        "Context was 95% full. This session starts fresh.\n\n"
        "Continue."
    )

    assembled = insert_handoff_into_resume_prompt(
        mechanical,
        task="Fix it.",
        handoff=result,
    )

    assert calls == 1
    assert result.valid is False
    assert result.fallback == HANDOFF_FALLBACK
    assert assembled is mechanical
    assert assembled.encode() == mechanical.encode()


def test_valid_handoff_is_between_task_and_unchanged_mechanical_tail() -> None:
    mechanical_tail = (
        "Previous session ended after 2 turns: context_full.\n\n"
        "Context was 95% full. This session starts fresh.\n\n"
        "Continue."
    )
    mechanical = f"Task:\nFix it.\n\n{mechanical_tail}"
    handoff = HandoffResult(
        valid=True,
        fallback=None,
        reason="ok",
        request={},
        raw_response=_valid_handoff(),
        summary=_valid_handoff(),
        tokens=400,
        modified_files=("src/app.py",),
    )

    assembled = insert_handoff_into_resume_prompt(
        mechanical,
        task="Fix it.",
        handoff=handoff,
    )

    assert assembled.startswith("Task:\nFix it.\n\n<handoff>\n## Goal")
    assert "\n</handoff>\n\n" in assembled
    assert assembled.split("\n</handoff>\n\n", 1)[1] == mechanical_tail


def test_valid_handoff_with_unexpected_prompt_shape_falls_back_unchanged() -> None:
    mechanical = "Previous session has no task prefix."
    handoff = HandoffResult(
        valid=True,
        fallback=None,
        reason="ok",
        request={},
        raw_response=_valid_handoff(),
        summary=_valid_handoff(),
        tokens=400,
        modified_files=("src/app.py",),
    )

    assembled = insert_handoff_into_resume_prompt(
        mechanical,
        task="Fix it.",
        handoff=handoff,
    )

    assert assembled is mechanical
