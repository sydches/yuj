"""Focused contract tests for model-written checkpoint leaf behavior."""
from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.llm_solver.harness._loop.checkpoint_summary import (
    CHECKPOINT_HEADERS,
    CHECKPOINT_MESSAGE_PREFIX,
    MechanicalAppendix,
    build_checkpoint_request,
    build_mechanical_appendix,
    generate_checkpoint,
    loop_guard_forces_digest,
    select_checkpoint_cut,
    serialize_checkpoint_head,
    summary_token_limit,
    validate_checkpoint_candidate,
)


class CharTokenizer:
    """Deterministic tokenizer whose units make cut assertions transparent."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    def count(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> int:
        self.calls.append((messages, tools))
        message_chars = sum(
            len(str(message.get("content") or ""))
            + len(json.dumps(message.get("tool_calls") or (), sort_keys=True))
            + 5
            for message in messages
        )
        tool_chars = len(json.dumps(tools, sort_keys=True)) if tools else 0
        return message_chars + tool_chars


def _assistant_turn(number: int, *, result_chars: int = 36) -> list[dict[str, Any]]:
    call_id = f"call-{number}"
    return [
        {
            "role": "assistant",
            "content": f"assistant-{number}-" + ("a" * 12),
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": f"src/file_{number}.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"result-{number}-" + ("r" * result_chars),
        },
    ]


def _messages(turns: int = 3) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "SYSTEM SENTINEL"},
        {"role": "user", "content": "Fix the checkpoint behavior."},
    ]
    for number in range(turns):
        messages.extend(_assistant_turn(number))
    return messages


def _valid_summary(*, modified_path: str = "src/app.py") -> str:
    return f"""\
## Long-term goal
Fix the checkpoint behavior.
## Mid-term goal
Complete the validated checkpoint compaction kernel.
## Near-term goal
Run the focused checkpoint tests.
## Constraints
Keep the deterministic digest as the fallback floor.
## Progress
Done: Updated {modified_path}.
In progress: Checking summary validation.
Blocked: None.
## Key decisions
Use an assistant-turn cut because it preserves tool pairing.
## Critical context
The modified file is {modified_path}; failing test is test_checkpoint_shape."""


def test_cut_walks_complete_assistant_turns_until_tail_target() -> None:
    tokenizer = CharTokenizer()
    messages = _messages(3)
    newest_tokens = tokenizer.count(messages[-2:])
    two_turn_tokens = tokenizer.count(messages[-4:])

    cut = select_checkpoint_cut(
        messages,
        tokenizer,
        keep_recent_tokens=newest_tokens + 1,
    )

    assert cut.target_satisfied is True
    assert cut.tail_tokens == two_turn_tokens
    assert cut.first_kept_turn == 1
    assert cut.first_kept_message_index == 4
    assert cut.tail[0]["role"] == "assistant"
    assert [message["role"] for message in cut.tail[:2]] == ["assistant", "tool"]
    assert cut.tail[0]["tool_calls"][0]["id"] == cut.tail[1]["tool_call_id"]
    assert cut.prefix == tuple(messages[:2])


def test_cut_rejects_unanswered_tool_call() -> None:
    messages = _messages(1)
    messages.pop()

    with pytest.raises(ValueError, match="missing contiguous result"):
        select_checkpoint_cut(
            messages,
            CharTokenizer(),
            keep_recent_tokens=1,
        )


def test_head_serialization_uses_raw_turns_and_clips_tool_results() -> None:
    messages = _messages(3)
    previous_checkpoint = {
        "role": "user",
        "content": (
            f"{CHECKPOINT_MESSAGE_PREFIX}\n"
            "<summary>STALE SYNTHETIC SENTINEL</summary>"
        ),
    }
    messages.insert(4, previous_checkpoint)
    messages[6]["content"] = "HEAD" + ("x" * 5_000) + "TAIL"

    serialized = serialize_checkpoint_head(
        messages,
        stop_message_index=7,
        start_turn=1,
        tool_result_chars=200,
    )

    assert "STALE SYNTHETIC SENTINEL" not in serialized
    assert "assistant-0" not in serialized
    assert "assistant-1" in serialized
    assert "[Assistant]" in serialized
    assert "[Tool call]" in serialized
    assert "[Tool result]" in serialized
    assert "chars omitted by compaction overflow guard" in serialized
    assert "HEAD" in serialized and "TAIL" in serialized


def test_mechanical_appendix_comes_only_from_trace_fields() -> None:
    events = [
        {
            "event": "tool_call",
            "tool_name": "read",
            "args_summary": "path='src/read_me.py', offset=1",
        },
        {
            "event": "tool_call",
            "tool_name": "edit",
            "args_summary": "path='truncated.py'",
            "write_like": True,
            "source_write_paths": ["src/app.py"],
            "outcome": "ok",
            "pass_fail": "pass",
        },
        {
            "event": "tool_call",
            "tool_name": "edit",
            "write_like": True,
            "source_write_paths": ["src/not_modified.py"],
            "outcome": "blocked",
            "gate_blocked": True,
        },
        {
            "event": "tool_call",
            "tool_name": "run_tests",
            "args_summary": "path='tests/test_app.py'",
            "action_summary": "run_tests(path='tests/test_app.py')",
            "pass_fail": "fail",
            "output_snippet": "collected 2 tests\n1 failed, 1 passed",
        },
    ]

    appendix = build_mechanical_appendix(events)

    assert appendix.read_files == ("src/read_me.py",)
    assert appendix.modified_files == ("src/app.py",)
    assert appendix.mutation_count == 1
    assert appendix.last_test_runner_digest == (
        "run_tests(path='tests/test_app.py') | fail | 1 failed, 1 passed"
    )
    rendered = appendix.render()
    assert '<modified-files>\n["src/app.py"]\n</modified-files>' in rendered
    assert "<mutation-count>1</mutation-count>" in rendered


def test_checkpoint_request_omits_tools_and_turns_thinking_off() -> None:
    assert summary_token_limit(
        reserve_tokens=10_000,
        configured_max_tokens=9_000,
    ) == 4_000
    assert summary_token_limit(
        reserve_tokens=1_000,
        configured_max_tokens=9_000,
    ) == 800

    request = build_checkpoint_request(
        model="local-model",
        task="Fix it.",
        serialized_head="[Assistant]\nI will inspect src/app.py",
        previous_summary="Earlier checkpoint.",
        modified_files=("src/app.py",),
        max_tokens=800,
    )

    assert request["model"] == "local-model"
    assert request["max_tokens"] == 800
    assert "tools" not in request
    assert "tool_choice" not in request
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "Treat the task" in request["messages"][0]["content"]
    assert "exact file paths" in request["messages"][1]["content"]


def test_validation_rejects_missing_header_and_modified_file() -> None:
    appendix = MechanicalAppendix((), ("src/app.py",), "", 1)
    prefix = _messages(0)
    tail = _assistant_turn(2)

    missing_header = _valid_summary().replace("## Key decisions\n", "")
    header_result = validate_checkpoint_candidate(
        missing_header,
        prefix=prefix,
        tail=tail,
        appendix=appendix,
        tokenizer=CharTokenizer(),
        budget=10_000,
        tokens_before=20_000,
    )
    assert header_result.valid is False
    assert "required headers" in header_result.reason

    missing_path = _valid_summary(modified_path="src/other.py")
    path_result = validate_checkpoint_candidate(
        missing_path,
        prefix=prefix,
        tail=tail,
        appendix=appendix,
        tokenizer=CharTokenizer(),
        budget=10_000,
        tokens_before=20_000,
    )
    assert path_result.valid is False
    assert path_result.reason == "checkpoint omitted modified file: src/app.py"


def test_validation_rejects_candidate_that_does_not_shrink() -> None:
    result = validate_checkpoint_candidate(
        _valid_summary(),
        prefix=_messages(0),
        tail=_assistant_turn(2),
        appendix=MechanicalAppendix((), ("src/app.py",), "", 1),
        tokenizer=CharTokenizer(),
        budget=10_000,
        tokens_before=10,
    )

    assert result.valid is False
    assert "does not shrink prompt" in result.reason
    assert result.compacted_messages is None


def test_generate_checkpoint_places_valid_summary_after_untouched_task() -> None:
    tokenizer = CharTokenizer()
    messages = _messages(3)
    previous_checkpoint = {
        "role": "user",
        "content": (
            f"{CHECKPOINT_MESSAGE_PREFIX}\n"
            "<summary>STALE SYNTHETIC SENTINEL</summary>"
        ),
    }
    messages.insert(4, previous_checkpoint)
    captured: list[dict[str, Any]] = []

    def call_model(request: dict[str, Any]) -> str:
        captured.append(request)
        return _valid_summary()

    result = generate_checkpoint(
        model="local-model",
        messages=messages,
        trace_events=[
            {
                "event": "tool_call",
                "tool_name": "edit",
                "write_like": True,
                "source_write_paths": ["src/app.py"],
                "outcome": "ok",
                "pass_fail": "pass",
            }
        ],
        tokenizer=tokenizer,
        keep_recent_tokens=1,
        max_summary_tokens=4_000,
        budget=3_000,
        tokens_before=5_000,
        previous_summary="PREVIOUS SUMMARY SENTINEL",
        previous_first_kept_turn=1,
        call_model=call_model,
        tools=[{"type": "function", "function": {"name": "read"}}],
    )

    assert result.valid is True
    assert result.fallback is None
    assert len(captured) == 1
    request_text = captured[0]["messages"][1]["content"]
    assert "PREVIOUS SUMMARY SENTINEL" in request_text
    assert "STALE SYNTHETIC SENTINEL" not in request_text
    assert "assistant-0" not in request_text
    assert "assistant-1" in request_text
    assert result.compacted_messages is not None
    compacted = result.compacted_messages
    assert compacted[0] == messages[0]
    assert compacted[1] == messages[1]
    assert compacted[2]["role"] == "user"
    assert compacted[2]["content"].startswith(CHECKPOINT_MESSAGE_PREFIX)
    assert compacted[3]["role"] == "assistant"
    assert compacted[3]["content"].startswith("assistant-2")
    assert result.tokens_after < 3_000
    assert result.tokens_after < result.tokens_before
    assert result.first_kept_turn == 2
    assert all(header in result.model_summary for header in CHECKPOINT_HEADERS)
    assert "<modified-files>" in result.summary_with_appendix


def test_generate_checkpoint_bad_model_output_falls_back_to_digest() -> None:
    calls = 0

    def call_model(_request: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return "not a structured checkpoint"

    result = generate_checkpoint(
        model="local-model",
        messages=_messages(3),
        trace_events=[],
        tokenizer=CharTokenizer(),
        keep_recent_tokens=1,
        max_summary_tokens=1_000,
        budget=3_000,
        tokens_before=5_000,
        call_model=call_model,
    )

    assert calls == 1
    assert result.valid is False
    assert result.fallback == "digest"
    assert result.compacted_messages is None
    assert "required headers" in result.reason


@pytest.mark.parametrize(
    ("turns", "window", "expected"),
    [
        ([], 8, False),
        ([10], 8, False),
        ([10, 18], 8, True),
        ([10, 19], 8, False),
        ([1, 50, 55], 8, True),
    ],
)
def test_loop_guard_switches_after_two_nearby_compactions(
    turns: list[int],
    window: int,
    expected: bool,
) -> None:
    assert loop_guard_forces_digest(
        turns,
        keep_recent_turns=window,
    ) is expected
