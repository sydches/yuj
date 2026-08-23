"""Runtime acceptance coverage for checkpoint context compaction."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.compaction import maybe_compact_messages
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.state_writer import project
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.types import SideRequestResult, Usage


class _Tokenizer:
    def count(self, messages, tools=None):
        payload = json.dumps(messages, sort_keys=True)
        tool_payload = json.dumps(tools or [], sort_keys=True)
        return max(1, (len(payload) + len(tool_payload)) // 4)


def _summary(path: str = "src/changed.py") -> str:
    return f"""\
## Long-term goal
Finish the requested integration.
## Mid-term goal
Integrate context compaction.
## Near-term goal
Run the focused compaction tests.
## Constraints
Keep deterministic fallback behavior.
## Progress
Done: updated {path}.
In progress: integration proof.
Blocked: none.
## Key decisions
Use the digest on validation failure because it is deterministic.
## Critical context
Modified path: {path}
"""


def _messages(pairs: int = 24, payload_chars: int = 500) -> list[dict]:
    messages = [
        {"role": "system", "content": "system stays exact"},
        {"role": "user", "content": "task stays exact"},
    ]
    for turn in range(pairs):
        messages.extend(
            [
                {"role": "assistant", "content": f"turn {turn}"},
                {
                    "role": "tool",
                    "tool_call_id": f"call-{turn}",
                    "content": "x" * payload_chars,
                },
            ]
        )
    return messages


def _session(tmp_path, response: str):
    trace_event = {
        "event": "tool_call",
        "session_number": 1,
        "turn_number": 2,
        "tool_name": "edit",
        "args_summary": "path='src/changed.py'",
        "source_write_paths": ["src/changed.py"],
        "write_like": True,
        "outcome": "ok",
        "result_summary": "updated",
    }
    trace_path = tmp_path / ".trace.jsonl"
    trace_path.write_text(json.dumps(trace_event) + "\n")
    calls: list[dict] = []

    def complete(payload):
        calls.append(payload)
        return SideRequestResult(response, Usage(100, 30))

    context = FullTranscript(token_estimator=lambda messages: _Tokenizer().count(messages))
    context.replace_all_messages(_messages())
    emitted: list[dict] = []
    session = SimpleNamespace(
        cfg=SimpleNamespace(
            model="test-model",
            context_size=4000,
            max_tokens_fraction=0.25,
            digest_compaction_safety_margin=0.05,
            digest_compaction_gate_min_mutations=0,
            digest_keep_recent_turns=8,
            compaction_method="checkpoint",
            checkpoint_keep_recent_tokens=120,
            checkpoint_max_summary_tokens=500,
        ),
        client=SimpleNamespace(complete_side_request=complete),
        context=context,
        _trace_path=trace_path,
        _trace_events=[trace_event],
        _compaction_turn=10,
        _session_number=1,
        _server_ctx_cache=4000,
        _tokenizer=_Tokenizer(),
        _tool_schemas=[],
        _output_dedup_cache={},
        _get_server_ctx=lambda: 4000,
    )
    session._emit = lambda event, **fields: emitted.append(
        {"event": event, **fields}
    )
    return session, calls, emitted


def test_checkpoint_branch_replaces_context_and_emits_exact_trace(tmp_path):
    session, calls, emitted = _session(tmp_path, _summary())
    original = list(session.context.get_messages())

    compacted = maybe_compact_messages(session, original)

    assert compacted[0] == original[0]
    assert compacted[1] == original[1]
    assert compacted[2]["role"] == "user"
    assert "<summary>" in compacted[2]["content"]
    assert session.context.get_messages() == compacted
    assert len(calls) == 1
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    event = emitted[-1]
    assert event["event"] == "compaction"
    assert event["method"] == "checkpoint"
    assert event["fallback"] == ""
    assert event["tokens_after"] < event["tokens_before"]
    assert event["first_kept_turn"] >= 0


def test_bad_checkpoint_uses_digest_and_records_fallback(tmp_path):
    session, calls, emitted = _session(tmp_path, "## Long-term goal\nmissing")

    compacted = maybe_compact_messages(session, session.context.get_messages())

    assert len(calls) == 1
    assert any("Compacted history" in str(message.get("content")) for message in compacted)
    assert emitted[-1]["method"] == "checkpoint"
    assert emitted[-1]["fallback"] == "digest"


def test_two_close_checkpoint_compactions_force_digest_for_session(tmp_path):
    session, _, emitted = _session(tmp_path, _summary())
    first = maybe_compact_messages(session, session.context.get_messages())
    session._compaction_turn = 16
    grown = list(first) + _messages(pairs=18)[2:]
    session.context.replace_all_messages(grown)

    maybe_compact_messages(session, grown)

    assert [event["method"] for event in emitted] == ["checkpoint", "checkpoint"]
    assert session._compaction_method_override == "digest"


def test_compaction_projection_uses_raw_metadata_not_model_summary():
    event = {
        "event": "compaction",
        "session_number": 2,
        "turn_number": 12,
        "tokens_before": 9000,
        "tokens_after": 4000,
        "first_kept_turn": 8,
        "method": "checkpoint",
        "fallback": "",
        "summary": "model text must not be projected",
    }
    state = project([event], max_result_chars=1000)

    assert state["state"]["last_compaction"] == {
        "session_number": 2,
        "turn_number": 12,
        "tokens_before": 9000,
        "tokens_after": 4000,
        "first_kept_turn": 8,
        "method": "checkpoint",
        "fallback": "",
    }
    assert "model text" not in json.dumps(state)


def test_side_request_omits_tools_and_does_not_advance_transcript(tmp_path):
    captured: list[dict] = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="summary", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        model_dump_json=lambda: "{}",
    )
    completions = SimpleNamespace(
        create=lambda **payload: captured.append(payload) or response
    )
    client = LlamaClient.__new__(LlamaClient)
    client.cfg = SimpleNamespace(model="same-model")
    client.profile = None
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._transcript_path = None
    client._transcript_file = None
    client._transcript_call_n = 0
    transcript = tmp_path / "transcript.log"
    client.set_transcript(transcript)

    result = client.complete_side_request(
        {
            "model": "same-model",
            "messages": [{"role": "user", "content": "summarize"}],
            "max_tokens": 50,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    )

    assert result.content == "summary"
    assert len(captured) == 1
    assert "tools" not in captured[0]
    assert "tool_choice" not in captured[0]
    assert captured[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert client._transcript_call_n == 0
    assert transcript.read_text() == ""
    client.close_transcript()


@pytest.mark.parametrize(
    ("overlay", "match"),
    [
        ("[context]\ncompaction_method='other'\n", "compaction_method"),
        ("[context]\ncheckpoint_keep_recent_tokens=-1\n", "keep_recent_tokens"),
        ("[context]\ncheckpoint_max_summary_tokens=0\n", "max_summary_tokens"),
    ],
)
def test_checkpoint_config_rejects_invalid_values(tmp_path, overlay, match):
    path = tmp_path / "invalid.toml"
    path.write_text(overlay)
    with pytest.raises(ValueError, match=match):
        load_config(user_config=[path])


def test_checkpoint_config_defaults_are_public():
    cfg = load_config()
    assert cfg.compaction_method == "digest"
    assert cfg.checkpoint_keep_recent_tokens == 0
    assert cfg.checkpoint_max_summary_tokens == 4000
