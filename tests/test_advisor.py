"""Acceptance coverage for the passive advisor / second-opinion model."""
from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.model_role_runtime import (
    build_model_role_runtime,
)
from scripts.llm_solver.harness._loop._session_setup import build_context_manager
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.context_strategies import resolve_context_class
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.replay_client import ReplayClient
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


FIXTURE_PROFILES = Path(__file__).parent / "fixtures" / "model_role_profiles"


def _turn(
    *,
    content: str | None = None,
    calls: list[ToolCall] | None = None,
    reason: str = "tool_calls",
) -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(calls or []),
        finish_reason=reason,
        usage=Usage(prompt_tokens=10, completion_tokens=2, cached_tokens=0),
    )


class _ScriptedClient:
    profile = None
    is_replay = False

    def __init__(self, *, primary=(), advisor=(), cfg=None):
        self.cfg = cfg
        self.primary = list(primary)
        self.advisor = list(advisor)
        self.chat_calls: list[list[dict]] = []
        self.advisor_calls: list[tuple[list[dict], list[dict]]] = []

    def chat(self, messages, tools, turn=0):
        self.chat_calls.append(json.loads(json.dumps(messages)))
        return self.primary.pop(0)

    def complete_tool_side_request(self, messages, tools, *, turn=0):
        self.advisor_calls.append(
            (
                json.loads(json.dumps(messages)),
                json.loads(json.dumps(tools)),
            )
        )
        return self.advisor.pop(0)

    def build_assistant_message(self, content, tool_calls):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in tool_calls
            ]
        return message

    def query_server_context(self):
        return 0


def _advisor_config(**overrides):
    values = {
        "advisor_enabled": True,
        "advisor_every_n_turns": 1,
        "advisor_immune_turns": 0,
        "advisor_max_note_chars": 200,
        "max_turns": 5,
    }
    values.update(overrides)
    return make_config(**values)


def _record_primary_turn(session: Session, turn: int) -> None:
    session._trace_events.append(
        {
            "event": "turn",
            "session_number": session._session_number,
            "turn_number": turn,
            "role": "main",
        }
    )


def test_public_config_defaults_overlay_validation_and_advisor_role(tmp_path: Path):
    defaults = load_config()
    assert defaults.advisor_enabled is False
    assert defaults.advisor_model == ""
    assert defaults.advisor_endpoint == ""
    assert defaults.advisor_every_n_turns == 5
    assert defaults.advisor_immune_turns == 3
    assert defaults.advisor_max_note_chars == 1200

    overlay = tmp_path / "advisor.toml"
    overlay.write_text(
        """\
[advisor]
enabled = true
model = "review-served-model"
endpoint = "http://127.0.0.1:8181/v1"
every_n_turns = 2
immune_turns = 4
max_note_chars = 333
"""
    )
    cfg = load_config(overlay)
    assert (
        cfg.advisor_enabled,
        cfg.advisor_model,
        cfg.advisor_endpoint,
        cfg.advisor_every_n_turns,
        cfg.advisor_immune_turns,
        cfg.advisor_max_note_chars,
    ) == (
        True,
        "review-served-model",
        "http://127.0.0.1:8181/v1",
        2,
        4,
        333,
    )

    runtime_cfg = make_config(
        model="main-served-model",
        profile_name="_base",
        base_url="http://127.0.0.1:8080/v1",
        advisor_enabled=True,
        advisor_model="review-served-model",
        advisor_endpoint="http://127.0.0.1:8181/v1",
    )
    main = _ScriptedClient(cfg=runtime_cfg)
    main.profile = load_profile("_base", FIXTURE_PROFILES)
    built = []

    def factory(role_cfg, role_profile):
        client = _ScriptedClient(cfg=role_cfg)
        client.profile = role_profile
        built.append(client)
        return client

    runtime = build_model_role_runtime(
        cfg=runtime_cfg,
        main_client=main,
        profiles_dir=FIXTURE_PROFILES,
        client_factory=factory,
    )
    advisor = runtime.router.client_for("advisor")
    assert advisor.client.cfg.model == "review-served-model"
    assert advisor.client.cfg.base_url == "http://127.0.0.1:8181/v1"
    assert advisor.resolution.effective_role == "advisor"
    assert built == [advisor.client]

    unsupported = _ScriptedClient(cfg=runtime_cfg)
    unsupported.profile = replace(
        load_profile("_base", FIXTURE_PROFILES), supports_tool_calls=False
    )
    with pytest.raises(ValueError, match="supports_tool_calls=true"):
        build_model_role_runtime(
            cfg=runtime_cfg,
            main_client=unsupported,
            profiles_dir=FIXTURE_PROFILES,
            client_factory=factory,
        )


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("enabled = 1", "advisor.enabled must be a boolean"),
        ("every_n_turns = 0", "advisor.every_n_turns"),
        ("immune_turns = -1", "advisor.immune_turns"),
        ("max_note_chars = 0", "advisor.max_note_chars"),
        ("endpoint = \"relative\"", "absolute http"),
    ],
)
def test_public_advisor_config_rejects_invalid_values(
    tmp_path: Path, body: str, error: str
):
    overlay = tmp_path / "invalid.toml"
    overlay.write_text(f"[advisor]\n{body}\n")
    with pytest.raises(ValueError, match=error):
        load_config(overlay)


def test_advisor_tool_surface_executes_read_but_quarantines_mutation(
    tmp_path: Path,
):
    source = tmp_path / "fact.txt"
    source.write_text("bounded evidence\n")
    (tmp_path / "prompt.txt").write_text("secret original task\n")
    (tmp_path / "WATCHDOG.md").write_text("Prioritize concrete correctness.\n")
    protected = tmp_path / "protected.txt"
    protected.write_text("original\n")
    client = _ScriptedClient(
        advisor=[
            _turn(
                calls=[
                    ToolCall("r1", "read", {"path": "fact.txt"}),
                    ToolCall("r2", "read", {"path": "prompt.txt"}),
                ]
            ),
            _turn(
                calls=[
                    ToolCall(
                        "a1",
                        "advise",
                        {"severity": "concern", "note": "Check the invariant."},
                    )
                ]
            ),
            _turn(
                calls=[
                    ToolCall(
                        "w1",
                        "write",
                        {"path": "protected.txt", "content": "mutated\n"},
                    )
                ]
            ),
        ]
    )
    session = Session(
        _advisor_config(), client, "system", "task", str(tmp_path)
    )

    _record_primary_turn(session, 0)
    session._capture_advisor_turn(0, "I changed the implementation.", [])
    assert session._maybe_run_advisor(0) is True
    second_messages = client.advisor_calls[1][0]
    assert "bounded evidence" in json.dumps(second_messages)
    assert "secret original task" not in json.dumps(second_messages)
    assert [
        schema["function"]["name"]
        for schema in client.advisor_calls[0][1]
    ] == ["read", "grep", "glob", "advise"]
    first_messages = client.advisor_calls[0][0]
    assert [message["role"] for message in first_messages] == ["system", "user"]
    assert "Prioritize concrete correctness." in first_messages[0]["content"]
    assert "I changed the implementation." in first_messages[1]["content"]
    assert '"primary_turn":0' in first_messages[1]["content"]
    assert "task" not in first_messages[1]["content"]
    session._inject_pending_advisor(1)

    _record_primary_turn(session, 1)
    session._capture_advisor_turn(1, "Another turn.", [])
    assert session._maybe_run_advisor(1) is False
    assert protected.read_text() == "original\n"
    rows = [
        json.loads(line)
        for line in (tmp_path / "advisor.jsonl").read_text().splitlines()
    ]
    quarantined = [row for row in rows if row["event"] == "quarantine"]
    assert quarantined[-1]["reason"] == "unknown_or_mutating_tool"
    assert quarantined[-1]["tool_names"] == ["write"]


def test_over_limit_note_is_quarantined_instead_of_truncated(tmp_path: Path):
    client = _ScriptedClient(
        advisor=[
            _turn(
                calls=[
                    ToolCall(
                        "a1",
                        "advise",
                        {"severity": "nit", "note": "x" * 21},
                    )
                ]
            )
        ]
    )
    session = Session(
        _advisor_config(advisor_max_note_chars=20),
        client,
        "system",
        "task",
        str(tmp_path),
    )
    _record_primary_turn(session, 0)
    session._capture_advisor_turn(0, "review", [])
    assert session._maybe_run_advisor(0) is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "advisor.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["event"] == "quarantine"
    assert rows[-1]["reason"] == "invalid_note"
    assert not any(row["event"] == "advisory" for row in rows)


@pytest.mark.parametrize("profile_name", (None, "_base"))
def test_tool_side_transport_isolated_from_primary_transcript(
    tmp_path: Path, profile_name: str | None
):
    cfg = make_config()
    profile = (
        load_profile(profile_name, FIXTURE_PROFILES)
        if profile_name is not None
        else None
    )
    client = LlamaClient(cfg, profile=profile)
    captured = []
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="advise",
                                arguments=json.dumps(
                                    {"severity": "nit", "note": "Check it."}
                                ),
                            )
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        model_dump_json=lambda: "{}",
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **payload: captured.append(payload) or response
            )
        )
    )
    transcript = tmp_path / "transcript.log"
    client.set_transcript(transcript)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "advise",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    result = client.complete_tool_side_request(
        [{"role": "user", "content": "delta"}], tools, turn=4
    )

    assert result.tool_calls == [
        ToolCall("call_4_0", "advise", {"severity": "nit", "note": "Check it."})
    ]
    assert captured[0]["tools"] == tools
    assert captured[0]["tool_choice"] == "auto"
    assert captured[0]["extra_body"]["cache_prompt"] is False
    assert client._transcript_call_n == 0
    assert transcript.read_text() == ""


def test_note_is_injected_into_next_model_turn_and_artifacts_are_separated(
    tmp_path: Path,
):
    note = "Re-read <critical> before finalizing."
    cfg = _advisor_config(advisor_immune_turns=3)
    client = _ScriptedClient(
        primary=[
            _turn(content="The task is complete.", reason="stop"),
            _turn(content="Now it is verified.", reason="stop"),
        ],
        advisor=[
            _turn(
                calls=[
                    ToolCall(
                        "a1",
                        "advise",
                        {"severity": "blocker", "note": note},
                    )
                ]
            )
        ],
    )
    trace_path = tmp_path / ".trace.jsonl"
    state_path = tmp_path / ".solver" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        '{"state":{},"trace":[],"gates":[],"evidence":[],"inference":[]}'
    )
    with trace_path.open("a") as trace_file:
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            context_manager=FullTranscript(),
            trace_file=trace_file,
            trace_path=trace_path,
            state_path=state_path,
            artifact_dir=tmp_path,
        )
        with patch.object(Session, "_get_server_ctx", return_value=0):
            result = session.run()

    assert result.done is True
    assert len(client.chat_calls) == 2
    second_request = "\n".join(
        str(message.get("content") or "") for message in client.chat_calls[1]
    )
    assert (
        '<injected-fragment source="advisor" severity="blocker">'
        in second_request
    )
    assert "Re-read &lt;critical&gt; before finalizing." in second_request

    trace_text = trace_path.read_text()
    trace_rows = [json.loads(line) for line in trace_text.splitlines()]
    event = next(row for row in trace_rows if row["event"] == "advisor_note")
    assert (event["severity"], event["chars"], event["turn"]) == (
        "blocker",
        len(note),
        0,
    )
    assert note not in trace_text
    state_text = state_path.read_text()
    assert note not in state_text
    assert "advisor" not in state_text

    advisor_text = (tmp_path / "advisor.jsonl").read_text()
    assert note in advisor_text
    assert '"event":"advisory_injected"' in advisor_text


@pytest.mark.parametrize(
    "mode",
    (
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
    ),
)
def test_projection_context_delivers_transient_note_once(
    tmp_path: Path, mode: str
):
    state_path = tmp_path / ".solver" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        '{"state":{},"trace":[],"gates":[],"evidence":[],"inference":[]}'
    )
    context = build_context_manager(
        resolve_context_class(mode),
        make_config(min_turns_before_context=0),
        tmp_path,
        "task",
        1,
        None,
    )
    assert context is not None
    context.add_system("system")
    context.add_user("task")
    fragment = (
        '<injected-fragment source="advisor" severity="nit">\n'
        "Check this.\n</injected-fragment>"
    )
    context.add_injected_fragment(fragment)
    assert fragment in context.get_messages()[-1]["content"]
    context.consume_injected_fragments()
    assert fragment not in context.get_messages()[-1]["content"]


def test_advisor_dedupe_and_cooldown_use_completed_primary_turns(
    tmp_path: Path,
):
    first = {"severity": "concern", "note": "Inspect the return value."}
    second = {"severity": "nit", "note": "Add the missing assertion."}
    client = _ScriptedClient(
        advisor=[
            _turn(calls=[ToolCall("a1", "advise", first)]),
            _turn(calls=[ToolCall("a2", "advise", first)]),
            _turn(calls=[ToolCall("a3", "advise", second)]),
        ]
    )
    session = Session(
        _advisor_config(advisor_immune_turns=2),
        client,
        "system",
        "task",
        str(tmp_path),
    )

    outcomes = []
    for ordinal in range(1, 6):
        turn = ordinal - 1
        _record_primary_turn(session, turn)
        session._capture_advisor_turn(turn, f"turn {turn}", [])
        outcomes.append(session._maybe_run_advisor(turn))
        if outcomes[-1]:
            session._inject_pending_advisor(turn + 1)

    assert outcomes == [True, False, False, False, True]
    assert len(client.advisor_calls) == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "advisor.jsonl").read_text().splitlines()
    ]
    skips = [row["reason"] for row in rows if row["event"] == "review_skipped"]
    assert skips == ["cooldown", "cooldown"]
    assert any(row["event"] == "advisory_deduplicated" for row in rows)
    assert len([row for row in rows if row["event"] == "advisory"]) == 2


def test_pending_advisory_rehydrates_for_the_next_session(tmp_path: Path):
    note = "Carry this note across the context rollover."
    first_client = _ScriptedClient(
        advisor=[
            _turn(
                calls=[
                    ToolCall(
                        "a1",
                        "advise",
                        {"severity": "concern", "note": note},
                    )
                ]
            )
        ]
    )
    cfg = _advisor_config()
    first = Session(
        cfg,
        first_client,
        "system",
        "task",
        str(tmp_path),
        session_number=1,
    )
    _record_primary_turn(first, 4)
    first._capture_advisor_turn(4, "roll over", [])
    assert first._maybe_run_advisor(4) is True

    resumed = Session(
        cfg,
        _ScriptedClient(),
        "system",
        "resume",
        str(tmp_path),
        session_number=2,
    )
    assert resumed._inject_pending_advisor(5) is True
    rendered = json.dumps(resumed.context.get_messages())
    assert note in rendered
    rows = [
        json.loads(line)
        for line in (tmp_path / "advisor.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["event"] == "advisory_injected"
    assert rows[-1]["session_number"] == 2


def _replay_transcript(path: Path) -> Path:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    path.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({"messages": []})
        + "\n=== turn 001 output ===\n"
        + json.dumps(response)
        + "\n"
    )
    return path


def test_default_off_replay_is_run_deterministic_and_has_no_advisor_artifact(
    tmp_path: Path,
):
    cfg = make_config(max_turns=1)
    assert cfg.advisor_enabled is False
    source = _replay_transcript(tmp_path / "source.log")
    observed = []
    for run_number in (1, 2):
        run_dir = tmp_path / f"run-{run_number}"
        run_dir.mkdir()
        trace = io.StringIO()
        session = Session(
            cfg,
            ReplayClient(source, strict_fidelity=False),
            "system",
            "task",
            str(run_dir),
            context_manager=FullTranscript(),
            trace_file=trace,
            artifact_dir=run_dir,
        )
        with patch.object(Session, "_get_server_ctx", return_value=0):
            result = session.run()
        observed.append((result, trace.getvalue(), session.context.get_messages()))
        assert session._advisor is None
        assert not (run_dir / "advisor.jsonl").exists()

    assert observed[0] == observed[1]
    assert "advisor_note" not in observed[0][1]
