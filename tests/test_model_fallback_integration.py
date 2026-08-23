"""Runtime acceptance coverage for model fallback chains."""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest

from _config_helpers import make_config
from scripts.llm_solver._shared.telemetry_paths import trace_path
from scripts.llm_solver.config import dump_config, load_config
from scripts.llm_solver.harness._loop.chat_io import _fallback_reason
from scripts.llm_solver.harness._loop.model_role_runtime import (
    begin_model_session,
    bind_session_model_roles,
    build_model_role_runtime,
)
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.loop import Session, solve_task
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


FIXTURE_PROFILES = Path(__file__).parent / "fixtures" / "model_role_profiles"
WEAK_ENDPOINT = "http://127.0.0.1:8182/v1"
EDITOR_ENDPOINT = "http://127.0.0.1:8282/v1"


def _connection_error():
    return openai.APIConnectionError(request=MagicMock())


def _turn(*, tool: bool = False, prompt_tokens: int = 100) -> TurnResult:
    calls = (
        [ToolCall(id="call-1", name="read", arguments={"path": "README.md"})]
        if tool
        else []
    )
    return TurnResult(
        content="Inspect." if tool else "Done.",
        tool_calls=calls,
        finish_reason="tool_calls" if tool else "stop",
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=10),
    )


class _FakeClient:
    def __init__(self, cfg, profile, responses=(), *, live_context=None):
        self.cfg = cfg
        self.profile = profile
        self.responses = list(responses)
        self.live_context = live_context
        self.chat_calls = 0
        self._session_id = "fallback-test"

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def query_server_context(self):
        return self.live_context

    def chat(self, messages, tools, turn=0):
        self.chat_calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def build_assistant_message(self, content, tool_calls):
        return {"role": "assistant", "content": content}


def _cfg(**overrides):
    values = {
        "model": "main-served-model",
        "profile_name": "_base",
        "base_url": "http://127.0.0.1:8080/v1",
        "context_size": 8192,
        "max_transient_retries": 1,
        "retry_backoff": (0,),
        "model_roles": {"weak": "", "editor": ""},
        "model_fallback_chain": {
            "main": [f"weak@{WEAK_ENDPOINT}"],
            "weak": [],
            "editor": [],
        },
        "model_fallback_revert": "never",
    }
    values.update(overrides)
    return make_config(**values)


def _runtime(cfg, main, responses_by_profile):
    clients = []

    def factory(role_cfg, role_profile):
        client = _FakeClient(
            role_cfg,
            role_profile,
            responses_by_profile.get(role_profile.name, ()),
            live_context=role_profile.context_size,
        )
        clients.append(client)
        return client

    runtime = build_model_role_runtime(
        cfg=cfg,
        main_client=main,
        profiles_dir=FIXTURE_PROFILES,
        client_factory=factory,
    )
    return runtime, clients


def _session(tmp_path, cfg, main, runtime, task="Fix it."):
    context = FullTranscript(token_estimator=lambda messages: 777)
    session = Session(
        cfg,
        main,
        "system",
        task,
        str(tmp_path),
        context_manager=context,
        trace_file=StringIO(),
        session_number=1,
    )
    bind_session_model_roles(session, main, runtime.token_ledger)
    return session


def test_public_fallback_config_defaults_validation_and_secret_redaction(
    tmp_path: Path,
):
    cfg = load_config()
    assert cfg.model_fallback_revert == "never"
    assert cfg.model_fallback_chain == {"main": [], "weak": [], "editor": []}

    invalid = tmp_path / "invalid-fallback.toml"
    invalid.write_text("[models]\nfallback_revert='cooldown'\n")
    with pytest.raises(ValueError, match="fallback_revert"):
        load_config(invalid)

    inline = tmp_path / "inline-fallback.toml"
    inline.write_text(
        "[models]\n"
        "fallback_revert='next_session'\n"
        "[[models.fallback_chain.main]]\n"
        "profile='weak'\n"
        f"endpoint='{WEAK_ENDPOINT}'\n"
        "model='served-weak'\n"
        "context_size=4096\n"
    )
    loaded = load_config(inline)
    assert loaded.model_fallback_revert == "next_session"
    assert loaded.model_fallback_chain["main"] == [{
        "profile": "weak",
        "endpoint": WEAK_ENDPOINT,
        "model": "served-weak",
        "context_size": 4096,
    }]

    secret_cfg = _cfg(model_fallback_chain={
        "main": [{
            "profile": "weak",
            "endpoint": WEAK_ENDPOINT,
            "api_key": "do-not-persist",
        }]
    })
    dumped = dump_config(secret_cfg)
    assert dumped["api_key"] == "<redacted>"
    assert dumped["model_fallback_chain"]["main"][0]["api_key"] == "<redacted>"
    assert "do-not-persist" not in json.dumps(dumped)


@pytest.mark.parametrize(
    ("detail", "default", "expected"),
    [
        ("CUDA out of memory", "transient_exhausted", "server_oom"),
        ("maximum context length exceeded", None, "context_overflow"),
        ("connection reset", "transient_exhausted", "transient_exhausted"),
        ("invalid tool schema", None, None),
    ],
)
def test_runtime_fallback_reasons_are_stable_and_failure_specific(
    detail,
    default,
    expected,
):
    assert _fallback_reason(RuntimeError(detail), default) == expected


def test_retry_exhaustion_atomically_rebinds_profile_context_and_estimator(
    tmp_path: Path,
):
    cfg = _cfg()
    main = _FakeClient(
        cfg,
        load_profile("_base", FIXTURE_PROFILES),
        [_connection_error(), _connection_error()],
        live_context=8192,
    )
    runtime, clients = _runtime(cfg, main, {"weak": [_turn()]})
    session = _session(tmp_path, cfg, main, runtime)
    assert session.context.estimate_tokens() == 777
    assert session.context._tok_cache == 777

    with patch("scripts.llm_solver.harness._loop.chat_io.time.sleep"):
        result = session._chat_with_retry(3)

    assert result == _turn()
    assert main.chat_calls == cfg.max_transient_retries + 1
    assert len(clients) == 1
    fallback = clients[0]
    assert fallback.chat_calls == 1
    assert session.client is fallback
    assert session.client.profile.name == "weak"
    assert session.cfg.context_size == 4096
    assert session.cfg.max_tokens == 2048
    assert session.context._tok_cache is None
    assert session._server_ctx_cache == 4096
    event = next(
        row for row in session._trace_events if row["event"] == "model_fallback"
    )
    assert event["turn_number"] == 3
    assert event["reason"] == "transient_exhausted"
    assert event["from_profile"] == "_base"
    assert event["to_profile"] == "weak"
    assert event["to_context_size"] == 4096


def test_too_small_fallback_is_traced_and_next_target_gets_fresh_budget(
    tmp_path: Path,
):
    cfg = _cfg(
        max_transient_retries=0,
        model_fallback_chain={
            "main": [
                f"weak@{WEAK_ENDPOINT}",
                f"editor@{EDITOR_ENDPOINT}",
            ]
        },
    )
    main = _FakeClient(
        cfg,
        load_profile("_base", FIXTURE_PROFILES),
        [_connection_error()],
        live_context=8192,
    )
    runtime, clients = _runtime(
        cfg,
        main,
        {"weak": [_turn()], "editor": [_turn(prompt_tokens=200)]},
    )
    session = _session(tmp_path, cfg, main, runtime, task="x" * 16_000)

    result = session._chat_with_retry(4)

    assert result.usage.prompt_tokens == 200
    assert [client.profile.name for client in clients] == ["weak", "editor"]
    assert clients[0].chat_calls == 0
    assert clients[1].chat_calls == 1
    transitions = [
        row for row in session._trace_events if row["event"] == "model_fallback"
    ]
    assert [row["to_profile"] for row in transitions] == ["weak", "editor"]
    assert [row["reason"] for row in transitions] == [
        "transient_exhausted",
        "context_window_exceeded",
    ]
    assert session.cfg.context_size == 16_384


def test_no_fitting_fallback_aborts_without_rebinding_the_session(tmp_path: Path):
    cfg = _cfg(max_transient_retries=0)
    main = _FakeClient(
        cfg,
        load_profile("_base", FIXTURE_PROFILES),
        [_connection_error()],
        live_context=8192,
    )
    runtime, clients = _runtime(cfg, main, {"weak": [_turn()]})
    session = _session(tmp_path, cfg, main, runtime, task="x" * 16_000)

    assert session._chat_with_retry(5) is None

    assert session.client is main
    assert clients[0].chat_calls == 0
    transition = next(
        row for row in session._trace_events if row["event"] == "model_fallback"
    )
    assert transition["to_profile"] == "weak"
    assert transition["reason"] == "transient_exhausted"


def test_next_session_policy_reverts_to_primary_client():
    cfg = _cfg(model_fallback_revert="next_session")
    main = _FakeClient(cfg, load_profile("_base", FIXTURE_PROFILES))
    runtime, clients = _runtime(cfg, main, {"weak": []})
    switched = runtime.router.switch_after_retry_exhaustion(
        "main", reason="transient_exhausted"
    )
    assert switched is not None
    assert clients and runtime.fallback_controller.current("main").profile.name == "weak"

    binding = begin_model_session(main, cfg)

    assert binding.client is main
    assert binding.resolution.profile.name == "_base"


def test_unset_side_role_follows_the_active_main_fallback():
    cfg = _cfg()
    main = _FakeClient(cfg, load_profile("_base", FIXTURE_PROFILES))
    runtime, clients = _runtime(cfg, main, {"weak": []})
    switched = runtime.router.switch_after_retry_exhaustion(
        "main", reason="transient_exhausted"
    )
    assert switched is not None

    side = runtime.router.client_for("weak")

    assert side.client is clients[0]
    assert side.resolution.requested_role == "weak"
    assert side.resolution.effective_role == "main"
    assert side.resolution.uses_main_fallback is True
    assert side.resolution.target == switched.routed_client.resolution.target


def test_trace_provenance_and_metrics_expose_effective_model_per_session(
    tmp_path: Path,
):
    (tmp_path / "prompt.txt").write_text("Fix it.")
    cfg = _cfg(
        max_transient_retries=0,
        max_turns=1,
        max_sessions=2,
    )
    main = _FakeClient(
        cfg,
        load_profile("_base", FIXTURE_PROFILES),
        [_connection_error()],
        live_context=8192,
    )
    runtime, clients = _runtime(
        cfg,
        main,
        {"weak": [_turn(tool=True), _turn(prompt_tokens=150)]},
    )

    with (
        patch("scripts.llm_solver.harness.loop._auto_commit"),
        patch("scripts.llm_solver.harness.loop.dispatch", return_value="README"),
        patch(
            "scripts.llm_solver.harness.loop.Session._get_server_ctx",
            side_effect=lambda: 8192,
        ),
    ):
        assert solve_task(tmp_path, cfg, main) is True

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    starts = [row for row in events if row["event"] == "session_start"]
    assert starts[0]["model_target"] == f"_base@{cfg.base_url}"
    assert starts[1]["model_target"] == f"weak@{WEAK_ENDPOINT}"
    assert starts[1]["model"] == "weak"
    transition = next(row for row in events if row["event"] == "model_fallback")
    assert transition["reason"] == "transient_exhausted"

    payload = json.loads((tmp_path / "metrics.json").read_text())
    metrics = payload["metrics"]
    assert metrics["model_fallback_used"] is True
    assert metrics["model_fallback_count"] == 1
    assert metrics["model_fallback_roles"] == ["main"]
    assert metrics["model_fallback_active_targets"] == {
        "main": f"weak@{WEAK_ENDPOINT}"
    }
    assert metrics["tokens_by_role"]["main"]["requests"] == 2
    provenance = payload["provenance"]
    assert provenance["initial_model_target"] == f"_base@{cfg.base_url}"
    assert provenance["model_fallback_revert"] == "never"
    assert provenance["model_fallback_chains"] == {
        "main": [f"weak@{WEAK_ENDPOINT}"]
    }
    state_text = (tmp_path / ".solver" / "state.json").read_text()
    assert "model_fallback" not in state_text
    assert clients[0].chat_calls == 2
