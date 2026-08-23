"""Runtime acceptance coverage for named auxiliary model roles."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver._shared.telemetry_paths import trace_path
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.model_role_runtime import (
    build_model_role_runtime,
)
from scripts.llm_solver.harness.adaptive_control.llm_detector_runtime import (
    maybe_run_llm_hurdle_detector,
)
from scripts.llm_solver.harness.loop import solve_task
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.types import (
    SideRequestResult,
    ToolCall,
    TurnResult,
    Usage,
)


FIXTURE_PROFILES = Path(__file__).parent / "fixtures" / "model_role_profiles"


def _valid_handoff() -> str:
    return """\
## Goal
Fix the requested behavior.
## Done
Inspected the implementation.
## In progress
Continue the integration.
## Blocked
Nothing is blocked.
## Key decisions
Keep deterministic mechanical fallback behavior.
## Critical paths/errors
No modified paths were observed.
## Next step
Run the focused verification."""


class _FakeClient:
    def __init__(self, cfg, profile, *, turns=(), side_text=""):
        self.cfg = cfg
        self.profile = profile
        self.turns = list(turns)
        self.side_text = side_text
        self.chat_calls: list[tuple[list[dict], list[dict], int]] = []
        self.side_calls: list[dict] = []
        self._session_id = ""

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def chat(self, messages, tools, turn=0):
        self.chat_calls.append((messages, tools, turn))
        return self.turns.pop(0)

    def complete_side_request(self, payload):
        self.side_calls.append(payload)
        return SideRequestResult(
            self.side_text,
            Usage(prompt_tokens=120, completion_tokens=30, cached_tokens=0),
        )

    def build_assistant_message(self, content, tool_calls):
        return {"role": "assistant", "content": content}


def _runtime_config(**overrides):
    values = {
        "model": "main-served-model",
        "profile_name": "_base",
        "base_url": "http://127.0.0.1:8080/v1",
        "context_size": 16_000,
        "model_roles": {
            "weak": {
                "profile": "weak",
                "endpoint": "http://127.0.0.1:8181/v1",
                "model": "weak-served-model",
            },
            "editor": "",
        },
    }
    values.update(overrides)
    return make_config(**values)


def test_public_role_config_accepts_strings_and_inline_targets(tmp_path: Path):
    defaults = load_config()
    assert defaults.model_roles == {"weak": "", "editor": ""}

    overlay = tmp_path / "roles.toml"
    overlay.write_text(
        """
[models.roles.weak]
profile = "weak"
endpoint = "http://127.0.0.1:8181/v1"
model = "weak-served-model"
context_size = 4096
"""
    )
    cfg = load_config(overlay)
    assert cfg.model_roles["weak"] == {
        "profile": "weak",
        "endpoint": "http://127.0.0.1:8181/v1",
        "model": "weak-served-model",
        "context_size": 4096,
    }


def test_runtime_eagerly_validates_roles_and_routes_unset_role_to_main():
    cfg = _runtime_config()
    main = _FakeClient(cfg, load_profile("_base", FIXTURE_PROFILES))
    built: list[_FakeClient] = []

    def factory(role_cfg, role_profile):
        client = _FakeClient(role_cfg, role_profile)
        built.append(client)
        return client

    runtime = build_model_role_runtime(
        cfg=cfg,
        main_client=main,
        profiles_dir=FIXTURE_PROFILES,
        client_factory=factory,
    )

    weak = runtime.router.client_for("weak")
    editor = runtime.router.client_for("editor")
    assert weak.client.cfg.base_url == "http://127.0.0.1:8181/v1"
    assert weak.client.cfg.model == "weak-served-model"
    assert weak.client.profile.name == "weak"
    assert editor.client is main
    assert editor.trace_fields() == {
        "role": "main",
        "requested_role": "editor",
        "role_fallback": "main",
    }
    assert built == [weak.client]

    bad = _runtime_config(model_roles={"weak": "broken"})
    with pytest.raises(ValueError, match="failed validation"):
        build_model_role_runtime(
            cfg=bad,
            main_client=main,
            profiles_dir=FIXTURE_PROFILES,
            client_factory=factory,
        )


def test_handoff_uses_weak_client_and_metrics_split_every_response_once(
    tmp_path: Path,
):
    (tmp_path / "prompt.txt").write_text("Fix it.")
    turns = [
        TurnResult(
            content="Inspect.",
            tool_calls=[
                ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
            ],
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=100, completion_tokens=10, cached_tokens=40),
        ),
        TurnResult(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(prompt_tokens=150, completion_tokens=15, cached_tokens=90),
        ),
    ]
    cfg = _runtime_config(
        max_turns=1,
        max_sessions=2,
        handoff_summary_enabled=True,
        handoff_max_tokens=500,
    )
    main = _FakeClient(cfg, load_profile("_base", FIXTURE_PROFILES), turns=turns)
    role_clients: list[_FakeClient] = []

    def factory(role_cfg, role_profile):
        client = _FakeClient(
            role_cfg,
            role_profile,
            side_text=_valid_handoff(),
        )
        role_clients.append(client)
        return client

    build_model_role_runtime(
        cfg=cfg,
        main_client=main,
        profiles_dir=FIXTURE_PROFILES,
        client_factory=factory,
    )

    with (
        patch("scripts.llm_solver.harness.loop._auto_commit"),
        patch("scripts.llm_solver.harness.loop.dispatch", return_value="README"),
        patch(
            "scripts.llm_solver.harness.loop.Session._get_server_ctx",
            return_value=16_000,
        ),
    ):
        assert solve_task(tmp_path, cfg, main) is True

    assert len(role_clients) == 1
    weak = role_clients[0]
    assert len(weak.side_calls) == 1
    assert main.side_calls == []
    assert "tools" not in weak.side_calls[0]

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    handoff = next(event for event in events if event["event"] == "handoff")
    assert handoff["role"] == "weak"

    metrics = json.loads((tmp_path / "metrics.json").read_text())["metrics"]
    assert metrics["tokens_by_role"] == {
        "main": {
            "requests": 2,
            "prompt_tokens": 250,
            "completion_tokens": 25,
            "cached_tokens": 130,
            "total_tokens": 275,
        },
        "weak": {
            "requests": 1,
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "cached_tokens": 0,
            "total_tokens": 150,
        },
    }


def test_model_backed_classifier_uses_weak_side_request(tmp_path: Path):
    atlas = tmp_path / "atlas.tsv"
    atlas.write_text(
        "dictionary_version\tfamily\tcell_count\tdescription\tcovered_by_prior\t"
        "uncovered\tpartial_or_late\tcells\n"
        "v1\tloop_churn\t1\trepetition\t1\t0\t0\tc1\n"
    )
    cfg = _runtime_config(
        llm_hurdle_detector_enabled=True,
        llm_hurdle_detector_cadence_turns=1,
        llm_hurdle_detector_atlas_dictionary_path=str(atlas),
        llm_hurdle_detector_input_contract_path="contract.tsv",
        llm_hurdle_detector_log_path=str(tmp_path / "detector.jsonl"),
        llm_hurdle_detector_max_trace_events=4,
        llm_hurdle_detector_max_field_chars=200,
        llm_hurdle_detector_max_state_snapshots=4,
        llm_hurdle_detector_prompt_version="test-v1",
    )
    main = _FakeClient(cfg, load_profile("_base", FIXTURE_PROFILES))
    role_clients: list[_FakeClient] = []

    def factory(role_cfg, role_profile):
        # Invalid verdict JSON is deliberate: routing/accounting must survive
        # classifier parse failure without interrupting the solver.
        client = _FakeClient(role_cfg, role_profile, side_text="{}")
        role_clients.append(client)
        return client

    runtime = build_model_role_runtime(
        cfg=cfg,
        main_client=main,
        profiles_dir=FIXTURE_PROFILES,
        client_factory=factory,
    )
    session = type("Session", (), {})()
    session.cfg = cfg
    session.client = main
    session._model_role_router = runtime.router
    session._role_token_ledger = runtime.token_ledger
    session._trace_path = tmp_path / ".trace.jsonl"
    session._trace_events = []

    row = maybe_run_llm_hurdle_detector(session, turn=0)

    assert row is not None
    assert row["role"] == "weak"
    assert len(role_clients) == 1
    assert len(role_clients[0].side_calls) == 1
    assert "tools" not in role_clients[0].side_calls[0]
    assert main.chat_calls == []
    assert runtime.token_ledger.snapshot()["weak"]["requests"] == 1
