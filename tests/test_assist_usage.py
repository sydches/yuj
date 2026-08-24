import hashlib
import json
import sqlite3
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist._anthropic import _anthropic_to_openai_response
from scripts.llm_assist._codex import _responses_to_openai
from scripts.llm_assist.store import SessionStore
from scripts.llm_assist.usage import (
    CostEvidence,
    QuotaEvidence,
    UsageEvidenceError,
    aggregate_session_usage,
    render_session_usage,
)
from scripts.llm_solver.harness._loop.chat_io import _aggregate_usage
from scripts.llm_solver.harness._loop.model_role_runtime import (
    ConsumerRoleClient,
    record_role_usage,
)
from scripts.llm_solver.harness._loop.model_roles import RoleTokenLedger
from scripts.llm_solver.harness._loop.usage_evidence import (
    SessionUsageAccumulator,
)
from scripts.llm_solver.harness.loop import Session, solve_task
from scripts.llm_solver.harness.state_writer import project
from scripts.llm_solver.server._streaming import (
    _StreamedChoice,
    _StreamedMessage,
    _StreamedResponse,
)
from scripts.llm_solver.server.request_controls import usage_from_response
from scripts.llm_solver.server.types import TurnResult, Usage
from tests._config_helpers import make_config


def _usage_event(
    session_number: int,
    *,
    input_tokens=100,
    output_tokens=20,
    cached_tokens=40,
    cost=None,
    quota=None,
) -> dict:
    return {
        "event": "session_usage",
        "trace_schema_version": 2,
        "session_number": session_number,
        "scope": "all_model_responses",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cost": cost,
        "quota": quota,
    }


def _write_trace(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def _tree_bytes(root: Path) -> dict[str, tuple[str, bytes]]:
    result = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        body = path.read_bytes()
        result[str(path.relative_to(root))] = (hashlib.sha256(body).hexdigest(), body)
    return result


def test_fully_known_resumed_usage_uses_exact_typed_arithmetic(tmp_path: Path):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            {"event": "session_start", "session_number": 1},
            _usage_event(
                1,
                input_tokens=100,
                output_tokens=20,
                cached_tokens=40,
                cost={"amount": "1.25", "currency": "USD"},
                quota={
                    "remaining": "900",
                    "limit": "1000",
                    "unit": "requests",
                    "scope": "account/day",
                },
            ),
            {"event": "session_end", "session_number": 1},
            {"event": "session_start", "session_number": 2},
            _usage_event(
                2,
                input_tokens=200,
                output_tokens=30,
                cached_tokens=80,
                cost={"amount": "0.75", "currency": "USD"},
                quota={
                    "remaining": "800",
                    "limit": "1000",
                    "unit": "requests",
                    "scope": "account/day",
                },
            ),
            {"event": "session_end", "session_number": 2},
        ],
    )

    result = aggregate_session_usage([trace])

    assert result.segments == 2
    assert result.input_tokens == 300
    assert result.output_tokens == 50
    assert result.cached_tokens == 120
    assert result.cache_ratio == Fraction(2, 5)
    assert result.cost == CostEvidence(amount=Decimal("2.00"), currency="USD")
    assert result.quota == QuotaEvidence(
        remaining=Decimal("800"),
        limit=Decimal("1000"),
        unit="requests",
        scope="account/day",
    )
    assert render_session_usage(result) == [
        "segments: 2",
        "input_tokens: 300",
        "output_tokens: 50",
        "cached_tokens: 120",
        "cache_ratio: 40.00%",
        "cost: 2.00 USD",
        "quota: 800/1000 requests remaining (account/day)",
    ]


def test_fully_unknown_usage_stays_unknown(tmp_path: Path):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            {"event": "session_start", "session_number": 1},
            _usage_event(
                1,
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
            ),
        ],
    )

    result = aggregate_session_usage([trace])

    assert result.segments == 1
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.cached_tokens is None
    assert result.cache_ratio is None
    assert result.cost is None
    assert result.quota is None
    assert render_session_usage(result)[1:] == [
        "input_tokens: unknown",
        "output_tokens: unknown",
        "cached_tokens: unknown",
        "cache_ratio: unknown",
        "cost: unknown",
        "quota: unknown",
    ]


def test_mixed_fields_become_unknown_independently(tmp_path: Path):
    quota_1 = {
        "remaining": "90",
        "limit": "100",
        "unit": "requests",
        "scope": "account/day",
    }
    quota_2 = {**quota_1, "remaining": "80"}
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            _usage_event(
                1,
                input_tokens=100,
                output_tokens=20,
                cached_tokens=None,
                cost={"amount": "0.10", "currency": "USD"},
                quota=quota_1,
            ),
            _usage_event(
                2,
                input_tokens=50,
                output_tokens=None,
                cached_tokens=20,
                cost=None,
                quota=quota_2,
            ),
        ],
    )

    result = aggregate_session_usage([trace])

    assert result.input_tokens == 150
    assert result.output_tokens is None
    assert result.cached_tokens is None
    assert result.cache_ratio is None
    assert result.cost is None
    assert result.quota == QuotaEvidence(
        remaining=Decimal("80"),
        limit=Decimal("100"),
        unit="requests",
        scope="account/day",
    )


def test_duplicate_paths_and_identical_events_count_each_segment_once(tmp_path: Path):
    first = _usage_event(1, input_tokens=10, output_tokens=2, cached_tokens=5)
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [first, dict(first), _usage_event(2, input_tokens=20, output_tokens=3, cached_tokens=10)],
    )

    result = aggregate_session_usage([trace, trace, trace.resolve()])

    assert result.segments == 2
    assert (result.input_tokens, result.output_tokens, result.cached_tokens) == (
        30,
        5,
        15,
    )


def test_conflicting_duplicate_segment_facts_fail_loudly(tmp_path: Path):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            _usage_event(1, input_tokens=10, cached_tokens=5),
            _usage_event(1, input_tokens=11, cached_tokens=5),
        ],
    )

    with pytest.raises(UsageEvidenceError, match="conflicting session_usage.*segment 1"):
        aggregate_session_usage([trace])


def test_legacy_segment_does_not_mix_primary_turns_with_all_response_facts(
    tmp_path: Path,
):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            {"event": "session_start", "session_number": 1},
            {
                "event": "turn",
                "session_number": 1,
                "turn_number": 0,
                "prompt_tokens": 100,
                "cached_tokens": 80,
            },
            {"event": "session_end", "session_number": 1},
            {"event": "session_start", "session_number": 2},
            _usage_event(2, input_tokens=50, output_tokens=5, cached_tokens=25),
        ],
    )

    result = aggregate_session_usage([trace])

    assert result.segments == 2
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.cached_tokens is None
    assert result.cache_ratio is None


def test_missing_earlier_segment_implied_by_resume_number_stays_unknown(
    tmp_path: Path,
):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [_usage_event(2, input_tokens=50, output_tokens=5, cached_tokens=25)],
    )

    result = aggregate_session_usage([trace])

    assert result.segments == 2
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.cached_tokens is None
    assert result.cache_ratio is None


def test_cache_ratio_requires_known_positive_input_count(tmp_path: Path):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [_usage_event(1, input_tokens=0, output_tokens=0, cached_tokens=0)],
    )

    result = aggregate_session_usage([trace])

    assert result.cached_tokens == 0
    assert result.cache_ratio is None
    assert "cache_ratio: unknown" in render_session_usage(result)


def test_cache_ratio_rendering_uses_exact_rounding(tmp_path: Path):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [_usage_event(1, input_tokens=6, output_tokens=0, cached_tokens=1)],
    )

    result = aggregate_session_usage([trace])

    assert result.cache_ratio == Fraction(1, 6)
    assert "cache_ratio: 16.67%" in render_session_usage(result)


def test_incompatible_typed_cost_and_quota_stay_unknown(tmp_path: Path):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            _usage_event(
                1,
                cost={"amount": "1", "currency": "USD"},
                quota={
                    "remaining": "9",
                    "limit": "10",
                    "unit": "requests",
                    "scope": "account/day",
                },
            ),
            _usage_event(
                2,
                cost={"amount": "1", "currency": "EUR"},
                quota={
                    "remaining": "90",
                    "limit": "100",
                    "unit": "tokens",
                    "scope": "project/month",
                },
            ),
        ],
    )

    result = aggregate_session_usage([trace])

    assert result.cost is None
    assert result.quota is None


def test_partially_known_quota_stays_unknown(tmp_path: Path):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            _usage_event(
                1,
                quota={
                    "remaining": "9",
                    "limit": "10",
                    "unit": "requests",
                    "scope": "account/day",
                },
            ),
            _usage_event(2, quota=None),
        ],
    )

    assert aggregate_session_usage([trace]).quota is None


def test_cost_aggregation_is_exact_beyond_decimal_context_precision(
    tmp_path: Path,
):
    trace = _write_trace(
        tmp_path / ".trace.jsonl",
        [
            _usage_event(
                1,
                cost={
                    "amount": "1234567890123456789012345678.01",
                    "currency": "USD",
                },
            ),
            _usage_event(
                2,
                cost={"amount": "0.02", "currency": "USD"},
            ),
        ],
    )

    result = aggregate_session_usage([trace])

    assert result.cost == CostEvidence(
        amount=Decimal("1234567890123456789012345678.03"),
        currency="USD",
    )


@pytest.mark.parametrize(
    "event,match",
    [
        (_usage_event(0), "positive integer"),
        (_usage_event(1, input_tokens=-1), "input_tokens"),
        (_usage_event(1, input_tokens=10, cached_tokens=11), "cached_tokens"),
        (
            _usage_event(1, cost={"amount": "NaN", "currency": "USD"}),
            "finite",
        ),
    ],
)
def test_corrupt_usage_shapes_fail_loudly(tmp_path: Path, event: dict, match: str):
    trace = _write_trace(tmp_path / ".trace.jsonl", [event])

    with pytest.raises(UsageEvidenceError, match=match):
        aggregate_session_usage([trace])


def test_session_usage_accumulator_preserves_unknown_cache_per_response():
    accumulator = SessionUsageAccumulator()
    accumulator.record(Usage(prompt_tokens=100, completion_tokens=10, cached_tokens=40))
    accumulator.record(Usage(prompt_tokens=50, completion_tokens=5, cached_tokens=None))

    assert accumulator.trace_fields() == {
        "scope": "all_model_responses",
        "input_tokens": 150,
        "output_tokens": 15,
        "cached_tokens": None,
        "cost": None,
        "quota": None,
    }


def test_missing_provider_token_counts_stay_unknown_in_segment_fact():
    accumulator = SessionUsageAccumulator()

    accumulator.record(usage_from_response({"usage": {}}))

    assert accumulator.trace_fields() == {
        "scope": "all_model_responses",
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
        "cost": None,
        "quota": None,
    }


def test_partially_known_provider_token_counts_stay_independent():
    accumulator = SessionUsageAccumulator()

    accumulator.record(
        usage_from_response({"usage": {"prompt_tokens": 12}})
    )

    assert accumulator.trace_fields()["input_tokens"] == 12
    assert accumulator.trace_fields()["output_tokens"] is None


def test_claude_adapter_preserves_total_input_and_cache_read_evidence():
    response = _anthropic_to_openai_response(
        {
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 7,
                "output_tokens": 2,
            },
        }
    )
    accumulator = SessionUsageAccumulator()

    accumulator.record(usage_from_response(response))

    assert accumulator.trace_fields() == {
        "scope": "all_model_responses",
        "input_tokens": 20,
        "output_tokens": 2,
        "cached_tokens": 7,
        "cost": None,
        "quota": None,
    }


def test_codex_adapter_preserves_input_cache_details():
    response = _responses_to_openai(
        {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens": 10,
            },
        }
    )
    accumulator = SessionUsageAccumulator()

    accumulator.record(usage_from_response(response))

    assert accumulator.trace_fields() == {
        "scope": "all_model_responses",
        "input_tokens": 100,
        "output_tokens": 10,
        "cached_tokens": 40,
        "cost": None,
        "quota": None,
    }


@pytest.mark.parametrize(
    "response",
    [
        _anthropic_to_openai_response({"content": []}),
        _responses_to_openai({"status": "completed", "output": []}),
    ],
)
def test_subscription_adapters_do_not_turn_missing_usage_into_zero(response):
    accumulator = SessionUsageAccumulator()

    accumulator.record(usage_from_response(response))

    assert accumulator.trace_fields()["input_tokens"] is None
    assert accumulator.trace_fields()["output_tokens"] is None
    assert accumulator.trace_fields()["cached_tokens"] is None


def test_streamed_response_without_usage_stays_unknown_for_report():
    response = _StreamedResponse(
        choices=[
            _StreamedChoice(
                message=_StreamedMessage(content="done"),
                finish_reason="stop",
            )
        ]
    )
    accumulator = SessionUsageAccumulator()

    assert json.loads(response.model_dump_json())["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    accumulator.record(usage_from_response(response))

    assert accumulator.trace_fields()["input_tokens"] is None
    assert accumulator.trace_fields()["output_tokens"] is None


def test_length_continuation_keeps_token_knowledge_independent():
    usage = _aggregate_usage(
        [
            Usage(prompt_tokens=10, completion_tokens=2),
            Usage(
                prompt_tokens=0,
                completion_tokens=3,
                prompt_tokens_known=False,
            ),
        ]
    )
    accumulator = SessionUsageAccumulator()

    accumulator.record(usage)

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert accumulator.trace_fields()["input_tokens"] is None
    assert accumulator.trace_fields()["output_tokens"] == 5


def test_side_role_usage_enters_the_assistant_segment_once():
    owner = MagicMock()
    owner.__dict__["_role_token_ledger"] = RoleTokenLedger()
    owner.__dict__["_session_usage_accumulator"] = SessionUsageAccumulator()
    routed = ConsumerRoleClient(MagicMock(), requested_role="weak")

    record_role_usage(
        owner,
        routed,
        Usage(prompt_tokens=60, completion_tokens=6, cached_tokens=30),
    )

    assert owner._session_usage_accumulator.trace_fields() == {
        "scope": "all_model_responses",
        "input_tokens": 60,
        "output_tokens": 6,
        "cached_tokens": 30,
        "cost": None,
        "quota": None,
    }


def test_assistant_run_emits_one_usage_fact_but_measurement_run_does_not(
    tmp_path: Path,
):
    def run(mode: str, root: Path) -> list[dict]:
        work = root / "work"
        artifacts = root / "artifacts"
        work.mkdir(parents=True)
        artifacts.mkdir(parents=True)
        (artifacts / "prompt.txt").write_text("Finish the task.\n")
        cfg = make_config(
            runtime_mode=mode,
            max_turns=1,
            max_sessions=1,
            context_size=16_000,
        )
        client = MagicMock()
        client.chat.return_value = TurnResult(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=10,
                cached_tokens=40,
            ),
        )
        client.build_assistant_message.return_value = {
            "role": "assistant",
            "content": "Done.",
        }
        with (
            patch("scripts.llm_solver.harness.loop._auto_commit"),
            patch.object(Session, "_get_server_ctx", return_value=16_000),
        ):
            assert solve_task(work, cfg, client, artifacts_dir=artifacts) is True
        return [json.loads(line) for line in (artifacts / ".trace.jsonl").read_text().splitlines()]

    assistant_events = run("assistant", tmp_path / "assistant")
    measurement_events = run("measurement", tmp_path / "measurement")

    facts = [event for event in assistant_events if event.get("event") == "session_usage"]
    assert len(facts) == 1
    assert facts[0] == {
        "event": "session_usage",
        "trace_schema_version": 2,
        "session_number": 1,
        "scope": "all_model_responses",
        "input_tokens": 100,
        "output_tokens": 10,
        "cached_tokens": 40,
        "cost": None,
        "quota": None,
    }
    assert not any(event.get("event") == "session_usage" for event in measurement_events)


def test_session_usage_event_does_not_change_model_facing_state_projection():
    start = {"event": "session_start", "session_number": 1}
    usage = _usage_event(1)

    before = project([start], max_result_chars=20_000)
    after = project([start, usage], max_result_chars=20_000)

    for section in ("state", "trace", "gates", "evidence", "inference"):
        assert after[section] == before[section]


def test_usage_command_is_offline_repeatable_and_byte_for_byte_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    assist_root = tmp_path / "assist"
    work = tmp_path / "work"
    work.mkdir()
    store = SessionStore(assist_root)
    record = store.create_session(
        cwd=work,
        model="test-model",
        prompt_text="Fix it.",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    artifacts = record.artifact_path
    artifacts.mkdir(parents=True)
    _write_trace(
        artifacts / ".trace.jsonl",
        [
            {"event": "session_start", "session_number": 1},
            _usage_event(
                1,
                input_tokens=100,
                output_tokens=20,
                cached_tokens=40,
            ),
            {"event": "session_end", "session_number": 1},
        ],
    )
    (artifacts / "session.json").write_text('{"model":"test-model"}\n')
    (artifacts / "metrics.json").write_text('{"metrics":{"total_tokens":999}}\n')
    (artifacts / "checkpoint.json").write_text('{"status":"completed"}\n')
    (artifacts / "transcript.log").write_text("verbatim model exchange\n")
    (artifacts / "prompt.txt").write_text("Fix it.\n")
    (artifacts / ".solver").mkdir()
    (artifacts / ".solver" / "state.json").write_text('{"state":{}}\n')
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(assist_root))
    before = _tree_bytes(assist_root)

    blocked = AssertionError("usage report crossed an offline boundary")
    with (
        patch("scripts.llm_assist.__main__._make_client", side_effect=blocked),
        patch("scripts.llm_assist._auth.CredentialSession.access", side_effect=blocked),
        patch("requests.sessions.Session.request", side_effect=blocked),
        patch("socket.create_connection", side_effect=blocked),
    ):
        assert main(["usage", record.session_id]) == 0
        first = capsys.readouterr().out
        assert main(["usage", record.session_id]) == 0
        second = capsys.readouterr().out

    assert first == second
    assert first == (
        f"session_id: {record.session_id}\n"
        f"session_ref: {record.short_id}\n"
        "segments: 1\n"
        "input_tokens: 100\n"
        "output_tokens: 20\n"
        "cached_tokens: 40\n"
        "cache_ratio: 40.00%\n"
        "cost: unknown\n"
        "quota: unknown\n"
    )
    assert _tree_bytes(assist_root) == before


def test_read_only_store_does_not_create_missing_state(tmp_path: Path):
    missing = tmp_path / "missing-assist-home"

    with pytest.raises(FileNotFoundError, match="session store does not exist"):
        SessionStore(missing, read_only=True)

    assert not missing.exists()


def test_read_only_store_reads_legacy_rows_without_running_migrations(tmp_path: Path):
    root = tmp_path / "legacy-assist-home"
    root.mkdir()
    database = root / "sessions.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            create table sessions (
                session_id text primary key,
                created_at text not null,
                updated_at text not null,
                cwd text not null,
                artifact_dir text not null,
                model text not null,
                status text not null,
                last_finish_reason text,
                prompt_text text not null,
                prompt_source text not null,
                context_mode text not null,
                system_prompt_path text,
                config_paths_json text not null
            )
            """
        )
        connection.execute(
            """
            insert into sessions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_12345678",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                str(tmp_path / "work"),
                str(root / "sessions" / "legacy_12345678"),
                "legacy-model",
                "completed",
                "stop",
                "Fix it.",
                "inline",
                "full",
                None,
                "[]",
            ),
        )
    before = database.read_bytes()

    record = SessionStore(root, read_only=True).get_session("legacy_12345678")

    assert record is not None
    assert record.provider is None
    assert record.worktree_path is None
    assert database.read_bytes() == before
