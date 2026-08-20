from pathlib import Path

from scripts.llm_solver.harness.raw_trace_state_ledger import (
    RawTraceStateLedger,
    replay_events,
)


def tool_event(
    turn: int,
    *,
    tool_name: str = "bash",
    args_summary: str = "",
    result_summary: str = "",
    prompt_tokens: int = 0,
    source_write_like: bool = False,
    write_like: bool = False,
    gate_blocked: bool = False,
    source_write_paths: list[str] | None = None,
) -> dict[str, object]:
    return {
        "event": "tool_call",
        "turn_number": turn,
        "tool_name": tool_name,
        "args_summary": args_summary,
        "result_summary": result_summary,
        "prompt_tokens": prompt_tokens,
        "source_write_like": source_write_like,
        "write_like": write_like,
        "gate_blocked": gate_blocked,
        "source_write_paths": source_write_paths or [],
    }


def test_ledger_ignores_non_tool_events() -> None:
    ledger = RawTraceStateLedger()

    assert ledger.update({"event": "session_start", "turn_number": 0}) is None
    assert ledger.last_snapshot is None


def test_repeated_args_use_only_visible_prefix() -> None:
    events = [
        tool_event(1, args_summary="cmd='sed -n 1,20p pkg/a.py'", result_summary="alpha", prompt_tokens=100),
        tool_event(
            2,
            args_summary="cmd='python - <<PY'",
            result_summary="wrote pkg/a.py",
            prompt_tokens=125,
            source_write_like=True,
            write_like=True,
            source_write_paths=["pkg/a.py"],
        ),
        tool_event(3, args_summary="cmd='sed -n 1,20p pkg/a.py'", result_summary="beta", prompt_tokens=160),
    ]

    snapshots = replay_events(events)
    repeated = snapshots[-1]

    assert repeated.args_seen_before == 1
    assert repeated.args_prev_turns == "1"
    assert repeated.turns_since_same_args == 2
    assert repeated.source_write_count_before == 1
    assert repeated.write_count_before == 1
    assert repeated.source_writes_since_last_same_args == 1
    assert repeated.writes_since_last_same_args == 1
    assert repeated.prompt_delta_since_last_same_args == 60


def test_since_last_same_args_excludes_the_prior_matching_turn_itself() -> None:
    events = [
        tool_event(
            1,
            args_summary="cmd='python edit.py'",
            result_summary="changed",
            source_write_like=True,
            write_like=True,
            source_write_paths=["pkg/a.py"],
        ),
        tool_event(
            2,
            args_summary="cmd='python edit.py'",
            result_summary="changed",
            source_write_like=True,
            write_like=True,
            source_write_paths=["pkg/a.py"],
        ),
    ]

    repeated = replay_events(events)[-1]

    assert repeated.args_seen_before == 1
    assert repeated.result_seen_before == 1
    assert repeated.pair_seen_before == 1
    assert repeated.source_writes_since_last_same_args == 0
    assert repeated.writes_since_last_same_args == 0


def test_done_and_gate_counters_are_raw_field_counters() -> None:
    events = [
        tool_event(1, tool_name="done", result_summary="finished"),
        tool_event(2, args_summary="cmd='pytest'", result_summary="blocked", gate_blocked=True),
        tool_event(3, args_summary="cmd='pytest'", result_summary="blocked"),
    ]

    snapshots = replay_events(events)

    assert snapshots[0].current_done_like is True
    assert snapshots[1].done_count_before == 1
    assert snapshots[2].gate_block_count_before == 1
    assert snapshots[2].gate_blocks_since_last_same_args == 0


def test_replay_through_turn_never_reads_future_events() -> None:
    events = [
        tool_event(1, args_summary="cmd='cat a.py'", result_summary="one"),
        tool_event(2, args_summary="cmd='cat b.py'", result_summary="two"),
        tool_event(3, args_summary="cmd='cat a.py'", result_summary="three"),
    ]

    snapshots = replay_events(events, through_turn=2)

    assert [snapshot.turn_number for snapshot in snapshots] == [1, 2]
    assert all(snapshot.args_seen_before == 0 for snapshot in snapshots)


def test_raw_trace_ledger_does_not_import_projection_or_regex_layers() -> None:
    source = Path("scripts/llm_solver/harness/raw_trace_state_ledger.py").read_text(encoding="utf-8")

    forbidden_tokens = [
        "project_tool_event",
        "recent_prefix_slots_from_events",
        "SIGNAL_DETECTORS",
        "classify_outcome",
        "_is_test_command",
        "import re",
        "re.compile",
    ]
    for token in forbidden_tokens:
        assert token not in source
