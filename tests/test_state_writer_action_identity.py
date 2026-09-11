"""Action identity survives compact display; fixtures execute no task tools."""
import json

import pytest

from scripts.llm_solver.harness.action_metadata import action_metadata
from scripts.llm_solver.harness import state_writer


def _read(path, *, summary=None):
    return {
        "event": "tool_call", "session_number": 1, "turn_number": 1,
        "tool_name": "read", "args_summary": summary or f"path='{path}'",
        "result_summary": "observed content",
        **action_metadata("read", {"path": path}),
    }


def _project(events):
    return state_writer.project(events, max_result_chars=1000, imperative_projection=True)


@pytest.mark.parametrize("display_cap", [20, 120, 200])
def test_distinct_long_paths_cannot_merge_into_one_hotspot(monkeypatch, display_cap):
    monkeypatch.setattr(state_writer, "_MAX_ACTION_CHARS", display_cap)
    prefix = "nested/" * 40
    events = [_read(prefix + name) for name in ("first.py", "second.py", "third.py")]
    result = _project(events)
    assert len({row["action"] for row in result["trace"]}) == 1
    assert result["process"]["read_hotspots"] == []


def test_legacy_display_equality_is_not_full_action_identity():
    event = _read("source.py")
    event.pop("action_sha256")
    assert _project([event] * 3)["process"]["read_hotspots"] == []


def test_same_display_can_represent_two_distinct_repeated_calls():
    prefix = "nested/" * 40
    first, second = _read(prefix + "one.py"), _read(prefix + "two.py")
    result = _project([first, second] * 3)
    hotspots = result["process"]["read_hotspots"]
    assert len(hotspots) == 2
    assert hotspots[0]["action"] == hotspots[1]["action"]
    assert {item["action_sha256"] for item in hotspots} == {
        first["action_sha256"], second["action_sha256"],
    }
    assert [item["count"] for item in hotspots] == [3, 3]


def test_recorded_identity_survives_summary_format_changes_and_unknown_rows():
    events = [_read("source.py", summary=summary) for summary in (
        "path='source.py'", 'path="source.py"', "source.py",
    )]
    unknown = dict(events[0])
    unknown.pop("action_sha256")
    events.insert(1, unknown)
    hotspots = _project(events)["process"]["read_hotspots"]
    assert len(hotspots) == 1
    assert hotspots[0]["count"] == 3
    assert hotspots[0]["action_sha256"] == events[0]["action_sha256"]


def test_default_projection_preserves_identity_without_enabling_process_claims():
    event = _read("source.py")
    result = state_writer.project([event], max_result_chars=1000)
    assert "process" not in result
    assert result["trace"][0]["action_sha256"] == event["action_sha256"]


def test_file_projection_and_memory_projection_use_same_identity(tmp_path):
    event = _read("nested/" * 40 + "source.py")
    events = [dict(event, turn_number=turn) for turn in range(3)]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in events))
    result = state_writer.project_from_trace(
        trace, max_result_chars=1000, imperative_projection=True,
    )
    assert result == _project(events)
    assert result["process"]["read_hotspots"][0]["action_sha256"] == event["action_sha256"]
