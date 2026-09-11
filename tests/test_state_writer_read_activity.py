"""Request counts cannot establish repeated information or required edits."""
import json

import pytest

from scripts.llm_solver.harness.action_metadata import action_metadata
from scripts.llm_solver.harness.state_writer import project, project_from_trace


def _read(path, *, result="content", blocked=False):
    return {
        "event": "tool_call", "tool_name": "read", "session_number": 1,
        "turn_number": 1, "args_summary": f"path='{path}'",
        "result_summary": result, "gate_blocked": blocked,
        **action_metadata("read", {"path": path}),
    }


def _project(events):
    return project(events, max_result_chars=1000, imperative_projection=True)


@pytest.mark.parametrize("count", [2, 8, 20])
def test_distinct_reads_report_windowed_requests_without_claiming_a_loop(count):
    events = [_read(f"section/{index}.txt") for index in range(count)]
    process = _project(events)["process"]
    assert process["read_loop"] is None
    assert process["read_activity"] == {
        "window_entries": min(count, 16),
        "read_like_requests": min(count, 16),
        "basis": "recorded_request_syntax",
        "progress": "unassessed",
    }
    assert process["required_next_action"] == ""
    assert "task" in process["suggested_next_action"]


def test_same_request_with_changed_output_is_only_request_recurrence():
    events = [_read("status.txt", result=f"observation {i}") for i in range(8)]
    process = _project(events)["process"]
    assert process["read_hotspots"][0]["count"] == 8
    assert process["read_loop"] is None
    assert process["read_activity"]["progress"] == "unassessed"


def test_blocked_requests_and_legacy_rows_do_not_become_completed_reads():
    events = [_read(f"part{i}", blocked=True) for i in range(8)]
    for event in events:
        event.pop("action_sha256")
    process = _project(events)["process"]
    assert process["read_activity"]["basis"] == "recorded_request_syntax"
    assert process["read_activity"]["read_like_requests"] == 8
    assert process["read_hotspots"] == []
    assert process["read_loop"] is None


def test_narrated_edit_does_not_create_a_requirement():
    event = _read("guide.md")
    event["reasoning"] = "Replace old with new in the guide."
    process = _project([event])["process"]
    assert process["phase"] == "candidate_edit_pending"
    assert process["required_next_action"] == ""
    assert "If" in process["suggested_next_action"]


def test_projection_boundary_and_file_replay(tmp_path):
    events = [_read(f"section/{i}.txt") for i in range(8)]
    assert "process" not in project(events, max_result_chars=1000)
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert project_from_trace(path, max_result_chars=1000, imperative_projection=True) == _project(events)
