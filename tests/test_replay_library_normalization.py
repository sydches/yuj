"""Library-shaped text is not proof that a replay difference is incidental."""
import json

import pytest

from test_replay_client import SYS, _resp, _transcript
from llm_solver.server.replay_client import ReplayClient, ReplayDivergence
from llm_solver.server.replay_client import normalize_volatile, normalize_volatile_v14_for_history


PAIRS = [
    ("Running Sphinx v5.0.0+/c9af4d7\n", "Running Sphinx v5.0.0+/ab12cd3\n"),
    ("Matplotlib is building the font cache; this may take a moment.\nbody\n", "body\n"),
]


@pytest.mark.parametrize("recorded,live", PAIRS)
@pytest.mark.parametrize("field,hashes", [("result_summary", False), ("args_summary", False),
                                        ("args_summary", True)])
def test_gate_rejects_library_text_changes(tmp_path, recorded, live, field, hashes):
    transcript = _transcript(tmp_path, [(SYS, _resp(tool="cat README.txt"))])
    event = {"event": "tool_call", "turn_number": 1, "tool_name": "bash",
             "args_summary": "cmd='cat README.txt'", "result_summary": "unchanged",
             "volatile_normalization_version": "replay_volatile_norm_v14"}
    event[field] = recorded
    if hashes:
        event["output_sha256"] = "same-output-hash"
    trace = tmp_path / "selected.trace.jsonl"
    trace.write_text(json.dumps(event) + "\n")
    client = ReplayClient(transcript, source_trace_path=trace)
    client.verify_executed_turn(dict(event))
    assert client.divergence is None
    changed = {**event, field: live}
    with pytest.raises(ReplayDivergence):
        client.verify_executed_turn(changed)
    assert client.divergence["field"] == field


@pytest.mark.parametrize("recorded,live", PAIRS)
def test_historical_inspection_preserves_old_comparison_without_changing_current(recorded, live):
    assert normalize_volatile_v14_for_history(recorded) == normalize_volatile_v14_for_history(live)
    assert normalize_volatile(recorded) != normalize_volatile(live)


def test_other_inherited_normalization_is_unchanged_not_certified():
    recorded, live = "PID: 12\n", "PID: 34\n"
    assert normalize_volatile(recorded) == normalize_volatile(live)
    assert normalize_volatile(recorded) == normalize_volatile_v14_for_history(recorded)
