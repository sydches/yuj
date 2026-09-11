"""Known audit 086 counterexamples use real commands and dispatch consumers."""
import json
import shlex
import shutil
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.file_changes import observed_dispatch
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def assert_projection_consumers(events, event, changed):
    from scripts.llm_solver.harness.state_writer import project
    from scripts.llm_solver.harness.context_strategies.salience_context import SalienceContext
    from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event

    state = project(events, max_result_chars=2000, imperative_projection=True)
    item = next(item for item in state["trace"] if item["turn"] == event["turn_number"])
    assert item["file_changes"] == event["file_changes"]
    assert state["process"]["last_mutation_step"] == (item["step"] if changed else None)
    assert state["process"]["last_failed_mutation_step"] is None
    assert state["process"]["phase"] == ("post_mutation_unverified" if changed else "pre_mutation_discovery")
    assert SalienceContext._is_mutation_item(item) is changed
    assert SalienceContext._is_successful_mutation_item(item) is changed
    slot = project_tool_event(event)
    assert slot["source_mutation"] == ("true" if changed else "false")
    assert slot["effective_source_mutation"] == ("true" if changed else "false")


CASES = [
    ("python", "from pathlib import Path; Path('module.py').write_text('after')", True),
    ("node", "require('fs').writeFileSync('module.py', 'after')", True),
    ("printed", 'print("Path(\'module.py\').write_text(\'after\')")', False),
    ("failed", "from pathlib import Path; Path('module.py').write_text('after'); raise SystemExit(1)", True),
    ("abort", "from pathlib import Path; Path('module.py').write_text('after'); raise SystemExit(1)", True),
    ("rename", "from pathlib import Path; Path('module.py').rename('renamed.py')", True),
    ("delete", "from pathlib import Path; Path('module.py').unlink()", True),
]


@pytest.mark.parametrize("case,program,changed", CASES)
def test_real_dispatch_consumers(tmp_path, case, program, changed, sandbox=False):
    task = tmp_path / "task"
    task.mkdir()
    (task / "module.py").write_text("before")
    executable = shutil.which("node") if case == "node" else sys.executable
    if executable is None:
        pytest.skip("Node is unavailable")
    cmd = shlex.join([executable, "-e" if case == "node" else "-c", program])
    cfg = make_config(max_turns=2, sandbox_bash=sandbox, turn_snapshots_enabled=True,
                      auto_commit=False, duplicate_abort=20,
                      error_abort_threshold=1 if case == "abort" else 0)
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(content=None, tool_calls=[ToolCall(id="c1", name="bash", arguments={"cmd": cmd})],
                   finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5)),
        TurnResult(content="finished", tool_calls=[], finish_reason="stop",
                   usage=Usage(prompt_tokens=10, completion_tokens=5)),
    ]
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = tmp_path / "trace.jsonl"
    with trace.open("a") as sink, patch(
        "scripts.llm_solver.harness.turn_snapshots.snapshot", return_value="fixture_snapshot"
    ) as snapshot:
        session = Session(cfg, client, "fixture", "run the fixture", str(task),
                          trace_file=sink, session_number=1)
        session._guards.verified_since_mutation = True
        session._guards.formal_verification_passed_since_mutation = True
        session.run()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    event = next(e for e in events if e.get("event") == "tool_call" and e.get("tool_call_id") == "c1")
    assert session._guards.has_mutated is changed
    assert snapshot.call_count == int(changed)
    assert event["source_write_like"] is changed
    assert (event["action_class"] == "source_write") is changed
    observation = event["file_changes"]
    assert observation["status"] == ("changed" if changed else "unchanged_metadata")
    assert observation["tool_call_id"] == "c1"
    assert observation["task_binding"]["working_directory"] == str(task)
    if changed:
        assert "module.py" in observation["changed_paths"]
        assert not session._guards.verified_since_mutation
        assert not session._guards.formal_verification_passed_since_mutation
        assert "module.py" in session._guards.post_mutation_source_paths
    if case == "rename":
        assert observation["changed_paths"] == ["module.py", "renamed.py"]
    if case == "printed":
        assert event["predicted_source_write_like"] is True
        assert (task / "module.py").read_text() == "before"
    if case in {"failed", "abort"}:
        assert event["exit_status"] == 1
    assert_projection_consumers(events, event, changed)


@pytest.mark.parametrize("status", ["unchanged_metadata", "unavailable", "incomplete"])
def test_present_observation_never_falls_back_to_predicted_write(status):
    from scripts.llm_solver.harness.action_metadata import action_metadata

    args = {"cmd": "python -c \"from pathlib import Path; Path('module.py').write_text('after')\""}
    event = {"event": "tool_call", "session_number": 1, "turn_number": 1,
        "tool_call_id": "fixture", "tool_name": "bash", "args_summary": repr(args),
        **action_metadata("bash", args), "outcome_version": "native_execution_v1",
        "outcome": "completed", "exit_status": 0, "pass_fail": "unknown",
        "file_changes": {"basis": "native_entry_metadata_v1", "status": status, "changed_paths": []}}
    assert_projection_consumers([event], event, False)


def test_native_dispatch_consumers(bwrap, tmp_path):
    test_real_dispatch_consumers(tmp_path, *CASES[1], sandbox=True)


def test_observation_uses_native_view(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / "file\nname").write_text("native")
    host = tmp_path / "view"
    (host / "hidden").write_text("host")
    metadata = {}

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        files.run('printf changed > "$1"', ["file\nname"], None)
        kwargs["execution_metadata"]["executed"] = True

    with activate_task_files(files, host_root=host):
        execute("bash", {"cmd": "fixture"}, cwd=str(host), cfg=make_config(),
                execution_metadata=metadata)
    assert metadata["file_changes"]["changed_paths"] == ["file\nname"]
    assert (host / "hidden").read_text() == "host"


def test_unavailable_observation_does_not_invent_no_change(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import file_changes

    def unavailable(*args):
        raise OSError("fixture denied")

    monkeypatch.setattr(file_changes, "_inventory", unavailable)
    metadata = {}

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        (tmp_path / "file").write_text("changed")

    execute("bash", {"cmd": "fixture"}, cwd=str(tmp_path),
            cfg=make_config(sandbox_bash=False), execution_metadata=metadata)
    assert metadata["file_changes"]["status"] == "unavailable"
    assert (tmp_path / "file").read_text() == "changed"


def test_exhausted_budget_does_not_launch_tool(tmp_path):
    from scripts.llm_solver.harness.time_budget import run_time_budget

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        pytest.fail("tool must not run after its allowance is exhausted")

    metadata = {}
    with run_time_budget(1e-9):
        result = execute("bash", {"cmd": "fixture"}, cwd=str(tmp_path),
                         cfg=make_config(sandbox_bash=False), execution_metadata=metadata)
    assert "budget is exhausted" in result
    assert metadata["executed"] is False
    assert metadata["file_changes"]["status"] == "unavailable"
