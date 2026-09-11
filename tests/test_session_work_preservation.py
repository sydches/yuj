"""Completed task writes survive process loss without commits or time rewrites."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_CHILD = r'''
import json
from pathlib import Path
import signal
import sys
from unittest.mock import MagicMock, patch

from _config_helpers import make_config
from llm_solver.harness.loop import Session, solve_task
from llm_solver.server.types import ToolCall, TurnResult, Usage

root, mode = Path(sys.argv[1]), sys.argv[2]
task, artifacts = root / "task", root / "artifacts"
cfg = make_config(max_turns=3, max_sessions=1, analysis_task_format="generic",
                  auto_commit=False, turn_snapshots_enabled=False,
                  tools_file_checkpoints_enabled=False,
                  sandbox_env_inherit="none", sandbox_env_set={})
client = MagicMock()
calls = 0

def chat(*args, **kwargs):
    global calls
    calls += 1
    if mode == "interrupt":
        if calls == 1:
            tool = ToolCall(id="saved-write", name="write",
                            arguments={"path": "created.txt", "content": "saved work\n"})
        else:
            (root / "ready").write_text("previous tool returned")
            while True:
                signal.pause()
    elif calls == 1:
        assert (task / "created.txt").read_text() == "saved work\n"
        tool = ToolCall(id="resumed-read", name="read", arguments={"path": "created.txt"})
    else:
        return TurnResult(content="Finished.", tool_calls=[], finish_reason="stop",
                          usage=Usage(prompt_tokens=15, completion_tokens=2))
    return TurnResult(content="Inspect the saved work." if mode == "resume" else "Write the file.",
                      tool_calls=[tool], finish_reason="tool_calls",
                      usage=Usage(prompt_tokens=10, completion_tokens=3))

def assistant(content, tool_calls, replay=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
            for call in tool_calls]
    return message

client.chat.side_effect = chat
client.build_assistant_message.side_effect = assistant
with patch.object(Session, "_get_server_ctx", return_value=cfg.context_size):
    result = solve_task(task, cfg, client, initial_prompt="Create and inspect the file.",
                        artifacts_dir=artifacts, resume_from_artifacts=mode == "resume")
if mode == "resume":
    assert result is True
'''


@pytest.mark.parametrize("with_git", [False, True], ids=["no-git", "git"])
@pytest.mark.parametrize("exit_signal", [signal.SIGTERM, signal.SIGKILL], ids=["term", "kill"])
def test_process_loss_preserves_work_index_times_and_resume(tmp_path, with_git, exit_signal):
    make = shutil.which("make")
    if make is None:
        pytest.skip("GNU make is required for the timestamp-sensitive contrast")
    task = tmp_path / "task"
    task.mkdir()
    (task / "source.txt").write_text("build input\n")
    (task / "built.txt").write_text("old output\n")
    (task / "Makefile").write_text("built.txt: source.txt\n\tcp source.txt built.txt\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside input\n")
    (task / "outside-link").symlink_to(outside)
    times = {task / "source.txt": 1_710_000_000_000_000_000,
             task / "built.txt": 1_700_000_000_000_000_000,
             outside: 1_690_000_000_000_000_000}
    for path, stamp in times.items():
        os.utime(path, ns=(stamp, stamp))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(task), *args], stderr=subprocess.PIPE)

    if with_git:
        git("init")
        git("add", "-A")
        git("-c", "user.name=fixture", "-c", "user.email=fixture@local", "commit", "-m", "base")
        (task / "user.txt").write_text("staged\n")
        git("add", "user.txt")
        (task / "user.txt").write_text("unstaged\n")
        head, index = git("rev-parse", "HEAD"), (task / ".git" / "index").read_bytes()
    assert subprocess.run([make, "-q", "built.txt"], cwd=task).returncode == 1

    child = tmp_path / "child.py"
    child.write_text(_CHILD)
    child_env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(_ROOT / "scripts"), str(_ROOT / "tests")])}
    child_env.pop("YUJ_CONTAINER", None)
    command = [sys.executable, str(child), str(tmp_path)]
    log_path = tmp_path / "child.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([*command, "interrupt"], env=child_env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (tmp_path / "ready").exists(), log_path.read_text()
            process.send_signal(exit_signal)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert process.returncode in (-int(exit_signal), 128 + int(exit_signal))
    assert (task / "created.txt").read_text() == "saved work\n"
    trace_path = tmp_path / "artifacts" / ".trace.jsonl"
    before = trace_path.read_bytes()
    events = [json.loads(line) for line in before.splitlines()]
    assert any(ev.get("tool_call_id") == "saved-write" and ev["event"] == "tool_call" for ev in events)
    assert not any(ev["event"] == "session_end" for ev in events)
    assert not (tmp_path / "artifacts" / "checkpoint.json").exists()

    resumed = subprocess.run([*command, "resume"], env=child_env, capture_output=True, text=True, timeout=20)
    assert resumed.returncode == 0, resumed.stderr
    assert trace_path.read_bytes().startswith(before)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert sum(ev["event"] == "turn_aborted" for ev in events) == 1
    assert [ev["session_number"] for ev in events if ev["event"] == "session_start"] == [1, 2]
    read = next(ev for ev in events if ev.get("tool_call_id") == "resumed-read" and ev["event"] == "tool_call")
    assert "saved work" in read["result_summary"]
    for path, stamp in times.items():
        assert path.stat().st_mtime_ns == stamp
    assert subprocess.run([make, "-q", "built.txt"], cwd=task).returncode == 1
    if with_git:
        assert git("rev-parse", "HEAD") == head
        assert (task / ".git" / "index").read_bytes() == index
        assert (task / "user.txt").read_text() == "unstaged\n"
    else:
        assert not (task / ".git").exists()
