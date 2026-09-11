"""117: CLI callers can explicitly select replaceable status destinations."""
import json
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver import __main__ as measurement
from scripts.llm_solver.server.types import TurnResult, Usage


@pytest.fixture
def cli_client(tmp_path, monkeypatch):
    cfg = make_config(max_sessions=1, max_turns=1, sandbox_bash=False,
                      auto_commit=False, turn_snapshots_enabled=False)
    client = MagicMock()
    client.profile = None
    client.query_server_context.return_value = cfg.context_size
    client.chat.return_value = TurnResult(content="finished", tool_calls=[],
                                         finish_reason="stop", usage=Usage(10, 2))
    client.requests = []

    def chat(messages, *args, **kwargs):
        client.requests.append(json.loads(json.dumps(messages)))
        return client.chat.return_value

    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": "finished"}
    monkeypatch.setattr(measurement, "load_config", lambda **kw: cfg)
    monkeypatch.setattr(measurement, "_build_client", lambda *a: client)
    monkeypatch.setattr(measurement, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(measurement, "build_model_role_runtime", lambda **kw: None)
    monkeypatch.setattr(measurement, "_write_server_metadata", lambda *a: (None, None))
    monkeypatch.setattr(measurement, "_build_run_metadata",
                        lambda **kw: {"run_dir": str(kw["run_dir"])})
    return client


def assert_status(root):
    assert json.loads((root / "checkpoint.json").read_text())["status"] == "completed"
    assert "metrics" in json.loads((root / "metrics.json").read_text())
    assert (root / ".trace.jsonl").is_file()


def test_repeated_and_transparent_resume_use_explicit_destination(tmp_path, cli_client):
    task, artifacts = tmp_path / "task", tmp_path / "selected"
    task.mkdir()
    for name in ("checkpoint.json", "metrics.json"):
        (task / name).write_text("project-owned")
    args = [str(tmp_path / "run"), "--task", str(task), "--prompt-text", "Inspect the files.",
            "--artifacts-dir", str(artifacts)]
    for _ in range(2):
        assert measurement.main(args) == 0
        assert_status(artifacts)
    prior = [{"role": "system", "content": "prior system"},
             {"role": "user", "content": "PRIOR_TASK_MARKER"}]
    transcript = tmp_path / "prior.log"
    transcript.write_text("=== turn 001 input ===\n" + json.dumps({"messages": prior}) + "\n")
    assert measurement.main([*args, "--resume", str(transcript)]) == 0
    assert_status(artifacts)
    assert cli_client.chat.call_count == 3
    assert cli_client.requests[-1] == prior
    for name in ("checkpoint.json", "metrics.json"):
        assert (task / name).read_text() == "project-owned"


def test_separate_output_keeps_default_task_prompt_source(tmp_path, cli_client):
    task = tmp_path / "task"
    task.mkdir()
    (task / "prompt.txt").write_text("TASK_INPUT_MARKER")
    artifacts = tmp_path / "selected"
    assert measurement.main([str(tmp_path / "run"), "--task", str(task),
                             "--artifacts-dir", str(artifacts)]) == 0
    assert_status(artifacts)
    assert any("TASK_INPUT_MARKER" in row.get("content", "") for row in cli_client.requests[-1])


def test_pending_errors_keep_distinct_explicit_records(tmp_path, cli_client):
    run, artifacts = tmp_path / "run", tmp_path / "selected"
    original = {"first": '{"status":"error"}', "second": 'malformed status'}
    for name, checkpoint in original.items():
        task = run / "repos" / name
        task.mkdir(parents=True)
        (task / "prompt.txt").write_text("Inspect the files.")
        (task / "checkpoint.json").write_text(checkpoint)
    assert measurement.main([str(run), "--artifacts-dir", str(artifacts)]) == 0
    assert cli_client.chat.call_count == 2
    for name, checkpoint in original.items():
        assert_status(artifacts / name)
        assert (run / "repos" / name / "checkpoint.json").read_text() == checkpoint


@pytest.mark.parametrize("batch", [False, True])
def test_omitted_destination_retains_collision_refusal(tmp_path, cli_client, batch):
    run = tmp_path / "run"
    task = run / "repos" / "first"
    task.mkdir(parents=True)
    (task / "prompt.txt").write_text("Inspect the files.")
    prior = '{"status":"error"}'
    (task / "checkpoint.json").write_text(prior)
    args = [str(run)] + ([] if batch else ["--task", str(task)])
    with pytest.raises(ValueError, match="Refusing to replace automatic status"):
        measurement.main(args)
    cli_client.chat.assert_not_called()
    assert (task / "checkpoint.json").read_text() == prior
