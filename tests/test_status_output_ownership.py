"""Automatic status names cannot authorize replacement of project files."""
import json
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.harness.loop import solve_task
from scripts.llm_solver.harness.solver import write_checkpoint, write_run_metrics
from scripts.llm_solver.server.types import TurnResult, Usage


@pytest.mark.parametrize("name", ["checkpoint.json", "metrics.json"])
@pytest.mark.parametrize("kind", ["file", "link", "dangling_link"])
def test_automatic_status_collision_refused_before_generation(tmp_path, name, kind):
    task = tmp_path / "task"
    task.mkdir()
    target = tmp_path / "project.json"
    # Even plausible harness JSON does not prove that the harness owns a file.
    original = '{"solver":"llm_solver","status":"completed","metrics":{}}'
    destination = task / name
    if kind == "file":
        destination.write_text(original)
    else:
        if kind == "link":
            target.write_text(original)
        destination.symlink_to(target)
    client = MagicMock()
    with pytest.raises(ValueError, match="Refusing to replace automatic status"):
        solve_task(task, make_config(), client, initial_prompt="Inspect the project.")
    client.chat.assert_not_called()
    if kind == "dangling_link":
        assert destination.is_symlink() and not target.exists()
    else:
        assert destination.read_text() == original


@pytest.mark.parametrize("writer,name,args", [
    (write_checkpoint, "checkpoint.json", ("model", "completed")),
    (write_run_metrics, "metrics.json", ({"tokens": 1}, {})),
])
def test_exclusive_status_write_preserves_late_collision(tmp_path, writer, name, args):
    destination = tmp_path / name
    destination.write_text("created after startup")
    with pytest.raises(FileExistsError):
        writer(tmp_path, *args, overwrite=False)
    assert destination.read_text() == "created after startup"


def test_explicit_artifacts_refresh_status_and_preserve_project_names(tmp_path):
    task, artifacts = tmp_path / "task", tmp_path / "artifacts"
    task.mkdir()
    for name in ("checkpoint.json", "metrics.json"):
        (task / name).write_text("project-owned")
    client = MagicMock()
    client.chat.return_value = TurnResult(
        content="done", tool_calls=[], finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )
    client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}
    for _ in range(2):
        assert solve_task(task, make_config(max_sessions=1), client,
                          initial_prompt="Inspect the project.", artifacts_dir=artifacts)
    assert client.chat.call_count == 2
    assert json.loads((artifacts / "checkpoint.json").read_text())["status"] == "completed"
    assert "metrics" in json.loads((artifacts / "metrics.json").read_text())
    for name in ("checkpoint.json", "metrics.json"):
        assert (task / name).read_text() == "project-owned"
