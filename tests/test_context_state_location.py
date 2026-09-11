"""State readers follow selected artifacts while source reads stay in the task."""
import json
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._loop._session_setup import build_context_manager
from scripts.llm_solver.harness.context_strategies import resolve_context_class


STATE_MODES = ["stateful", "compound", "focused_compound", "compound_selective",
               "salience", "yconcise", "yslot"]


def write_state(directory, marker, path="source.py"):
    target = directory / ".solver/state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "state": {"current_attempt": marker, "next_action": marker},
        "trace": [{"step": 1, "action": f"write(path='{path}')",
                   "source_write_paths": [path], "result": "written"}],
        "evidence": [], "todos": [], "gates": [], "inference": [],
    }))
    return target


def build(mode, task, artifacts, session=1):
    cfg = make_config(min_turns_before_context=0, context_ignore_state=False)
    return build_context_manager(resolve_context_class(mode), cfg, task, "Task", session,
                                 None, artifact_dir=artifacts)


@pytest.mark.parametrize("mode", STATE_MODES)
@pytest.mark.parametrize("state_exists", [False, True])
def test_selected_state_does_not_fall_back_to_project_names(tmp_path, mode, state_exists):
    task, artifacts = tmp_path / "task", tmp_path / "records"
    task.mkdir()
    artifacts.mkdir()
    original = write_state(task, "PROJECT_STATE_MUST_NOT_BE_READ",
                           "PROJECT_STATE_MUST_NOT_BE_READ.py").read_bytes()
    if state_exists:
        write_state(artifacts, "SELECTED_STATE_AVAILABLE", "SELECTED_STATE_AVAILABLE.py")
    ctx = build(mode, task, artifacts)
    ctx.add_system("System")
    ctx.add_user("Task")
    messages = json.dumps(ctx.get_messages())
    assert "PROJECT_STATE_MUST_NOT_BE_READ" not in messages
    assert ("SELECTED_STATE_AVAILABLE" in messages) is state_exists
    assert (task / ".solver/state.json").read_bytes() == original


@pytest.mark.parametrize("mode", ["stateful", "compound", "yconcise", "yslot", "concise", "slot"])
@pytest.mark.parametrize("source_path", ["source.py", "domain/state.json"])
def test_resumed_context_reads_task_files_from_external_state(tmp_path, mode, source_path):
    task, artifacts = tmp_path / "task", tmp_path / "records"
    task.mkdir()
    artifacts.mkdir()
    write_state(task, "wrong record", "wrong.py")
    write_state(artifacts, "selected record", source_path)
    (task / source_path).parent.mkdir(parents=True, exist_ok=True)
    (task / source_path).write_text("CURRENT_TASK_SOURCE\n")
    (task / "wrong.py").write_text("WRONG_PROJECT_RECORD\n")
    (artifacts / source_path).parent.mkdir(parents=True, exist_ok=True)
    (artifacts / source_path).write_text("PRIVATE_ARTIFACT_FILE\n")
    ctx = build(mode, task, artifacts, session=2)
    if hasattr(ctx, "_recent_tool_results"):
        body = json.dumps(list(ctx._recent_tool_results))
    else:
        body = str(ctx._ws.files)
    assert "CURRENT_TASK_SOURCE" in body
    assert "WRONG_PROJECT_RECORD" not in body
    assert "PRIVATE_ARTIFACT_FILE" not in body
    assert ctx._cwd == task


def test_legacy_factory_location_still_uses_task(tmp_path):
    write_state(tmp_path, "LEGACY_STATE")
    cfg = make_config(min_turns_before_context=0, context_ignore_state=False)
    ctx = build_context_manager(resolve_context_class("stateful"), cfg, tmp_path,
                                "Task", 1, None)
    ctx.add_system("System")
    ctx.add_user("Task")
    assert "LEGACY_STATE" in json.dumps(ctx.get_messages())


@pytest.mark.parametrize("default_context", [False, True])
@pytest.mark.parametrize("state_exists", [False, True])
def test_driver_binds_selected_artifacts_to_model_context(tmp_path, monkeypatch, default_context, state_exists):
    from scripts.llm_solver.harness.loop import Session, solve_task
    from scripts.llm_solver.server.types import TurnResult, Usage

    task, artifacts = tmp_path / "task", tmp_path / "records"
    task.mkdir()
    artifacts.mkdir()
    write_state(task, "PROJECT_STATE_MUST_NOT_BE_READ")
    if state_exists:
        write_state(artifacts, "SELECTED_STATE_AVAILABLE")
    cfg = make_config(max_turns=1, max_sessions=1, min_turns_before_context=0,
                      context_ignore_state=False, state_writer_enabled=False,
                      auto_commit=False, turn_snapshots_enabled=False,
                      tools_file_checkpoints_enabled=False, analysis_task_format="generic")
    client = MagicMock()
    client.chat.return_value = TurnResult(content="Done", tool_calls=[], finish_reason="stop",
                                         usage=Usage(prompt_tokens=10, completion_tokens=2))
    client.build_assistant_message.return_value = {"role": "assistant", "content": "Done"}
    monkeypatch.setattr(Session, "_get_server_ctx", lambda self: cfg.context_size)
    solve_task(task, cfg, client, initial_prompt="Task", artifacts_dir=artifacts,
               context_class=None if default_context else resolve_context_class("stateful"))
    assert client.chat.called
    messages = json.dumps(client.chat.call_args.args[0])
    assert ("SELECTED_STATE_AVAILABLE" in messages) is state_exists
    assert "PROJECT_STATE_MUST_NOT_BE_READ" not in messages


@pytest.mark.parametrize("state_exists", [False, True])
def test_reconstructed_context_keeps_selected_state_after_cache_invalidation(tmp_path, state_exists):
    from scripts.llm_solver.harness.turn_snapshots import ConversationSnapshot, _build_context_from_snapshot

    task, artifacts = tmp_path / "task", tmp_path / "records"
    task.mkdir()
    artifacts.mkdir()
    write_state(task, "PROJECT_STATE_MUST_NOT_BE_READ")
    if state_exists:
        write_state(artifacts, "SELECTED_STATE_AVAILABLE")
    original = build("stateful", task, artifacts)
    original.add_system("System")
    original.add_user("Task")
    before = copy.deepcopy(original.get_messages())
    snapshot = ConversationSnapshot(1, 0, "fixture-no-git", "Task",
                                    copy.deepcopy(original.get_history_messages()), before)
    session = SimpleNamespace(context=original, cwd=str(task), _artifact_dir=artifacts,
                              cfg=make_config(min_turns_before_context=0, context_ignore_state=False))
    restored = _build_context_from_snapshot(session, snapshot)
    assert restored.get_messages() == before
    restored.add_tool_result("fixture", "Ordinary completed read", tool_name="read")
    after = json.dumps(restored.get_messages())
    assert "PROJECT_STATE_MUST_NOT_BE_READ" not in after
    assert ("SELECTED_STATE_AVAILABLE" in after) is state_exists
