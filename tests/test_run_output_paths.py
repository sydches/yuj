"""Output ownership does not depend on the task's directory depth."""
from types import SimpleNamespace
from unittest.mock import Mock
import logging
from pathlib import Path

import pytest

from _config_helpers import make_config
from scripts.llm_solver import __main__ as cli
from scripts.llm_solver._shared.telemetry_paths import telemetry_dir
from scripts.llm_solver.harness import savings, system_log
from scripts.llm_solver.harness._loop import _driver_setup as setup


def test_library_defaults_separate_same_named_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "_record_session_start_costs", lambda *args: None)
    transcripts = []
    client = SimpleNamespace(set_transcript=transcripts.append)
    try:
        for parent in ("first", "second"):
            task = tmp_path / parent / "project"
            task.mkdir(parents=True)
            setup.setup_savings_and_transcript(
                SimpleNamespace(), client, task, None, None, "", None, None,
            )
            root = telemetry_dir(task)
            assert transcripts[-1] == root / "transcripts" / "project.log"
            assert (root / "savings" / "project.jsonl").is_file()
            assert (root / "system_log.jsonl").is_file()
            assert list(task.iterdir()) == []
        assert transcripts[0] != transcripts[1]
        assert not (tmp_path / "savings").exists()
        assert not (tmp_path / "transcripts").exists()
    finally:
        savings.close_ledger()
        system_log.close_system_log()


def test_library_explicit_output_directories_remain_authoritative(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "_record_session_start_costs", lambda *args: None)
    task = tmp_path / "workspace" / "project"
    task.mkdir(parents=True)
    client = SimpleNamespace(set_transcript=Mock())
    savings_dir = tmp_path / "selected" / "counts"
    transcript_dir = tmp_path / "selected" / "messages"
    try:
        setup.setup_savings_and_transcript(
            SimpleNamespace(), client, task, savings_dir, transcript_dir,
            "", None, None,
        )
        client.set_transcript.assert_called_once_with(transcript_dir / "project.log")
        assert (savings_dir / "project.jsonl").is_file()
        assert (savings_dir.parent / "system_log.jsonl").is_file()
        assert not telemetry_dir(task).exists()
    finally:
        savings.close_ledger()
        system_log.close_system_log()


@pytest.mark.parametrize("spelling", ["dot", "parent", "symlink"])
def test_library_output_defaults_follow_resolved_workspace(tmp_path, monkeypatch, spelling):
    task = tmp_path / "projects" / "sample"
    task.mkdir(parents=True)
    (task / "child").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(task, target_is_directory=True)
    monkeypatch.chdir(task)
    supplied = {"dot": Path("."), "parent": Path("child/.."), "symlink": alias}[spelling]
    monkeypatch.setattr(setup, "_record_session_start_costs", lambda *args: None)
    client = SimpleNamespace(set_transcript=Mock())
    expected = task.parent / ".yuj_sample"
    try:
        setup.setup_savings_and_transcript(
            SimpleNamespace(), client, supplied, None, None, "", None, None,
        )
        assert telemetry_dir(supplied) == expected
        client.set_transcript.assert_called_once_with(expected / "transcripts" / "sample.log")
        assert (expected / "savings" / "sample.jsonl").is_file()
        assert (expected / "system_log.jsonl").is_file()
        assert sorted(p.name for p in task.iterdir()) == ["child"]
        assert not (tmp_path / ".yuj_alias").exists()
    finally:
        savings.close_ledger()
        system_log.close_system_log()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_cli_passes_owned_output_directories(tmp_path, monkeypatch, batch, explicit):
    run = tmp_path / "records"
    task = run / "repos" / "task" if batch else tmp_path / "unrelated" / "deep" / "task"
    task.mkdir(parents=True)
    (task / "prompt.txt").write_text("Synthetic task; no model will run.\n")
    cfg = make_config(runtime_worktree="off")
    client = SimpleNamespace(query_server_context=lambda: None)
    solve = Mock(return_value=True)
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cli, "load_profile", lambda *args: SimpleNamespace(
        name="fixture", inherits=(), profile_dir=None,
    ))
    monkeypatch.setattr(cli, "validate_model_role_profiles", lambda **kwargs: None)
    monkeypatch.setattr(cli, "build_model_role_runtime", lambda **kwargs: None)
    monkeypatch.setattr(cli, "_build_client", lambda *args: client)
    monkeypatch.setattr(cli, "_write_server_metadata", lambda *args: (None, None))
    monkeypatch.setattr(cli, "_build_run_metadata", lambda **kwargs: {"run_dir": str(run)})
    monkeypatch.setattr(cli, "_write_session_json", lambda *args: None)
    monkeypatch.setattr(cli, "solve_task", solve)
    args = [str(run)]
    if not batch:
        args += ["--task", str(task)]
    savings_dir = tmp_path / "chosen-counts" if explicit else run / "savings"
    transcript_dir = tmp_path / "chosen-messages" if explicit else run / "transcripts"
    if explicit:
        args += ["--savings-dir", str(savings_dir), "--transcript-dir", str(transcript_dir)]
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    try:
        assert cli.main(args) == 0
        solve.assert_called_once()
        assert solve.call_args.args[0] == task
        assert solve.call_args.kwargs["savings_dir"] == savings_dir
        assert solve.call_args.kwargs["transcript_dir"] == transcript_dir
    finally:
        for handler in list(root_logger.handlers):
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
                handler.close()
