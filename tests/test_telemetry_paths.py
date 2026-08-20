"""Telemetry must live outside the model's workspace."""
from pathlib import Path

from scripts.llm_solver._shared.telemetry_paths import (
    ensure_telemetry_dir,
    legacy_trace_path,
    resolve_trace_path,
    telemetry_dir,
    telemetry_file,
    trace_path,
)


def test_telemetry_dir_is_outside_the_workspace(tmp_path):
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    assert not telemetry_dir(repo).is_relative_to(repo)


def test_trace_is_not_written_into_the_workspace(tmp_path):
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    assert not trace_path(repo).is_relative_to(repo)


def test_ledger_and_detector_land_beside_the_trace(tmp_path):
    """Both anchor to the trace's parent; moving the trace must move them."""
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    for name in ("adaptive_control_ledger.jsonl", "llm_hurdle_detector.jsonl"):
        p = telemetry_file(repo, name)
        assert p.parent == trace_path(repo).parent
        assert not p.is_relative_to(repo)


def test_absolute_ledger_path_is_honoured(tmp_path):
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    pinned = tmp_path / "elsewhere" / "ledger.jsonl"
    assert telemetry_file(repo, str(pinned)) == pinned


def test_workspace_stays_clean_after_writing_telemetry(tmp_path):
    """The model's `ls` must show nothing harness-owned."""
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    (repo / "src.py").write_text("x = 1\n")

    ensure_telemetry_dir(repo)
    trace_path(repo).write_text('{"event": "session_start"}\n')
    telemetry_file(repo, "adaptive_control_ledger.jsonl").write_text("{}\n")

    assert sorted(p.name for p in repo.iterdir()) == ["src.py"]


def test_resolve_prefers_current_layout(tmp_path):
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    ensure_telemetry_dir(repo)
    trace_path(repo).write_text("current\n")
    legacy_trace_path(repo).write_text("legacy\n")
    assert resolve_trace_path(repo) == trace_path(repo)


def test_resolve_falls_back_to_pre_split_runs(tmp_path):
    """Historical runs wrote the trace inside the workspace; keep reading them."""
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    legacy_trace_path(repo).write_text("legacy\n")
    assert resolve_trace_path(repo) == legacy_trace_path(repo)


def test_resolve_reports_against_current_layout_when_absent(tmp_path):
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    assert resolve_trace_path(repo) == trace_path(repo)


def test_two_arms_of_the_same_task_do_not_differ_in_the_workspace(tmp_path):
    """Telemetry files do not change the model's workspace view."""
    def workspace_view(name, *, rescue):
        repo = tmp_path / name / "repo"
        repo.mkdir(parents=True)
        (repo / "src.py").write_text("x = 1\n")
        ensure_telemetry_dir(repo)
        trace_path(repo).write_text("x" * (5096 if rescue else 4656))
        if rescue:
            telemetry_file(repo, "adaptive_control_ledger.jsonl").write_text("{}\n")
            telemetry_file(repo, "llm_hurdle_detector.jsonl").write_text("{}\n")
        return sorted((p.name, p.stat().st_size) for p in repo.iterdir())

    assert workspace_view("baseline", rescue=False) == workspace_view("rescue", rescue=True)
