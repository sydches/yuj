"""Invisible per-turn git snapshots — contract tests.

The feature's two promises:
  1. any snapshotted turn can be restored exactly (rewind/branch point);
  2. the model can never observe that snapshots happen (no log entry, no
     status change, no index disturbance).
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from tests.test_task_files import bwrap

from scripts.llm_solver._shared.telemetry_paths import telemetry_dir
from scripts.llm_solver.harness.turn_snapshots import (
    MAP_NAME,
    ensure_snapshot_setup,
    read_map,
    sha_at_or_before,
    snapshot,
)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout.strip()


def _make_repo(tmp_path):
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)])
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def test_snapshot_returns_sha_and_writes_map(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text("x = 2\n")
    sha = snapshot(repo, 5)
    assert sha and len(sha) >= 7
    assert read_map(repo) == [(5, sha)]


def test_restore_rebuilds_exact_state(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("fresh = True\n")
    sha = snapshot(repo, 7)
    # mutate further (later turns), then rewind to turn 7's snapshot
    (repo / "a.py").write_text("x = 999\n")
    (repo / "new.py").unlink()
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", sha, "--", "."])
    assert (repo / "a.py").read_text() == "x = 2\n"
    assert (repo / "new.py").read_text() == "fresh = True\n"


@pytest.mark.parametrize("ambient", [False, True])
def test_invisible_to_log_status_and_index(tmp_path, monkeypatch, ambient):
    if ambient:
        monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    else:
        monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    log_before = _git(repo, "log", "--oneline", "--all")
    (repo / "a.py").write_text("x = 3\n")
    status_before = _git(repo, "status", "--porcelain")
    staged_before = _git(repo, "diff", "--cached", "--name-only")
    assert snapshot(repo, 9)
    assert _git(repo, "log", "--oneline", "--all") == log_before
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "diff", "--cached", "--name-only") == staged_before


def test_snapshot_object_survives_and_parents_head(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    ensure_snapshot_setup(repo)
    (repo / "a.py").write_text("x = 4\n")
    sha = snapshot(repo, 3)
    head = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "rev-parse", f"{sha}^") == head
    assert _git(repo, "cat-file", "-t", sha) == "commit"


def test_multiple_turns_and_rewind_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    shas = {}
    for turn, content in ((4, "x = 40\n"), (9, "x = 90\n"), (15, "x = 150\n")):
        (repo / "a.py").write_text(content)
        shas[turn] = snapshot(repo, turn)
    assert sha_at_or_before(repo, 9) == shas[9]
    assert sha_at_or_before(repo, 12) == shas[9]
    assert sha_at_or_before(repo, 3) is None
    assert len({s for s in shas.values()}) == 3


def test_failure_is_silent_and_warned_once(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    not_repo = tmp_path / "run" / "repo"
    not_repo.mkdir(parents=True)  # no git here -> snapshot must fail softly
    session = SimpleNamespace()
    assert snapshot(not_repo, 1, session=session) is None
    assert getattr(session, "_snapshot_warned") is True
    assert snapshot(not_repo, 2, session=session) is None
    assert read_map(not_repo) == []


def test_map_lives_in_telemetry_not_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text("x = 5\n")
    snapshot(repo, 2)
    assert (telemetry_dir(repo) / MAP_NAME).exists()
    assert not (repo / MAP_NAME).exists()


@pytest.mark.parametrize("layout", ["root", "outside", "inside"])
def test_snapshot_excludes_owned_artifacts_only(tmp_path, monkeypatch, layout):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    owner = {
        "root": repo,
        "outside": tmp_path / "artifacts",
        "inside": repo / "artifacts [one]",
    }[layout]
    owner.mkdir(exist_ok=True)
    (repo / "project").mkdir()
    names = ("prompt.txt", "checkpoint.json", "metrics.json")
    for directory in (repo, repo / "project", owner):
        for name in names:
            (directory / name).write_text("before\n")
    session = SimpleNamespace(_artifact_dir=owner)
    first = snapshot(repo, 1, session=session)
    assert first
    for directory in (repo, repo / "project", owner):
        for name in names:
            (directory / name).write_text("after\n")
    second = snapshot(repo, 2, session=session)
    assert second
    entries = set(_git(repo, "ls-tree", "-r", "--name-only", second).splitlines())
    expected = {"a.py", *(f"project/{name}" for name in names)}
    if layout != "root":
        expected.update(names)
    assert entries == expected
    for path in expected - {"a.py"}:
        assert _git(repo, "show", f"{first}:{path}") == "before"
        assert _git(repo, "show", f"{second}:{path}") == "after"


@pytest.mark.parametrize("session", [None, SimpleNamespace(), SimpleNamespace(_artifact_dir=None)])
def test_snapshot_without_artifact_location_preserves_project_files(
    tmp_path, monkeypatch, session,
):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    names = ("prompt.txt", "checkpoint.json", "metrics.json")
    for name in names:
        (repo / name).write_text(f"project data: {name}\n")
    sha = snapshot(repo, 1, session=session)
    assert sha
    assert set(_git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()) == {
        "a.py", *names,
    }
    for name in names:
        assert _git(repo, "show", f"{sha}:{name}") == f"project data: {name}"


def test_snapshot_drops_newly_owned_paths_from_private_index(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    (repo / "metrics.json").write_text("project data\n")
    session = SimpleNamespace(_artifact_dir=tmp_path / "artifacts")
    first = snapshot(repo, 1, session=session)
    assert "metrics.json" in _git(repo, "ls-tree", "--name-only", first).splitlines()
    session._artifact_dir = repo
    second = snapshot(repo, 2, session=session)
    assert second
    assert "metrics.json" not in _git(repo, "ls-tree", "--name-only", second).splitlines()


@pytest.mark.parametrize("subdir", ["", "component", " component [one] 'quoted'"])
def test_local_session_converts_owned_paths_to_git_coordinates(tmp_path, monkeypatch, subdir):
    from unittest.mock import MagicMock
    from _config_helpers import make_config
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.harness.turn_snapshots import _snapshot_owned_paths

    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "isolated.gitconfig"))
    repo = _make_repo(tmp_path)
    workspace = repo / subdir
    workspace.mkdir(exist_ok=True)
    owner = workspace / "records"
    owner.mkdir()
    names = ("prompt.txt", "checkpoint.json", "metrics.json")
    for name in names:
        (workspace / name).write_text("ordinary project data\n")
        (owner / name).write_text("owned artifact\n")
    prefix = subdir + "/" if subdir else ""

    def entries(sha):
        return set(subprocess.run(["git", "-C", str(repo), "ls-tree", "-rz", "--name-only", sha],
                                 check=True, capture_output=True, text=True).stdout.split("\0"))

    first = snapshot(workspace, 0)  # Unknown ownership must preserve these files.
    assert first
    assert all(prefix + "records/" + name in entries(first) for name in names)
    current = Session(make_config(sandbox_bash=False), MagicMock(), "system", "task",
                      str(workspace), artifact_dir=owner)
    assert all("records/" + name in _snapshot_owned_paths(workspace, current) for name in names)
    index = (repo / ".git/index").read_bytes()
    second = snapshot(workspace, 1, session=current)
    assert second
    assert all(prefix + "records/" + name not in entries(second) for name in names)
    assert all(prefix + name in entries(second) for name in names)
    assert (repo / ".git/index").read_bytes() == index


def test_container_snapshot_preserves_unverified_artifact_entries(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness import turn_snapshots as snapshots
    from scripts.llm_solver.harness import task_environment
    import importlib
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')

    repo = _make_repo(tmp_path)
    owner = repo / "artifacts with 'quotes'"
    owner.mkdir()
    (owner / "metrics.json").write_text("harness data\n")
    (repo / "metrics.json").write_text("project data\n")
    real_run = subprocess.run
    view = tmp_path / 'observed work space'
    view.mkdir()

    def emulate_container(argv, **kwargs):
        assert argv[:6] == [
            "docker", "exec", '-i', "--workdir", str(view), 'inspected-container-id',
        ]
        assert kwargs['binary'] is True
        return real_run([bwrap, '--ro-bind', '/', '/', '--dev', '/dev',
                         '--unshare-net', '--bind', str(repo), str(view),
                         '--chdir', str(view), '--', *argv[6:]],
                        input=kwargs.get('input_bytes'), capture_output=True)

    with monkeypatch.context() as patch:
        patch.setenv("YUJ_CONTAINER", "fixture")
        patch.setattr(task_environment, "discover_task_environment", lambda _: task_environment.TaskEnvironment(
            str(repo), str(view), (str(view),), container='fixture', container_id='inspected-container-id'))
        patch.setattr(execution, '_execute', emulate_container)
        sha = snapshot(repo, 1, session=SimpleNamespace(_artifact_dir=owner))
    assert sha
    store = snapshots.snapshot_object_store(repo, sha)
    assert set(_git(store, "ls-tree", "-r", "--name-only", sha).splitlines()) == {
        "a.py", "metrics.json", "artifacts with 'quotes'/metrics.json",
    }


@pytest.mark.parametrize("layout", ["workspace_alias", "external_alias", "internal_alias", "alias_outside"])
def test_snapshot_resolves_owned_directory_aliases(tmp_path, monkeypatch, layout):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    names = ("prompt.txt", "checkpoint.json", "metrics.json")
    for name in names:
        (repo / name).write_text("project data\n")
    owner = repo / "actual artifacts"
    owner.mkdir()
    for name in names:
        (owner / name).write_text("harness data\n")
    work = repo
    expected = {"a.py", *names}
    if layout == "workspace_alias":
        work = tmp_path / "workspace alias"
        work.symlink_to(repo, target_is_directory=True)
    elif layout == "external_alias":
        alias = tmp_path / "owner alias"
        alias.symlink_to(owner, target_is_directory=True)
        owner = alias
    elif layout == "internal_alias":
        alias = repo / "owner alias"
        alias.symlink_to(owner, target_is_directory=True)
        owner = alias
        expected.add("owner alias")
    else:
        external = tmp_path / "outside artifacts"
        external.mkdir()
        alias = repo / "owner alias"
        alias.symlink_to(external, target_is_directory=True)
        owner = alias
        expected.update(f"actual artifacts/{name}" for name in names)
        expected.add("owner alias")
    before = _git(repo, "status", "--porcelain")
    sha = snapshot(work, 1, session=SimpleNamespace(_artifact_dir=owner))
    assert sha
    assert set(_git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()) == expected
    assert _git(repo, "status", "--porcelain") == before


def test_snapshot_keeps_project_target_of_artifact_file_symlink(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    owner = repo / "artifacts"
    owner.mkdir()
    target = repo / "project-data.json"
    target.write_text("ordinary project data\n")
    (owner / "metrics.json").symlink_to(target)
    sha = snapshot(repo, 1, session=SimpleNamespace(_artifact_dir=owner))
    assert sha
    assert set(_git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()) == {
        "a.py", "project-data.json",
    }
    assert _git(repo, "show", f"{sha}:project-data.json") == "ordinary project data"
