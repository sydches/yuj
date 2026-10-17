"""Project output evidence comes from stable task bytes, never task Git config."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.llm_solver.harness.project_artifacts import project_artifact_evidence
from scripts.llm_solver.harness.task_files import FileMetadata
from tests.test_task_files import bwrap, namespace_files


def _git(root, *arguments):
    if shutil.which("git") is None:
        pytest.skip("Git unavailable")
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=True,
                          env={"PATH": os.environ.get("PATH", ""), "GIT_CONFIG_NOSYSTEM": "1",
                               "GIT_CONFIG_GLOBAL": os.devnull})


def _entry(path):
    info = path.lstat()
    def timestamp(value):
        return f"{value // 1_000_000_000}.{value % 1_000_000_000:09d}0".encode()
    return (b"l" if path.is_symlink() else b"f", oct(info.st_mode & 0o7777)[2:].encode(),
            str(info.st_size).encode(), timestamp(info.st_mtime_ns), timestamp(info.st_ctime_ns),
            str(info.st_ino).encode(), b"")


def _inventory(root):
    return {path.relative_to(root).as_posix(): _entry(path)
            for path in [root / ".git/index", *root.rglob(".gitignore")]
            if path.exists() or path.is_symlink()}


class ObservedFiles:
    def __init__(self, root):
        self.root = root
        self.reads = []

    def read_observation(self, path):
        self.reads.append(path)
        target = self.root / path
        data, info = target.read_bytes(), target.stat()
        return data, FileMetadata(info.st_mode, info.st_size, int(info.st_mtime), int(info.st_ctime),
                                  info.st_ino, info.st_dev, info.st_mtime_ns, info.st_ctime_ns)


@pytest.fixture
def project(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("build/\n*.png\n!keep.png\n")
    (tmp_path / "tracked.png").write_text("tracked input")
    _git(tmp_path, "add", "-f", ".gitignore", "tracked.png")
    return tmp_path, ObservedFiles(tmp_path)


def test_project_rules_are_batched_and_tracked_missing_files_are_protected(project):
    root, files = project
    (root / "tracked.png").unlink()
    observed = _inventory(root)
    result = project_artifact_evidence(files, observed, observed,
                                      ["build/out.js", "new.png", "keep.png", "tracked.png", "source.py"])
    assert result["status"] == "available"
    assert set(result["matches"]) == {"build/out.js", "new.png"}
    assert result["matches"]["new.png"]["source"] == ".gitignore"
    assert result["matches"]["new.png"]["pattern"] == "*.png"
    assert len(result["index_sha256"]) == 64
    assert set(result["ignore_sha256"]) == {".gitignore"}
    assert files.reads == [".git/index", ".gitignore"]
    assert project_artifact_evidence(files, observed, observed, ["new.png"])["matches"]
    assert files.reads == [".git/index", ".gitignore"]


def test_only_applicable_nested_rules_are_read(project):
    root, files = project
    (root / "nested").mkdir()
    (root / "nested/.gitignore").write_text("*.dat\n!keep.dat\n")
    (root / "unrelated").mkdir()
    (root / "unrelated/.gitignore").write_text("secret\n")
    observed = _inventory(root)
    result = project_artifact_evidence(files, observed, observed,
                                      ["nested/output.dat", "nested/keep.dat", "other.dat"])
    assert set(result["matches"]) == {"nested/output.dat"}
    assert result["matches"]["nested/output.dat"]["source"] == "nested/.gitignore"
    assert files.reads == [".git/index", ".gitignore", "nested/.gitignore"]
    assert project_artifact_evidence(files, observed, observed, ["nested/output.dat"])["matches"] == result["matches"]


def test_unrelated_batch_rules_do_not_change_an_artifacts_receipt(project):
    root, files = project
    (root / "nested").mkdir()
    (root / "nested/.gitignore").write_text("*.dat\n")
    observed = _inventory(root)
    first = project_artifact_evidence(files, observed, observed, ["new.png"])
    other = project_artifact_evidence(files, observed, observed, ["new.png", "nested/new.dat"])
    assert first["matches"]["new.png"] == other["matches"]["new.png"]


def test_repeated_calls_use_fresh_nested_ignore_rules(project):
    root, files = project
    (root / "nested").mkdir()
    ignore = root / "nested/.gitignore"
    ignore.write_text("*.dat\n")
    first_inventory = _inventory(root)
    candidates = ["nested/first.dat", "nested/second.dat", "nested/new.txt"]
    first = project_artifact_evidence(files, first_inventory, first_inventory, candidates)
    assert set(first["matches"]) == {"nested/first.dat", "nested/second.dat"}

    ignore.write_text("*.txt\n# changed rules\n")
    second_inventory = _inventory(root)
    changed = project_artifact_evidence(files, first_inventory, second_inventory, candidates)
    assert changed["status"] == "unknown"
    assert changed["matches"] == {}
    second = project_artifact_evidence(files, second_inventory, second_inventory, candidates)
    assert set(second["matches"]) == {"nested/new.txt"}
    assert second["ignore_sha256"]["nested/.gitignore"] != first["ignore_sha256"]["nested/.gitignore"]
    assert files.reads.count("nested/.gitignore") == 2


@pytest.mark.parametrize("pattern", ["a/b/", "a/**/b/"])
def test_candidate_parent_directories_keep_directory_only_rule_semantics(project, pattern):
    root, files = project
    (root / ".gitignore").write_text(pattern + "\n")
    observed = _inventory(root)
    assert project_artifact_evidence(files, observed, observed, ["a/b"])["matches"] == {}
    candidates = ["a/b", "a/b/first", "a/b/second"]
    result = project_artifact_evidence(files, observed, observed, candidates)
    assert set(result["matches"]) == set(candidates)
    assert {receipt["pattern"] for receipt in result["matches"].values()} == {pattern}


def test_no_project_ignore_rules_needs_no_metadata_reads_or_git(project, monkeypatch):
    root, files = project
    (root / ".gitignore").unlink()
    observed = _inventory(root)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no Git call is needed"))
    assert project_artifact_evidence(files, observed, observed, ["new.png"])["matches"] == {}
    assert not files.reads


@pytest.mark.parametrize("changed", [".gitignore", "nested/.gitignore", ".git/index"])
def test_changed_or_new_metadata_does_not_authorize_artifacts(project, changed):
    root, files = project
    before = _inventory(root)
    target = root / changed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(target.read_bytes() + b"# changed\n" if target.exists() else b"*.png\n")
    result = project_artifact_evidence(files, before, _inventory(root), ["nested/new.png"])
    assert result["status"] == "unknown"
    assert result["matches"] == {}
    assert not files.reads


def test_metadata_change_after_inventory_is_detected(project):
    root, files = project
    observed = _inventory(root)
    (root / ".gitignore").write_text("different declaration\n")
    result = project_artifact_evidence(files, observed, observed, ["new.png"])
    assert result["status"] == "unknown"
    assert result["matches"] == {}


def test_task_config_global_config_and_info_excludes_are_never_used(project, monkeypatch, tmp_path):
    root, files = project
    marker = root / "executed"
    (root / ".git/config").write_text(
        f"[include]\npath = {root / 'outside-config'}\n"
        f"[core]\nfsmonitor = touch {marker}\n"
    )
    (root / ".git/info/exclude").write_text("private.txt\n")
    global_config = root / "global-config"
    global_config.write_text(f"[core]\nexcludesFile = {root / 'global-ignore'}\n")
    (root / "global-ignore").write_text("global.txt\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.excludesFile")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(root / "global-ignore"))
    observed = _inventory(root)
    result = project_artifact_evidence(files, observed, observed, ["new.png", "private.txt", "global.txt"])
    assert set(result["matches"]) == {"new.png"}
    assert not marker.exists()
    assert files.reads == [".git/index", ".gitignore"]


@pytest.mark.parametrize("case", ["missing", "corrupt", "symlink_ignore", "symlink_index"])
def test_unsupported_project_metadata_stays_unknown(project, case):
    root, files = project
    if case == "missing":
        (root / ".git/index").unlink()
    elif case == "corrupt":
        (root / ".git/index").write_bytes(b"DIRCcorrupt")
    else:
        path = root / (".gitignore" if case == "symlink_ignore" else ".git/index")
        data = path.read_bytes()
        path.unlink()
        (root / "other").write_bytes(data)
        path.symlink_to(root / "other")
    observed = _inventory(root)
    result = project_artifact_evidence(files, observed, observed, ["new.png"])
    assert result["status"] == "unknown"
    assert result["matches"] == {}


@pytest.mark.parametrize("candidate", ["../outside.png", "/outside.png", ".git/config", "a/../out.png"])
def test_candidate_names_cannot_escape_the_private_view(project, candidate):
    root, files = project
    observed = _inventory(root)
    assert project_artifact_evidence(files, observed, observed, [candidate])["status"] == "unknown"
    assert not files.reads


def test_native_task_rules_do_not_fall_back_to_the_host_shadow(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    _git(source, "init", "-q")
    (source / ".gitignore").write_text("native.png\n")
    _git(source, "add", ".gitignore")
    (tmp_path / "view/.gitignore").write_text("host.png\n")
    observed = _inventory(source)
    result = project_artifact_evidence(files, observed, observed, ["native.png", "host.png"])
    assert result["status"] == "available"
    assert set(result["matches"]) == {"native.png"}


def test_literal_candidate_names_are_nul_delimited(project):
    root, files = project
    observed = _inventory(root)
    names = ["a space.png", "a\nnewline.png", "-leading.png"]
    result = project_artifact_evidence(files, observed, observed, names)
    assert set(result["matches"]) == set(names)
