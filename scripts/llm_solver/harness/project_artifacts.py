"""Apply task-supplied ignore rules to candidate check outputs in a private view.

Git sees only a checked index, relevant ignore files and candidate names. It
never reads the task repository configuration or its source files. This is
bookkeeping over observed task bytes, not a second task execution environment.
"""
from decimal import Decimal
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from weakref import WeakKeyDictionary

from .time_budget import execution_deadline, remaining_before
from .operation_timing import measured


_READS = WeakKeyDictionary()


class LocalProjectFiles:
    """Use checked local reads only for explicitly unsandboxed task dispatch."""

    def __init__(self, root):
        self.root = Path(root)

    def read_observation(self, path):
        from .local_file_access import read_observation
        return read_observation(self.root, self.root / path, with_metadata=True)


def _regular(entry):
    return bool(entry and entry[0] == b"f")


def _same_metadata(entry, observed):
    return bool(observed.is_file and _regular(entry)
                and int(entry[1], 8) == observed.mode & 0o7777
                and int(entry[2]) == observed.size
                and int(Decimal(entry[3].decode()) * 1_000_000_000) == observed.mtime_ns
                and int(Decimal(entry[4].decode()) * 1_000_000_000) == observed.ctime_ns
                and int(entry[5]) == observed.inode)


def _read_stable(files, path, entry):
    cache = _READS.setdefault(files, {})
    previous = cache.get(path)
    if previous is not None and previous[0] == entry:
        return previous[1:]
    data, observed = files.read_observation(path)
    if not _same_metadata(entry, observed):
        raise ValueError("project metadata changed after observation")
    digest = hashlib.sha256(data).hexdigest()
    cache[path] = (entry, data, digest)
    return data, digest


def _relative(path):
    candidate = PurePosixPath(path)
    if (not path or "\0" in path or candidate.is_absolute()
            or any(part in {"..", ".git"} for part in candidate.parts)
            or candidate.as_posix() != path):
        raise ValueError("invalid project artifact path")
    return candidate


@measured("changes.project_output_evidence")
def project_artifact_evidence(files, before, after, candidates):
    """Return ignored, untracked candidates using unchanged project metadata.

    Inventories include the regular task-local ``.git/index`` explicitly.
    The caller owns candidate eligibility and retention of observed artifacts.
    Missing or unsupported Git state leaves candidates unclassified.
    """
    result = {"status": "unknown", "matches": {}, "reason": ""}
    try:
        path_set = set(candidates)
        paths = sorted(path_set)
        if not paths:
            return {**result, "status": "available"}
        index_entry = before.get(".git/index")
        if not _regular(index_entry) or index_entry != after.get(".git/index"):
            return {**result, "reason": "index_missing_or_changed"}
        sources = set()
        path_sources = {}
        for name in paths:
            candidate = _relative(name)
            applicable = set()
            for parent in candidate.parents:
                source = (parent / ".gitignore").as_posix()
                old, new = before.get(source), after.get(source)
                if old != new or (old is not None and not _regular(old)):
                    return {**result, "reason": "ignore_rules_changed_or_unsupported"}
                if old is not None:
                    sources.add(source)
                    applicable.add(source)
            path_sources[name] = applicable
        if not sources:
            return {**result, "status": "available", "ignore_sha256": {}}
        index, index_sha = _read_stable(files, ".git/index", index_entry)
        if not index.startswith(b"DIRC"):
            return {**result, "reason": "unsupported_index"}
        ignore_data = {}
        ignore_hashes = {}
        for source in sorted(sources):
            ignore_data[source], ignore_hashes[source] = _read_stable(files, source, after[source])
        git = shutil.which("git")
        if git is None:
            return {**result, "reason": "bookkeeping_git_unavailable"}
        with tempfile.TemporaryDirectory(prefix="yuj-project-artifacts-") as temporary:
            private = Path(temporary)
            worktree, git_dir = private / "view", private / "git"
            worktree.mkdir()
            (git_dir / "objects").mkdir(parents=True)
            (git_dir / "refs").mkdir()
            (git_dir / "HEAD").write_text("ref: refs/heads/observed\n")
            (git_dir / "config").write_text("[core]\nrepositoryformatversion = 0\nbare = false\n")
            (git_dir / "index").write_bytes(index)
            for name in paths:
                (worktree / name).parent.mkdir(parents=True, exist_ok=True)
            for source, data in ignore_data.items():
                destination = worktree / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            completed = subprocess.run(
                [git, f"--git-dir={git_dir}", f"--work-tree={worktree}",
                 "-c", "core.fsmonitor=false", "check-ignore", "-v", "-z", "--stdin"],
                cwd=worktree, input=b"".join(os.fsencode(path) + b"\0" for path in paths),
                capture_output=True,
                env={"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                     "GIT_ATTR_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"},
                timeout=remaining_before(execution_deadline()),
            )
        if completed.returncode not in {0, 1}:
            return {**result, "reason": "unsupported_git_metadata"}
        fields = completed.stdout.split(b"\0")
        if fields.pop() != b"" or len(fields) % 4:
            raise ValueError("invalid ignore evidence")
        matches = {}
        for offset in range(0, len(fields), 4):
            source, line, pattern, path = (os.fsdecode(value) for value in fields[offset:offset + 4])
            if source not in ignore_hashes or path not in path_set or not line.isdigit():
                raise ValueError("unexpected ignore evidence")
            if not pattern.startswith("!"):
                matches[path] = {"source": source, "line": int(line), "pattern": pattern,
                                 "index_sha256": index_sha,
                                 "ignore_sha256": {name: ignore_hashes[name]
                                                   for name in sorted(path_sources[path])}}
        return {"status": "available", "matches": matches, "reason": "",
                "index_sha256": index_sha, "ignore_sha256": ignore_hashes}
    except (OSError, ValueError, RuntimeError, ArithmeticError, subprocess.SubprocessError) as error:
        return {**result, "reason": type(error).__name__}
