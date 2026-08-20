"""Shared bash write/mutation classifiers.

The harness has two related but distinct bash classifications:

- action metadata records writes that are visible from the command body
  itself, and extracts source-looking paths for state projection.
- guardrails and historical state replay keep a legacy-permissive
  mutation heuristic so older traces and gate behavior remain stable.

Keeping both policies here prevents regex drift while making the distinction
explicit at call sites.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


SOURCE_FILE_EXTENSIONS = (
    # Python / docs / project text already covered by the original policy.
    "py", "pyi", "rst", "txt", "md", "toml", "cfg", "ini", "yaml", "yml",
    # Common implementation languages in non-Python benchmarks.
    "rs", "go", "js", "jsx", "ts", "tsx", "mjs", "cjs",
    "java", "kt", "kts", "scala", "groovy",
    "c", "h", "cc", "hh", "cpp", "hpp", "cxx", "hxx",
    "cs", "php", "rb", "swift", "m", "mm", "r", "jl", "lua",
    # Shell/build/template/schema files often used as task source.
    "sh", "bash", "zsh", "fish", "sql", "html", "htm", "css", "scss",
    "sass", "less", "vue", "svelte", "gradle", "sbt", "bazel", "bzl",
    "cmake", "proto", "graphql", "gql",
)

_SOURCE_EXT_RE = "|".join(re.escape(ext) for ext in SOURCE_FILE_EXTENSIONS)


BASH_ACTION_SHELL_WRITE_RE = re.compile(
    r"(?:^|[;&|\n]\s*)(?:"
    r"apply_patch\b|patch\s+-p\d*\b|sed\s+-i\b|perl\s+-0?pi\b|"
    r"cat\s+>|tee\s+-a?\b"
    r")",
    re.DOTALL,
)
BASH_ACTION_REDIRECT_SOURCE_RE = re.compile(
    rf"(?:^|[\s;&|])(?:\d?>|>>)\s*"
    rf"(?:/testbed/)?[A-Za-z0-9_./+-]+\.({_SOURCE_EXT_RE})\b",
    re.DOTALL,
)
BASH_ACTION_PYTHON_WRITE_RE = re.compile(
    r"\bpython3?\b[\s\S]*(?:"
    r"write_(?:text|bytes)\s*\(|os\.replace\s*\(|shutil\.(?:move|copyfile)\s*\(|"
    r"open\s*\([^)]*,\s*['\"][wax]['\"]|fileinput\b"
    r")",
    re.DOTALL,
)
BASH_LEGACY_MUTATION_RE = re.compile(
    r"(?:^|[;&|\n]\s*)(?:"
    r"apply_patch\b|patch\s+-p\d*\b|sed\s+-i\b|perl\s+-pi\b|"
    r"python3?\s+-\s*<<|cat\s+>|tee\s+-a?\b"
    r")"
)
BASH_LEGACY_PYTHON_WRITE_RE = re.compile(
    r"\bpython3?\s+(?:-c|-|<<).*(?:write_(?:text|bytes)\s*\(|"
    r"open\s*\([^)]*['\"]w['\"]|fileinput\b)",
    re.DOTALL,
)
FILE_TOKEN_RE = re.compile(
    r"(?<![\w/.-])(?:/testbed/)?[A-Za-z0-9_./+-]+\."
    rf"(?:{_SOURCE_EXT_RE})\b"
)
PY_PATH_RE = re.compile(r"(?:open|Path)\(\s*['\"]([^'\"]+)['\"]")
NON_SOURCE_HINT_RE = re.compile(
    r"(^|/)(?:tests?|__tests__|spec)(?:/|$)|"
    r"(^|/)(?:test_[^/]*|[^/]*(?:_test|\.test|\.spec)\.[^/]+|conftest\.py)(?:/|$)|"
    r"(^|/)(?:tox\.ini|pyproject\.toml|setup\.cfg|pytest\.ini|"
    r"package\.json|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|"
    r"Cargo\.toml|Cargo\.lock|go\.mod|go\.sum|CMakeLists\.txt)$"
)

STATE_WRITER_MUTATION_PREFIXES = (
    "edit(",
    "write(",
    "str_replace(",
    "create(",
    "apply_patch(",
)


@dataclass(frozen=True)
class BashWriteClassification:
    """Mechanical classification of one bash command."""

    action_write_like: bool
    legacy_mutation_like: bool
    source_write_paths: tuple[str, ...]

    @property
    def source_write_like(self) -> bool:
        return self.action_write_like and bool(self.source_write_paths)


def normalize_trace_path(path: str) -> str:
    """Normalize path tokens emitted in model tool arguments."""
    path = path.strip().strip("'\"")
    if path.startswith("/testbed/"):
        path = path[len("/testbed/"):]
    if path.startswith("./"):
        path = path[2:]
    return path


def is_source_path(path: str) -> bool:
    """Return True for source-looking paths, excluding test/config hints."""
    if not path:
        return False
    if "://" in path:
        return False
    return NON_SOURCE_HINT_RE.search(path) is None


def extract_source_write_paths(text: str) -> tuple[str, ...]:
    """Extract ordered source-looking paths from action text."""
    paths: list[str] = []
    seen: set[str] = set()
    for match in FILE_TOKEN_RE.finditer(text or ""):
        path = normalize_trace_path(match.group(0))
        if is_source_path(path) and path not in seen:
            seen.add(path)
            paths.append(path)
    for match in PY_PATH_RE.finditer(text or ""):
        path = normalize_trace_path(match.group(1))
        if is_source_path(path) and path not in seen:
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def is_bash_action_write_like(cmd: str) -> bool:
    """True when a bash command body visibly performs a write."""
    return bool(
        cmd
        and (
            BASH_ACTION_SHELL_WRITE_RE.search(cmd)
            or BASH_ACTION_REDIRECT_SOURCE_RE.search(cmd)
            or BASH_ACTION_PYTHON_WRITE_RE.search(cmd)
        )
    )


def is_bash_legacy_mutation_like(cmd: str) -> bool:
    """True for the legacy guardrail/state-replay bash mutation heuristic."""
    return bool(
        cmd
        and (
            BASH_LEGACY_MUTATION_RE.search(cmd)
            or BASH_LEGACY_PYTHON_WRITE_RE.search(cmd)
        )
    )


def classify_bash_write(cmd: str) -> BashWriteClassification:
    """Return both bash write policies plus source path metadata."""
    action_write_like = is_bash_action_write_like(cmd)
    return BashWriteClassification(
        action_write_like=action_write_like,
        legacy_mutation_like=is_bash_legacy_mutation_like(cmd),
        source_write_paths=(
            extract_source_write_paths(cmd) if action_write_like else ()
        ),
    )


__all__ = [
    "BASH_ACTION_PYTHON_WRITE_RE",
    "BASH_ACTION_REDIRECT_SOURCE_RE",
    "BASH_ACTION_SHELL_WRITE_RE",
    "BASH_LEGACY_MUTATION_RE",
    "BASH_LEGACY_PYTHON_WRITE_RE",
    "BashWriteClassification",
    "SOURCE_FILE_EXTENSIONS",
    "STATE_WRITER_MUTATION_PREFIXES",
    "classify_bash_write",
    "extract_source_write_paths",
    "is_bash_action_write_like",
    "is_bash_legacy_mutation_like",
    "is_source_path",
    "normalize_trace_path",
]
