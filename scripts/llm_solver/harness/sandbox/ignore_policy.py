"""Repository ignore-file policy shared by read tools and sandboxes.

The public configuration names one or more ignore files at the task root.  A
policy loads those files once, compiles their gitignore-style rules, and then
serves the same visibility decision to every consumer.  No model-facing tool
should parse an ignore file independently.

Supported rule syntax:

* blank lines and unescaped leading ``#`` comments;
* ``*``, ``?``, ``[]`` and ``**`` wildcards;
* a leading ``/`` to anchor a rule at the task root;
* a trailing ``/`` to match directories (and their descendants) only; and
* a leading ``!`` to negate an earlier rule.

Within one file, the last matching rule wins.  Across configured ignore-file
names, the first file that has a matching rule wins.  That makes the declared
file-name order an explicit precedence boundary rather than concatenating
files and accidentally reversing it.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re


DEFAULT_IGNORE_FILE_NAMES: tuple[str, ...] = (".yujignore",)
MAX_IGNORE_FILE_BYTES = 1024 * 1024
MAX_IGNORE_RULES = 50_000


class IgnorePolicyError(ValueError):
    """Raised when an ignore source is malformed or escapes the task root."""


class IgnoredPathError(FileNotFoundError):
    """A hidden path, shaped as FileNotFoundError for read-tool invisibility."""


def _strip_unescaped_trailing_spaces(line: str) -> str:
    r"""Apply gitignore's trailing-space rule while preserving ``\ ``."""
    characters = list(line)
    while characters and characters[-1] == " ":
        backslashes = 0
        index = len(characters) - 2
        while index >= 0 and characters[index] == "\\":
            backslashes += 1
            index -= 1
        if backslashes % 2:
            # Remove the escape immediately before this final literal space.
            del characters[-2]
            break
        characters.pop()
    return "".join(characters)


def _character_class_regex(content: str) -> str:
    """Translate one glob character class without allowing path separators."""
    if not content:
        return r"\[\]"
    negate = content[0] in ("!", "^")
    if negate:
        content = content[1:]
    if not content:
        return r"\[\]"
    rendered: list[str] = []
    for index, char in enumerate(content):
        if char == "\\":
            rendered.append(r"\\")
        elif char == "]":
            rendered.append(r"\]")
        elif char == "^" and index == 0:
            rendered.append(r"\^")
        elif char == "-" and index not in (0, len(content) - 1):
            rendered.append("-")
        else:
            rendered.append(re.escape(char))
    prefix = "^" if negate else ""
    # A range such as [.-0] contains '/', so guard the class as a whole.
    return rf"(?!/)[{prefix}{''.join(rendered)}]"


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a gitignore glob against one POSIX relative-path string."""
    translated: list[str] = ["^"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            translated.append(re.escape(pattern[index + 1]))
            index += 2
            continue
        if char == "*":
            star_end = index + 1
            while star_end < len(pattern) and pattern[star_end] == "*":
                star_end += 1
            if star_end - index >= 2:
                if star_end < len(pattern) and pattern[star_end] == "/":
                    translated.append(r"(?:[^/]+/)*")
                    index = star_end + 1
                else:
                    translated.append(r".*")
                    index = star_end
            else:
                translated.append(r"[^/]*")
                index += 1
            continue
        if char == "?":
            translated.append(r"[^/]")
            index += 1
            continue
        if char == "[":
            end = index + 1
            if end < len(pattern) and pattern[end] in ("!", "^"):
                end += 1
            if end < len(pattern) and pattern[end] == "]":
                end += 1
            while end < len(pattern) and pattern[end] != "]":
                end += 1
            if end >= len(pattern):
                translated.append(r"\[")
                index += 1
            else:
                translated.append(_character_class_regex(pattern[index + 1:end]))
                index = end + 1
            continue
        translated.append(re.escape(char))
        index += 1
    translated.append("$")
    try:
        return re.compile("".join(translated))
    except re.error as exc:
        raise IgnorePolicyError(
            f"invalid ignore glob {pattern!r}: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class IgnoreRule:
    """One compiled ignore-file rule."""

    pattern: str
    negated: bool
    anchored: bool
    directory_only: bool
    source_name: str
    line_number: int
    _regex: re.Pattern[str]
    _basename_only: bool

    def _matches_candidate(self, candidate: str, *, is_dir: bool) -> bool:
        if self.directory_only and not is_dir:
            return False
        target = PurePosixPath(candidate).name if self._basename_only else candidate
        return self._regex.fullmatch(target) is not None

    def matches(self, relative_path: str, *, is_dir: bool = False) -> bool:
        """Return whether this rule applies to a path or one of its parents."""
        relative_path = relative_path.strip("/")
        if not relative_path or relative_path == ".":
            return False
        if self._matches_candidate(relative_path, is_dir=is_dir):
            return True
        parent = PurePosixPath(relative_path).parent
        while str(parent) not in ("", "."):
            if self._matches_candidate(parent.as_posix(), is_dir=True):
                return True
            parent = parent.parent
        return False


def _parse_rule(line: str, *, source_name: str, line_number: int) -> IgnoreRule | None:
    line = line.rstrip("\r")
    line = _strip_unescaped_trailing_spaces(line)
    if not line:
        return None
    if line.startswith("#"):
        return None
    if line.startswith(r"\#"):
        line = line[1:]

    negated = False
    if line.startswith("!"):
        negated = True
        line = line[1:]
    elif line.startswith(r"\!"):
        line = line[1:]
    if not line:
        raise IgnorePolicyError(
            f"{source_name}:{line_number}: empty ignore pattern"
        )
    if "\x00" in line:
        raise IgnorePolicyError(
            f"{source_name}:{line_number}: ignore pattern contains NUL"
        )

    anchored = line.startswith("/")
    if anchored:
        line = line[1:]
    directory_only = line.endswith("/") and not line.endswith(r"\/")
    if directory_only:
        line = line[:-1]
    if not line:
        return None

    path_segments = PurePosixPath(line).parts
    if ".." in path_segments:
        raise IgnorePolicyError(
            f"{source_name}:{line_number}: '..' is not allowed in ignore patterns"
        )
    basename_only = not anchored and "/" not in line
    return IgnoreRule(
        pattern=line,
        negated=negated,
        anchored=anchored,
        directory_only=directory_only,
        source_name=source_name,
        line_number=line_number,
        _regex=_glob_regex(line),
        _basename_only=basename_only,
    )


def parse_ignore_lines(
    lines: Iterable[str], *, source_name: str = ".yujignore",
) -> tuple[IgnoreRule, ...]:
    """Parse ignore-file lines and return immutable compiled rules."""
    rules: list[IgnoreRule] = []
    for line_number, line in enumerate(lines, start=1):
        if len(rules) >= MAX_IGNORE_RULES:
            raise IgnorePolicyError(
                f"{source_name}: exceeds maximum of {MAX_IGNORE_RULES} rules"
            )
        rule = _parse_rule(
            line, source_name=source_name, line_number=line_number,
        )
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


@dataclass(frozen=True, slots=True)
class IgnoreSource:
    """One loaded root ignore file and its secret-free provenance."""

    name: str
    sha256: str
    size_bytes: int
    rules: tuple[IgnoreRule, ...]

    def decision(self, relative_path: str, *, is_dir: bool) -> bool | None:
        decision: bool | None = None
        for rule in self.rules:
            if rule.matches(relative_path, is_dir=is_dir):
                decision = not rule.negated
        return decision


def _validate_file_names(file_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(file_names, (str, bytes)):
        raise IgnorePolicyError("state.ignore_file_names must be a list of names")
    validated: list[str] = []
    seen: set[str] = set()
    for name in file_names:
        if not isinstance(name, str) or not name or "\x00" in name:
            raise IgnorePolicyError(
                "state.ignore_file_names entries must be non-empty strings"
            )
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or name.endswith("/"):
            raise IgnorePolicyError(
                f"ignore file name {name!r} must stay within the task root"
            )
        normalized = posix.as_posix()
        if normalized not in seen:
            seen.add(normalized)
            validated.append(normalized)
    return tuple(validated)


def _aggregate_hash(sources: tuple[IgnoreSource, ...]) -> str | None:
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0].sha256
    digest = hashlib.sha256(b"yuj-ignore-policy-v1\x00")
    for source in sources:
        name = source.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(source.sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IgnorePolicy:
    """Loaded ignore sources for one immutable task-root view."""

    root: Path
    sources: tuple[IgnoreSource, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        if not root.is_dir():
            raise IgnorePolicyError(f"task root is not a directory: {root}")
        object.__setattr__(self, "root", root)

    @property
    def aggregate_hash(self) -> str | None:
        """Hash recorded in ``session_start`` (raw SHA-256 for one file)."""
        return _aggregate_hash(self.sources) if self.enabled else None

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(source.name for source in self.sources) if self.enabled else ()

    def trace_fields(self) -> dict[str, object]:
        """Return provenance fields containing hashes/names, never rule text."""
        return {
            "ignore_file_hash": self.aggregate_hash,
            "ignore_file_names": list(self.source_names),
        }

    def _relative(self, path: str | os.PathLike[str]) -> tuple[str, Path]:
        raw = os.fspath(path)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise IgnorePolicyError("path must be a non-empty NUL-free string")
        target = Path(raw)
        if not target.is_absolute():
            target = self.root / target
        # abspath removes '..' without following symlinks.  Matching the
        # lexical repository entry is important when that entry is a symlink.
        target = Path(os.path.abspath(target))
        try:
            relative = target.relative_to(self.root).as_posix()
        except ValueError:
            raise IgnorePolicyError(f"path escapes task root: {raw}") from None
        return relative, target

    def decision(
        self,
        path: str | os.PathLike[str],
        *,
        is_dir: bool | None = None,
    ) -> bool | None:
        """Return True/False for a matched path, or None when no rule matches."""
        if not self.enabled:
            return None
        relative, target = self._relative(path)
        directory = target.is_dir() if is_dir is None else bool(is_dir)
        for source in self.sources:
            decision = source.decision(relative, is_dir=directory)
            if decision is not None:
                return decision
        return None

    def is_ignored(
        self,
        path: str | os.PathLike[str],
        *,
        is_dir: bool | None = None,
    ) -> bool:
        return self.decision(path, is_dir=is_dir) is True

    def require_visible(
        self,
        path: str | os.PathLike[str],
        *,
        is_dir: bool | None = None,
    ) -> None:
        """Raise FileNotFoundError-compatible error when a path is hidden."""
        if self.is_ignored(path, is_dir=is_dir):
            raise IgnoredPathError(os.fspath(path))

    def filter_paths(
        self, paths: Iterable[str | os.PathLike[str]],
    ) -> tuple[str | os.PathLike[str], ...]:
        """Preserve input order while removing ignored entries."""
        return tuple(path for path in paths if not self.is_ignored(path))

    def existing_ignored_paths(self) -> tuple[str, ...]:
        """Return sorted absolute existing entries for sandbox mask expansion.

        The walk never follows directory symlinks.  Descendants are still
        visited beneath an ignored directory because a later negation may make
        one of them visible.
        """
        if not self.enabled or not self.sources:
            return ()
        hidden: list[str] = []
        for directory, dir_names, file_names in os.walk(
            self.root, topdown=True, followlinks=False,
        ):
            dir_names.sort()
            file_names.sort()
            base = Path(directory)
            for name in dir_names:
                entry = base / name
                if self.is_ignored(entry, is_dir=True):
                    hidden.append(str(entry))
            for name in file_names:
                entry = base / name
                if self.is_ignored(entry, is_dir=False):
                    hidden.append(str(entry))
        return tuple(sorted(hidden))


def load_ignore_policy(
    root: str | os.PathLike[str],
    *,
    enabled: bool = True,
    file_names: Sequence[str] = DEFAULT_IGNORE_FILE_NAMES,
) -> IgnorePolicy:
    """Load configured ignore files from ``root`` once, with stable hashes."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise IgnorePolicyError(f"task root is not a directory: {root_path}")
    if not isinstance(enabled, bool):
        raise IgnorePolicyError("state.ignore_file_enabled must be a boolean")
    names = _validate_file_names(file_names)
    if not enabled:
        return IgnorePolicy(root_path, (), enabled=False)

    sources: list[IgnoreSource] = []
    for name in names:
        candidate = root_path / name
        if not candidate.exists() and not candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError:
            raise IgnorePolicyError(
                f"ignore file {name!r} resolves outside the task root"
            ) from None
        if not resolved.is_file():
            raise IgnorePolicyError(f"ignore file {name!r} is not a regular file")
        raw = resolved.read_bytes()
        if len(raw) > MAX_IGNORE_FILE_BYTES:
            raise IgnorePolicyError(
                f"ignore file {name!r} exceeds {MAX_IGNORE_FILE_BYTES} bytes"
            )
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IgnorePolicyError(
                f"ignore file {name!r} is not valid UTF-8"
            ) from exc
        sources.append(IgnoreSource(
            name=name,
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
            rules=parse_ignore_lines(text.splitlines(), source_name=name),
        ))
    return IgnorePolicy(root_path, tuple(sources), enabled=True)
