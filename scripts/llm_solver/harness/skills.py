"""Agent Skills discovery, validation, and prompt catalog rendering.

Discovery reads only YAML frontmatter from ``SKILL.md``.  The Markdown body
and bundled resources remain on disk until the model uses its ordinary
``read`` or sandboxed ``bash`` tool.
"""
from __future__ import annotations

import html
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import yaml

from .project_instructions import _UnreadableMatcher, find_project_root


log = logging.getLogger(__name__)

DEFAULT_SKILLS_DIRS = (
    "~/.pi/agent/skills",
    "~/.agents/skills",
    ".pi/skills",
    ".agents/skills",
)
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_LIMIT_BYTES = 64 * 1024
_DISCOVERY_MAX_DEPTH = 6
_DISCOVERY_MAX_DIRS = 2000
_SKIP_DIR_NAMES = frozenset({".git", "node_modules", "__pycache__"})


class SkillError(ValueError):
    """A configured or discovered skill is not safe to load."""


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


@dataclass(frozen=True, slots=True)
class Skill:
    """Validated startup metadata for one Agent Skill."""

    name: str
    description: str
    path: Path
    directory: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: Mapping[str, str] | None = None
    allowed_tools: str | None = None
    disable_model_invocation: bool = False

    def trace_record(self) -> dict[str, object]:
        """Return the raw ``session_start`` provenance for this skill."""
        return {
            "name": self.name,
            "path": str(self.path),
            "disable_model_invocation": self.disable_model_invocation,
        }


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    """The immutable, first-wins skill set for one task run."""

    skills: tuple[Skill, ...] = ()

    @property
    def readable_dirs(self) -> tuple[str, ...]:
        """Canonical skill directories that tools may expose for reading."""
        return tuple(dict.fromkeys(str(skill.directory) for skill in self.skills))

    def trace_records(self) -> list[dict[str, object]]:
        return [skill.trace_record() for skill in self.skills]

    def format_prompt_block(self) -> str:
        """Render only model-invocable metadata, never a skill body."""
        visible = tuple(
            skill for skill in self.skills
            if not skill.disable_model_invocation
        )
        if not visible:
            return ""
        lines = [
            "<skills>",
            "The following skills provide task-specific instructions. When a "
            "task matches a description, use read on the listed SKILL.md before "
            "proceeding. Resolve referenced paths from that skill directory and "
            "use absolute paths in tool calls.",
        ]
        for skill in visible:
            description = " ".join(skill.description.split())
            lines.append(
                f"{skill.name}: {html.escape(description, quote=False)} "
                f"({html.escape(str(skill.path), quote=False)})"
            )
        lines.append("</skills>")
        return "\n".join(lines)


def validate_skill_settings(
    enabled: object,
    skills_dirs: Sequence[str],
    skill_paths: Sequence[str],
) -> None:
    """Validate public settings without touching the filesystem."""
    if not isinstance(enabled, bool):
        raise ValueError("skills_enabled must be a boolean")
    for field, values in (
        ("skills_dirs", skills_dirs),
        ("skill_paths", skill_paths),
    ):
        for value in values:
            if not value.strip() or _has_control_characters(value):
                raise ValueError(
                    f"{field} entries must be non-empty paths without control "
                    "characters"
                )


def _frontmatter_bytes(path: Path) -> bytes:
    """Read only the leading YAML envelope, stopping before the body."""
    try:
        with path.open("rb") as stream:
            first = stream.readline(_FRONTMATTER_LIMIT_BYTES + 1)
            if first.rstrip(b"\r\n") != b"---":
                raise SkillError(
                    f"{path}: SKILL.md must start with YAML frontmatter '---'"
                )
            chunks: list[bytes] = []
            size = 0
            while True:
                line = stream.readline(_FRONTMATTER_LIMIT_BYTES + 1)
                if not line:
                    raise SkillError(
                        f"{path}: SKILL.md frontmatter has no closing '---'"
                    )
                if line.rstrip(b"\r\n") == b"---":
                    return b"".join(chunks)
                size += len(line)
                if size > _FRONTMATTER_LIMIT_BYTES:
                    raise SkillError(
                        f"{path}: SKILL.md frontmatter exceeds "
                        f"{_FRONTMATTER_LIMIT_BYTES} bytes"
                    )
                chunks.append(line)
    except SkillError:
        raise
    except OSError as exc:
        raise SkillError(f"{path}: cannot read SKILL.md ({type(exc).__name__})") from exc


def _optional_string(
    frontmatter: Mapping[str, object], field: str, *, max_chars: int | None = None,
) -> str | None:
    value = frontmatter.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SkillError(f"frontmatter field {field!r} must be a non-empty string")
    if max_chars is not None and len(value) > max_chars:
        raise SkillError(
            f"frontmatter field {field!r} exceeds {max_chars} characters"
        )
    return value


def load_skill(path: str | Path) -> Skill:
    """Parse and strictly validate one ``SKILL.md`` metadata envelope."""
    try:
        skill_path = Path(path).resolve(strict=True)
    except OSError as exc:
        raise SkillError(
            f"{path}: cannot resolve SKILL.md ({type(exc).__name__})"
        ) from exc
    if _has_control_characters(str(skill_path)):
        raise SkillError(
            f"{str(skill_path)!r}: skill path contains control characters"
        )
    if not skill_path.is_file() or skill_path.name != "SKILL.md":
        raise SkillError(f"{skill_path}: skill path must name a SKILL.md file")
    try:
        source = _frontmatter_bytes(skill_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError(f"{skill_path}: frontmatter must be UTF-8") from exc
    try:
        value = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise SkillError(f"{skill_path}: invalid YAML frontmatter") from exc
    if not isinstance(value, dict):
        raise SkillError(f"{skill_path}: frontmatter must be a YAML mapping")
    frontmatter: Mapping[str, object] = value

    name = frontmatter.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or len(name) > 64:
        raise SkillError(
            f"{skill_path}: name must be 1-64 lowercase letters, numbers, or "
            "single hyphens"
        )
    if name != skill_path.parent.name:
        raise SkillError(
            f"{skill_path}: name {name!r} must match parent directory "
            f"{skill_path.parent.name!r}"
        )

    description = frontmatter.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > 1024
    ):
        raise SkillError(
            f"{skill_path}: description must be a non-empty string of at most "
            "1024 characters"
        )

    license_name = _optional_string(frontmatter, "license")
    compatibility = _optional_string(
        frontmatter, "compatibility", max_chars=500,
    )
    allowed_tools = _optional_string(frontmatter, "allowed-tools")
    metadata_value = frontmatter.get("metadata")
    metadata: dict[str, str] | None = None
    if metadata_value is not None:
        if not isinstance(metadata_value, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in metadata_value.items()
        ):
            raise SkillError(
                f"{skill_path}: metadata must map string keys to string values"
            )
        metadata = dict(metadata_value)
    disabled = frontmatter.get("disable-model-invocation", False)
    if not isinstance(disabled, bool):
        raise SkillError(
            f"{skill_path}: disable-model-invocation must be a boolean"
        )

    return Skill(
        name=name,
        description=description,
        path=skill_path,
        directory=skill_path.parent,
        license=license_name,
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=allowed_tools,
        disable_model_invocation=disabled,
    )


def _expand_path(value: str, *, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve(strict=False)


def _project_search_dirs(cwd: Path, project_root: Path) -> tuple[Path, ...]:
    try:
        cwd.relative_to(project_root)
    except ValueError as exc:  # pragma: no cover - find_project_root guarantees it
        raise SkillError(f"task cwd {cwd} is outside project root {project_root}") from exc
    output = []
    current = cwd
    while True:
        output.append(current)
        if current == project_root:
            return tuple(output)
        current = current.parent


def _walk_skill_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    if not root.is_dir():
        raise SkillError(f"skills_dirs entry is not a directory: {root}")
    stack: list[tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()
    visited = 0
    while stack:
        directory, depth = stack.pop()
        try:
            canonical = directory.resolve(strict=True)
        except OSError as exc:
            raise SkillError(
                f"cannot resolve skills directory {directory} "
                f"({type(exc).__name__})"
            ) from exc
        if canonical in seen:
            continue
        seen.add(canonical)
        visited += 1
        if visited > _DISCOVERY_MAX_DIRS:
            raise SkillError(
                f"skills discovery under {root} exceeds {_DISCOVERY_MAX_DIRS} "
                "directories"
            )
        candidate = canonical / "SKILL.md"
        if depth > 0 and candidate.is_file():
            yield candidate
        if depth >= _DISCOVERY_MAX_DEPTH:
            continue
        try:
            children = sorted(
                (
                    child for child in canonical.iterdir()
                    if child.name not in _SKIP_DIR_NAMES and child.is_dir()
                ),
                key=lambda child: child.name,
                reverse=True,
            )
        except OSError as exc:
            raise SkillError(
                f"cannot scan skills directory {canonical} "
                f"({type(exc).__name__})"
            ) from exc
        stack.extend((child, depth + 1) for child in children)


def _candidate_files(
    cwd: Path,
    *,
    skills_dirs: Sequence[str],
    skill_paths: Sequence[str],
    project_root: Path,
) -> Iterator[tuple[Path, bool]]:
    # Exact paths are the most specific setting and therefore participate in
    # first-wins collision handling before discovered directory entries.
    for value in skill_paths:
        candidate = _expand_path(value, base=cwd)
        if candidate.is_dir():
            candidate = candidate / "SKILL.md"
        if not candidate.is_file():
            raise SkillError(f"configured skill_path does not exist: {candidate}")
        if candidate.name != "SKILL.md":
            raise SkillError(f"configured skill_path must name SKILL.md: {candidate}")
        yield candidate, True

    project_dirs = _project_search_dirs(cwd, project_root)
    for value in skills_dirs:
        expanded = os.path.expandvars(os.path.expanduser(value))
        configured = Path(expanded)
        roots = (
            (configured.resolve(strict=False),)
            if configured.is_absolute()
            else tuple((directory / configured).resolve(strict=False)
                       for directory in project_dirs)
        )
        for root in roots:
            for path in _walk_skill_files(root):
                yield path, False


def discover_skills(
    cwd: str | Path,
    *,
    enabled: bool,
    skills_dirs: Sequence[str] = DEFAULT_SKILLS_DIRS,
    skill_paths: Sequence[str] = (),
    root_markers: Sequence[str] = (".git", ".hg", ".sl"),
    unreadable_paths: Sequence[str] = (),
) -> SkillCatalog:
    """Discover and validate skills once, retaining the first name collision."""
    validate_skill_settings(enabled, skills_dirs, skill_paths)
    if not enabled:
        return SkillCatalog()
    work_dir = Path(cwd).resolve(strict=True)
    project_root = find_project_root(work_dir, root_markers)
    blocked = _UnreadableMatcher(work_dir, unreadable_paths)
    seen_files: set[Path] = set()
    by_name: dict[str, Skill] = {}
    for raw_path, explicit in _candidate_files(
        work_dir,
        skills_dirs=skills_dirs,
        skill_paths=skill_paths,
        project_root=project_root,
    ):
        path = raw_path.resolve(strict=True)
        if path in seen_files:
            continue
        seen_files.add(path)
        if blocked.blocks(path):
            if explicit:
                raise SkillError(
                    f"configured skill is hidden by sandbox policy: {path}"
                )
            log.warning("skill hidden by sandbox policy: %s", path)
            continue
        skill = load_skill(path)
        previous = by_name.get(skill.name)
        if previous is not None:
            log.warning(
                "skill name collision: keeping first %s and ignoring %s",
                previous.path,
                skill.path,
            )
            continue
        by_name[skill.name] = skill
    return SkillCatalog(tuple(by_name.values()))


__all__ = [
    "DEFAULT_SKILLS_DIRS",
    "Skill",
    "SkillCatalog",
    "SkillError",
    "discover_skills",
    "load_skill",
    "validate_skill_settings",
]
