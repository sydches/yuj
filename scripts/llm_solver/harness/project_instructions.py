"""Deterministic discovery and assembly of repository instruction files.

This is a leaf facility: it accepts ordinary paths and values rather than a
harness ``Config`` object, and it neither emits trace events nor mutates the
conversation.  The session setup layer owns those integration steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
from typing import Iterable, Sequence

from .prompt_imports import process_imports


DEFAULT_PROJECT_DOC_NAMES = ("AGENTS.md", "CLAUDE.md")
DEFAULT_PROJECT_ROOT_MARKERS = (".git", ".hg", ".sl")
DEFAULT_OVERRIDE_NAME = "AGENTS.override.md"
DEFAULT_PROJECT_DOC_MAX_BYTES = 32 * 1024
_GLOB_META = frozenset("*?[")


@dataclass(frozen=True, slots=True)
class InstructionDocument:
    """One selected instruction file and the bounded content loaded from it."""

    source_path: Path
    display_path: str
    content: str
    byte_count: int
    scope: str
    truncated: bool = False

    def format_block(self) -> str:
        """Wrap content in the model-facing project-instructions envelope."""
        return (
            f'<project-instructions path="{_xml_attr(self.display_path)}">\n'
            f"{self.content.rstrip()}\n"
            "</project-instructions>"
        )

    def trace_record(self) -> dict[str, object]:
        """Return secret-free, JSON-ready session-start metadata."""
        return {
            "path": self.display_path,
            "bytes": self.byte_count,
            "scope": self.scope,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class InstructionDiagnostic:
    """A non-fatal candidate read failure, expressed with a safe path label."""

    path: str
    error_kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ProjectInstructions:
    """Resolved instruction chain and its deterministic provenance."""

    content: str
    documents: tuple[InstructionDocument, ...]
    document_bytes: int
    max_bytes: int
    truncated: bool
    project_root: Path
    diagnostics: tuple[InstructionDiagnostic, ...] = ()
    developer_instructions: str = ""
    imported_bytes: int = 0
    resolved_bytes: int = 0
    prompt_import_tree: tuple[dict[str, object], ...] = ()

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(document.display_path for document in self.documents)

    def trace_records(self) -> list[dict[str, object]]:
        return [document.trace_record() for document in self.documents]


class _UnreadableMatcher:
    """Expand sandbox unreadable patterns once and reject descendants."""

    def __init__(self, base_dir: Path, patterns: Sequence[str]) -> None:
        blocked: set[Path] = set()
        for original in patterns:
            pattern = str(original)
            if pattern.startswith("optional:"):
                pattern = pattern[len("optional:"):]
            expanded = os.path.expandvars(os.path.expanduser(pattern))
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            text = str(candidate)
            if any(char in text for char in _GLOB_META):
                blocked.update(
                    Path(match).resolve(strict=False)
                    for match in glob.glob(
                        text,
                        recursive=True,
                        include_hidden=True,
                    )
                )
            else:
                blocked.add(candidate.resolve(strict=False))
        self._blocked = tuple(sorted(blocked, key=lambda item: str(item)))

    def blocks(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return any(
            resolved == blocked or blocked in resolved.parents
            for blocked in self._blocked
        )


def find_project_root(
    cwd: str | Path,
    markers: Sequence[str] = DEFAULT_PROJECT_ROOT_MARKERS,
) -> Path:
    """Find the nearest marked ancestor, falling back to ``cwd``."""
    current = Path(cwd).resolve()
    if not current.is_dir():
        raise ValueError(f"project instruction cwd is not a directory: {cwd}")
    marker_names = _validated_names(markers, field="project_root_markers")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in marker_names):
            return candidate
    return current


def _validated_names(names: Iterable[str], *, field: str) -> tuple[str, ...]:
    output: list[str] = []
    for raw in names:
        name = str(raw)
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"{field} entries must be non-empty filenames: {name!r}")
        if name not in output:
            output.append(name)
    if not output:
        raise ValueError(f"{field} must contain at least one filename")
    return tuple(output)


def validate_project_instruction_settings(
    doc_names: Sequence[str],
    max_bytes: int,
    root_markers: Sequence[str],
) -> None:
    """Validate public settings without touching the filesystem."""
    _validated_names(doc_names, field="project_doc_names")
    _validated_names(root_markers, field="project_root_markers")
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        raise ValueError("project_doc_max_bytes must be a non-negative integer")


def _walk_root_to_cwd(root: Path, cwd: Path) -> tuple[Path, ...]:
    try:
        relative = cwd.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"cwd {cwd} is outside project root {root}") from exc
    directories = [root]
    current = root
    for part in relative.parts:
        current = current / part
        directories.append(current)
    return tuple(directories)


def _within(path: Path, directory: Path) -> bool:
    resolved = path.resolve(strict=False)
    base = directory.resolve(strict=False)
    return resolved == base or base in resolved.parents


def _candidate_names(
    doc_names: Sequence[str],
    *,
    override_name: str,
) -> tuple[str, ...]:
    return _validated_names(
        (override_name, *doc_names),
        field="project_doc_names",
    )


def _safe_display_path(path: Path, *, scope: str, project_root: Path) -> str:
    if scope == "project":
        return path.relative_to(project_root).as_posix()
    return f"global/{path.name}"


def _read_first_nonempty(
    directory: Path,
    *,
    names: Sequence[str],
    scope: str,
    allowed_root: Path,
    project_root: Path,
    unreadable: _UnreadableMatcher,
    seen_paths: set[Path],
) -> tuple[Path, str, bytes] | InstructionDiagnostic | None:
    last_diagnostic: InstructionDiagnostic | None = None
    for name in names:
        candidate = directory / name
        display = _safe_display_path(
            candidate,
            scope=scope,
            project_root=project_root,
        )
        if unreadable.blocks(candidate):
            continue
        resolved = candidate.resolve(strict=False)
        if not _within(resolved, allowed_root) or resolved in seen_paths:
            continue
        if not candidate.is_file():
            continue
        try:
            raw = candidate.read_bytes()
        except OSError as exc:
            last_diagnostic = InstructionDiagnostic(
                display,
                "read_error",
                _safe_os_error("could not read instruction file", exc),
            )
            continue
        text = raw.decode("utf-8-sig", errors="replace")
        if not text.strip():
            continue
        return resolved, text, text.encode("utf-8")
    return last_diagnostic


def _truncate_utf8(data: bytes, limit: int) -> tuple[str, int, bool]:
    if len(data) <= limit:
        return data.decode("utf-8"), len(data), False
    prefix = data[:limit]
    text = prefix.decode("utf-8", errors="ignore")
    used = len(text.encode("utf-8"))
    return text, used, True


def discover_project_instructions(
    cwd: str | Path,
    *,
    global_dir: str | Path | None = None,
    doc_names: Sequence[str] = DEFAULT_PROJECT_DOC_NAMES,
    max_bytes: int = DEFAULT_PROJECT_DOC_MAX_BYTES,
    root_markers: Sequence[str] = DEFAULT_PROJECT_ROOT_MARKERS,
    override_name: str = DEFAULT_OVERRIDE_NAME,
    unreadable_paths: Sequence[str] = (),
    developer_instructions: str = "",
    defer_byte_cap: bool = False,
) -> ProjectInstructions:
    """Discover and concatenate global and root-to-cwd instruction files.

    One first-nonempty file is selected per directory.  Global guidance is
    considered first, followed by project directories from root to ``cwd``.
    The source-content budget is measured in UTF-8 bytes and may truncate the
    final selected document at a valid character boundary. Session assembly
    uses ``defer_byte_cap`` so imports can be expanded before this same budget
    is applied to the resolved content.
    """
    if max_bytes < 0:
        raise ValueError("project_doc_max_bytes must be non-negative")
    resolved_cwd = Path(cwd).resolve()
    project_root = find_project_root(resolved_cwd, root_markers)
    names = _candidate_names(doc_names, override_name=override_name)
    unreadable = _UnreadableMatcher(project_root, unreadable_paths)
    locations: list[tuple[Path, str, Path]] = []
    if global_dir is not None:
        resolved_global = Path(global_dir).expanduser().resolve()
        locations.append((resolved_global, "global", resolved_global))
    locations.extend(
        (directory, "project", project_root)
        for directory in _walk_root_to_cwd(project_root, resolved_cwd)
    )

    documents: list[InstructionDocument] = []
    diagnostics: list[InstructionDiagnostic] = []
    seen_paths: set[Path] = set()
    used = 0
    chain_truncated = False
    for directory, scope, allowed_root in locations:
        if not defer_byte_cap and used >= max_bytes:
            chain_truncated = True
            break
        if not directory.is_dir() or unreadable.blocks(directory):
            continue
        selected = _read_first_nonempty(
            directory,
            names=names,
            scope=scope,
            allowed_root=allowed_root,
            project_root=project_root,
            unreadable=unreadable,
            seen_paths=seen_paths,
        )
        if isinstance(selected, InstructionDiagnostic):
            diagnostics.append(selected)
            continue
        if selected is None:
            continue
        source_path, _text, encoded = selected
        remaining = len(encoded) if defer_byte_cap else max_bytes - used
        content, byte_count, truncated = _truncate_utf8(encoded, remaining)
        if not content:
            chain_truncated = True
            break
        display_path = _safe_display_path(
            source_path,
            scope=scope,
            project_root=project_root,
        )
        documents.append(
            InstructionDocument(
                source_path=source_path,
                display_path=display_path,
                content=content,
                byte_count=byte_count,
                scope=scope,
                truncated=truncated,
            )
        )
        seen_paths.add(source_path)
        used += byte_count
        if truncated:
            chain_truncated = True
            break

    parts: list[str] = []
    if developer_instructions.strip():
        parts.append(developer_instructions.rstrip())
    parts.extend(document.format_block() for document in documents)
    return ProjectInstructions(
        content="\n\n".join(parts),
        documents=tuple(documents),
        document_bytes=used,
        max_bytes=max_bytes,
        truncated=chain_truncated,
        project_root=project_root,
        diagnostics=tuple(diagnostics),
        developer_instructions=developer_instructions,
        resolved_bytes=used,
    )


def resolve_project_instruction_imports(
    project: ProjectInstructions,
    *,
    enabled: bool,
    max_depth: int,
    unreadable_paths: Sequence[str] = (),
) -> ProjectInstructions:
    """Resolve selected documents once, then cap the final UTF-8 content.

    Project documents may import anywhere below the discovered project root.
    A global document is intentionally narrower: it may import only below its
    chosen global directory. The returned trace envelopes contain safe labels
    and byte counts, never prompt bodies or host paths.
    """
    documents: list[InstructionDocument] = []
    import_tree: list[dict[str, object]] = []
    source_bytes = 0
    imported_bytes = 0
    resolved_bytes = 0
    chain_truncated = False

    for index, document in enumerate(project.documents):
        remaining = project.max_bytes - resolved_bytes
        if remaining <= 0:
            chain_truncated = True
            break
        allowed_root = (
            project.project_root
            if document.scope == "project"
            else document.source_path.parent
        )
        resolved_content = document.content
        tree: list[dict[str, object]] = []
        document_imported_bytes = 0
        if enabled:
            processed = process_imports(
                document.content,
                document.source_path.parent,
                (allowed_root,),
                max_depth=max_depth,
                source_path=document.source_path,
                unreadable_paths=unreadable_paths,
            )
            resolved_content = processed.content
            tree = processed.trace_tree()
            document_imported_bytes = processed.imported_bytes

        content, admitted_bytes, truncated = _truncate_utf8(
            resolved_content.encode("utf-8"), remaining
        )
        import_tree.append(
            {
                "owner": "project_instruction",
                "source": document.display_path,
                "source_bytes": document.byte_count,
                "imported_bytes": document_imported_bytes,
                "imports": tree,
            }
        )
        if not content:
            chain_truncated = True
            break
        documents.append(
            InstructionDocument(
                source_path=document.source_path,
                display_path=document.display_path,
                content=content,
                byte_count=document.byte_count,
                scope=document.scope,
                truncated=truncated,
            )
        )
        source_bytes += document.byte_count
        imported_bytes += document_imported_bytes
        resolved_bytes += admitted_bytes
        if truncated:
            chain_truncated = True
            break
        if (
            resolved_bytes >= project.max_bytes
            and index + 1 < len(project.documents)
        ):
            chain_truncated = True
            break

    parts: list[str] = []
    if project.developer_instructions.strip():
        parts.append(project.developer_instructions.rstrip())
    parts.extend(document.format_block() for document in documents)
    return ProjectInstructions(
        content="\n\n".join(parts),
        documents=tuple(documents),
        document_bytes=source_bytes,
        max_bytes=project.max_bytes,
        truncated=chain_truncated,
        project_root=project.project_root,
        diagnostics=project.diagnostics,
        developer_instructions=project.developer_instructions,
        imported_bytes=imported_bytes,
        resolved_bytes=resolved_bytes,
        prompt_import_tree=tuple(import_tree),
    )


def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _safe_os_error(action: str, error: OSError) -> str:
    """Describe an OS failure without retaining its absolute filename."""
    details = type(error).__name__
    if error.errno is not None:
        details += f" errno={error.errno}"
    return f"{action} ({details})"


__all__ = [
    "DEFAULT_OVERRIDE_NAME",
    "DEFAULT_PROJECT_DOC_MAX_BYTES",
    "DEFAULT_PROJECT_DOC_NAMES",
    "DEFAULT_PROJECT_ROOT_MARKERS",
    "InstructionDiagnostic",
    "InstructionDocument",
    "ProjectInstructions",
    "discover_project_instructions",
    "find_project_root",
    "resolve_project_instruction_imports",
]
