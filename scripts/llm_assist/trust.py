"""Workspace-scoped trust for repository-provided startup behavior."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..llm_solver.config import Config
from ..llm_solver.harness.project_instructions import (
    DEFAULT_OVERRIDE_NAME,
    find_project_root,
)
from ..llm_solver.harness.prompt_imports import process_imports
from .store import SessionStore


_MANIFEST_VERSION = 1
_SKILL_DEPTH_LIMIT = 6
_SKILL_DIRECTORY_LIMIT = 2_000


class WorkspaceTrustError(RuntimeError):
    """Workspace behavior cannot be inspected or is not trusted."""


@dataclass(frozen=True, slots=True)
class BehaviorItem:
    category: str
    path: str
    logical_path: str
    kind: str

    def stored_record(self) -> dict[str, str]:
        return {
            "category": self.category,
            "path": self.path,
            "logical_path": self.logical_path,
            "kind": self.kind,
        }

    def digest_record(self) -> dict[str, str]:
        return {
            "category": self.category,
            "logical_path": self.logical_path,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class BehaviorManifest:
    workspace: Path
    behavior_root: Path
    items: tuple[BehaviorItem, ...]
    digest: str

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({item.category for item in self.items}))

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": _MANIFEST_VERSION,
                "workspace": str(self.workspace),
                "behavior_root": str(self.behavior_root),
                "manifest_digest": self.digest,
                "categories": list(self.categories),
                "items": [item.stored_record() for item in self.items],
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _inside(path: Path, root: Path) -> bool:
    candidate = path.absolute()
    base = root.absolute()
    try:
        candidate.relative_to(base)
    except ValueError:
        return False
    return True


def require_trust_store_outside_workspace(
    store_root: Path,
    workspace: Path,
) -> None:
    root = Path(store_root).expanduser().resolve()
    target = Path(workspace).expanduser().resolve()
    if _inside(root, target):
        raise WorkspaceTrustError(
            "assistant state must stay outside the selected workspace before "
            "workspace trust can be recorded"
        )


def _require_regular_file(path: Path) -> None:
    try:
        path.lstat()
    except OSError as exc:
        raise WorkspaceTrustError(
            f"cannot inspect workspace behavior file {path}: {type(exc).__name__}"
        ) from exc
    if path.is_symlink():
        raise WorkspaceTrustError(
            f"workspace behavior path must not be a symbolic link: {path}"
        )
    if not path.is_file():
        raise WorkspaceTrustError(
            f"workspace behavior path is not a regular file: {path}"
        )


def _logical_path(path: Path, roots: Sequence[Path]) -> str:
    absolute = path.absolute()
    for root in roots:
        try:
            relative = absolute.relative_to(root.absolute())
        except ValueError:
            continue
        return f"project/{relative.as_posix()}"
    return f"external/{path.name}"


def _file_item(
    category: str,
    path: Path,
    *,
    logical_roots: Sequence[Path],
) -> BehaviorItem:
    _require_regular_file(path)
    return BehaviorItem(
        category=category,
        path=str(path.absolute()),
        logical_path=_logical_path(path, logical_roots),
        kind="file",
    )


def _workspace_path_item(
    category: str,
    raw_path: Path,
    *,
    workspace: Path,
    logical_roots: Sequence[Path],
) -> BehaviorItem:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = Path(os.path.abspath(candidate))
    if not _inside(candidate, workspace):
        raise WorkspaceTrustError(
            f"task attachment path escapes the selected workspace: {raw_path}"
        )
    current = workspace
    for part in candidate.relative_to(workspace).parts:
        current = current / part
        try:
            current.lstat()
        except OSError as exc:
            raise WorkspaceTrustError(
                f"cannot inspect task attachment path {raw_path}: "
                f"{type(exc).__name__}"
            ) from exc
        if current.is_symlink():
            raise WorkspaceTrustError(
                f"task attachment path must not cross a symbolic link: {current}"
            )
    if candidate.is_file():
        kind = "file"
    elif candidate.is_dir():
        kind = "directory"
    else:
        raise WorkspaceTrustError(
            "task attachment path must name a regular file or directory: "
            f"{raw_path}"
        )
    return BehaviorItem(
        category=category,
        path=str(candidate),
        logical_path=_logical_path(candidate, logical_roots),
        kind=kind,
    )


def _configured_path(base: Path, value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else base / expanded


def _workspace_compaction_hook_sources(
    reference: str,
    *,
    cwd: Path,
    project_root: Path,
) -> tuple[Path, ...]:
    """Locate repository Python sources without importing the configured hook."""
    normalized = str(reference or "").strip()
    if normalized.count(":") != 1:
        return ()
    module_name, _function_name = normalized.split(":", 1)
    parts = module_name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        return ()
    found: list[Path] = []
    for root in dict.fromkeys((cwd, project_root)):
        module_base = root.joinpath(*parts)
        module_file = module_base.with_suffix(".py")
        package_init = module_base / "__init__.py"
        if module_file.is_file() or module_file.is_symlink():
            found.append(module_file)
        elif package_init.is_file() or package_init.is_symlink():
            found.append(package_init)
    return tuple(dict.fromkeys(found))


def _root_to_cwd(project_root: Path, cwd: Path) -> tuple[Path, ...]:
    relative = cwd.resolve().relative_to(project_root.resolve())
    directories = [project_root]
    current = project_root
    for part in relative.parts:
        current = current / part
        directories.append(current)
    return tuple(directories)


def _project_instruction_sources(cfg: Config, cwd: Path) -> tuple[Path, ...]:
    if not cfg.project_docs_enabled:
        return ()
    project_root = find_project_root(cwd, cfg.project_root_markers)
    names = tuple(dict.fromkeys((DEFAULT_OVERRIDE_NAME, *cfg.project_doc_names)))
    selected: list[Path] = []
    for directory in _root_to_cwd(project_root, cwd):
        for name in names:
            candidate = directory / name
            if candidate.is_symlink():
                raise WorkspaceTrustError(
                    f"project instruction must not be a symbolic link: {candidate}"
                )
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text(encoding="utf-8-sig", errors="replace")
            except OSError as exc:
                raise WorkspaceTrustError(
                    f"cannot inspect project instruction {candidate}: "
                    f"{type(exc).__name__}"
                ) from exc
            if text.strip():
                selected.append(candidate)
                break
    return tuple(selected)


def _imported_prompt_paths(
    source: Path,
    *,
    allowed_roots: Sequence[Path],
    max_depth: int,
) -> tuple[Path, ...]:
    try:
        text = source.read_text(encoding="utf-8-sig", errors="replace")
        processed = process_imports(
            text,
            source.parent,
            allowed_roots,
            max_depth=max_depth,
            source_path=source,
        )
    except (OSError, ValueError) as exc:
        raise WorkspaceTrustError(
            f"cannot inspect prompt imports from {source}: {exc}"
        ) from exc
    imported: list[Path] = []
    for label in processed.imported_files:
        for root in allowed_roots:
            candidate = root / label
            if candidate.is_file() or candidate.is_symlink():
                imported.append(candidate)
    return tuple(dict.fromkeys(imported))


def _skill_sources(cfg: Config, cwd: Path, project_root: Path) -> tuple[Path, ...]:
    sources: list[Path] = []
    for value in cfg.skill_paths:
        candidate = _configured_path(cwd, value)
        if candidate.is_dir():
            candidate = candidate / "SKILL.md"
        if _inside(candidate, project_root) and (
            candidate.is_file() or candidate.is_symlink()
        ):
            sources.append(candidate)
    search_directories = tuple(reversed(_root_to_cwd(project_root, cwd)))
    for value in cfg.skills_dirs:
        expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
        candidates = (
            (expanded,)
            if expanded.is_absolute()
            else tuple(directory / expanded for directory in search_directories)
        )
        for candidate in candidates:
            if not _inside(candidate, project_root) or not candidate.is_dir():
                continue
            for skill in _find_skill_files(candidate):
                sources.append(skill)
    return tuple(dict.fromkeys(sources))


def _find_skill_files(root: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    visited = 0
    while stack:
        directory, depth = stack.pop()
        visited += 1
        if visited > _SKILL_DIRECTORY_LIMIT:
            raise WorkspaceTrustError(
                f"workspace skill discovery exceeds {_SKILL_DIRECTORY_LIMIT} "
                f"directories: {root}"
            )
        if directory.is_symlink():
            raise WorkspaceTrustError(
                f"workspace skill directory must not be a symbolic link: {directory}"
            )
        skill = directory / "SKILL.md"
        if depth > 0 and skill.is_file():
            found.append(skill)
        if depth >= _SKILL_DEPTH_LIMIT:
            continue
        try:
            children: list[Path] = []
            for entry in os.scandir(directory):
                if entry.name in {".git", "node_modules", "__pycache__"}:
                    continue
                path = Path(entry.path)
                if path.is_symlink():
                    if entry.name == "SKILL.md" or entry.is_dir():
                        raise WorkspaceTrustError(
                            "workspace skill discovery must not cross a "
                            f"symbolic link: {path}"
                        )
                    continue
                if entry.is_dir(follow_symlinks=False):
                    children.append(path)
            children.sort(reverse=True)
        except WorkspaceTrustError:
            raise
        except OSError as exc:
            raise WorkspaceTrustError(
                f"cannot inspect workspace skill directory {directory}: "
                f"{type(exc).__name__}"
            ) from exc
        stack.extend((child, depth + 1) for child in children)
    return tuple(found)


def _deduplicate_items(items: Iterable[BehaviorItem]) -> tuple[BehaviorItem, ...]:
    by_key: dict[tuple[str, str, str], BehaviorItem] = {}
    for item in items:
        key = (item.category, item.logical_path, item.kind)
        by_key[key] = item
    return tuple(
        sorted(
            by_key.values(),
            key=lambda item: (item.category, item.logical_path, item.kind),
        )
    )


def discover_workspace_behavior(
    cfg: Config,
    *,
    workspace: Path,
    behavior_root: Path | None = None,
    config_paths: Sequence[Path] = (),
    system_prompt_file: Path | None = None,
    task_attachment_paths: Sequence[Path] = (),
) -> BehaviorManifest:
    """Inventory repository startup inputs without activating them."""
    scope = Path(workspace).expanduser().resolve()
    target = Path(behavior_root or workspace).expanduser().resolve()
    if not scope.is_dir() or not target.is_dir():
        empty_payload = json.dumps(
            {"schema_version": _MANIFEST_VERSION, "items": []},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return BehaviorManifest(
            workspace=scope,
            behavior_root=target,
            items=(),
            digest=hashlib.sha256(empty_payload).hexdigest(),
        )
    target_project_root = find_project_root(target, cfg.project_root_markers)
    scope_project_root = find_project_root(scope, cfg.project_root_markers)
    logical_roots = tuple(dict.fromkeys((target_project_root, scope_project_root)))
    items: list[BehaviorItem] = []

    for raw_path in task_attachment_paths:
        items.append(
            _workspace_path_item(
                "task_attachments",
                Path(raw_path),
                workspace=scope,
                logical_roots=logical_roots,
            )
        )

    repository_configs: list[Path] = []
    for raw_path in config_paths:
        path = Path(raw_path).expanduser().resolve()
        if _inside(path, scope) or _inside(path, target):
            repository_configs.append(path)
            items.append(
                _file_item("configuration", path, logical_roots=logical_roots)
            )
    configured_categories: list[str] = []
    if cfg.hooks_enabled and cfg.hooks:
        configured_categories.append("lifecycle_hooks")
    if cfg.compaction_hook.strip():
        configured_categories.append("compaction_hook")
    if cfg.lsp_enabled and cfg.lsp_servers:
        configured_categories.append("language_servers")
    if cfg.post_edit_checks:
        configured_categories.append("post_edit_checks")
    for category in configured_categories:
        for path in repository_configs:
            items.append(
                _file_item(category, path, logical_roots=logical_roots)
            )
    for source in _workspace_compaction_hook_sources(
        cfg.compaction_hook,
        cwd=target,
        project_root=target_project_root,
    ):
        items.append(
            _file_item(
                "compaction_hook", source, logical_roots=logical_roots
            )
        )

    prompt_roots = tuple(
        dict.fromkeys(
            (
                target_project_root,
                *(
                    (Path(system_prompt_file).expanduser().resolve().parent,)
                    if system_prompt_file is not None
                    else ()
                ),
            )
        )
    )
    if system_prompt_file is not None:
        prompt_path = Path(system_prompt_file).expanduser().resolve()
        prompt_paths = (prompt_path,)
        if cfg.imports_enabled and prompt_path.is_file():
            prompt_paths += _imported_prompt_paths(
                prompt_path,
                allowed_roots=prompt_roots,
                max_depth=cfg.imports_max_depth,
            )
        for path in prompt_paths:
            if _inside(path, target_project_root) or _inside(path, scope_project_root):
                items.append(
                    _file_item("system_prompt", path, logical_roots=logical_roots)
                )

    for source in _project_instruction_sources(cfg, target):
        items.append(
            _file_item("project_instructions", source, logical_roots=logical_roots)
        )
        if cfg.imports_enabled:
            for imported in _imported_prompt_paths(
                source,
                allowed_roots=(target_project_root,),
                max_depth=cfg.imports_max_depth,
            ):
                items.append(
                    _file_item(
                        "project_instructions",
                        imported,
                        logical_roots=logical_roots,
                    )
                )

    if cfg.skills_enabled:
        for source in _skill_sources(cfg, target, target_project_root):
            items.append(
                _file_item("skills", source, logical_roots=logical_roots)
            )

    if cfg.injections_enabled:
        injection_dir = _configured_path(target, cfg.injections_dir)
        if _inside(injection_dir, target_project_root) and injection_dir.is_dir():
            for source in sorted(injection_dir.glob("*.md")):
                items.append(
                    _file_item("injections", source, logical_roots=logical_roots)
                )
                if cfg.imports_enabled:
                    for imported in _imported_prompt_paths(
                        source,
                        allowed_roots=(target_project_root,),
                        max_depth=cfg.imports_max_depth,
                    ):
                        items.append(
                            _file_item(
                                "injections",
                                imported,
                                logical_roots=logical_roots,
                            )
                        )

    if cfg.stream_rules_enabled:
        stream_dir = _configured_path(target, cfg.stream_rules_dir)
        if _inside(stream_dir, target) and stream_dir.is_dir():
            for source in sorted(stream_dir.glob("*.md")):
                items.append(
                    _file_item("stream_rules", source, logical_roots=logical_roots)
                )

    if cfg.state_ignore_file_enabled:
        for name in cfg.state_ignore_file_names:
            source = target / name
            if source.is_file() or source.is_symlink():
                items.append(
                    _file_item("ignore_policy", source, logical_roots=logical_roots)
                )

    resolved_items = _deduplicate_items(items)
    digest_payload = json.dumps(
        {
            "schema_version": _MANIFEST_VERSION,
            "items": [item.digest_record() for item in resolved_items],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return BehaviorManifest(
        workspace=scope,
        behavior_root=target,
        items=resolved_items,
        digest=hashlib.sha256(digest_payload).hexdigest(),
    )


def workspace_trust_state(store: SessionStore, manifest: BehaviorManifest) -> str:
    if not manifest.items:
        return "not_required"
    record = store.get_workspace_trust(manifest.workspace)
    if record is None:
        return "untrusted"
    return "trusted"


def save_workspace_trust(
    store: SessionStore,
    manifest: BehaviorManifest,
) -> None:
    require_trust_store_outside_workspace(store.root, manifest.workspace)
    store.set_workspace_trust(
        manifest.workspace,
        manifest_digest=manifest.digest,
        manifest_json=manifest.to_json(),
    )


def require_saved_workspace_trust(
    store: SessionStore,
    manifest: BehaviorManifest,
) -> None:
    state = workspace_trust_state(store, manifest)
    if state in {"not_required", "trusted"}:
        return
    raise WorkspaceTrustError(
        f"workspace behavior trust is {state}; inspect and trust this workspace"
    )


def render_workspace_behavior(
    manifest: BehaviorManifest,
    *,
    state: str,
) -> str:
    lines = [
        f"workspace: {manifest.workspace}",
        f"workspace_trust: {state}",
        "behavior_categories: " + (", ".join(manifest.categories) or "none"),
        f"behavior_manifest_sha256: {manifest.digest}",
    ]
    for item in manifest.items:
        lines.append(f"behavior: {item.category} {item.path}")
    return "\n".join(lines) + "\n"


__all__ = [
    "BehaviorItem",
    "BehaviorManifest",
    "WorkspaceTrustError",
    "discover_workspace_behavior",
    "render_workspace_behavior",
    "require_saved_workspace_trust",
    "require_trust_store_outside_workspace",
    "save_workspace_trust",
    "workspace_trust_state",
]
