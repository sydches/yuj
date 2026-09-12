"""Post-mutation component verification selection and state."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shlex
from typing import Any

from ..._shared.classification import classify_outcome, is_error_result
from ...language_quirks import (
    load_run_tests_quirk_object,
)
from ..command_redirect import split_shell_fragments, strip_leading_assignments
from ..bash_write_classification import is_workspace_path, normalize_trace_path
from .extractors import MUTATION_TOOLS, _is_bash_write_like
from .state import PASS, Decision, GuardrailState
from .custom_execution import (
    completed_custom_execution, _observe_runtime_executable,
    observed_component_runner_base_cmd,
)


def _file_revision(cwd: str | Path, path: str) -> str:
    try:
        from ..task_path import resolve_task_path
        target = resolve_task_path(cwd, path)
        with target.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return "missing"
    except (OSError, RuntimeError, ValueError):
        return ""


def verification_tree_matches(state: GuardrailState, cwd: str | Path | None) -> bool:
    """Check only the files observed in successful edits, not task semantics."""
    revisions = state.verification_file_revisions
    if not revisions:
        return True
    return bool(cwd) and all(
        digest and _file_revision(cwd, path) == digest
        for path, digest in revisions.items()
    )


def verification_changes_tree(tc_name: str, tc_args: dict | None) -> bool:
    """Do not credit a test bundled with an explicit Git tree change."""
    if tc_name != "bash" or not isinstance(tc_args, dict):
        return False
    for fragment in split_shell_fragments(str(tc_args.get("cmd") or "")):
        try:
            argv = shlex.split(strip_leading_assignments(fragment.text))
        except ValueError:
            continue
        if not argv or argv[0].rsplit("/", 1)[-1] != "git":
            continue
        args = iter(argv[1:])
        for arg in args:
            if arg in {"-C", "-c", "--git-dir", "--work-tree"}:
                next(args, None)
            elif not arg.startswith("-"):
                if arg in {
                    "stash", "checkout", "switch", "restore", "reset",
                    "revert", "cherry-pick", "rebase", "merge", "apply", "clean",
                }:
                    return True
                break
    return False


@dataclass(frozen=True)
class ComponentVerificationTarget:
    """One mechanically selected target for the registered test runner."""

    path: str
    display: str
    runner: str
    source_path: str
    native_sources: tuple[str, ...] = ()


def automatic_component_verification_due(
    state: GuardrailState,
    cfg: Any,
) -> bool:
    """Return whether this source revision needs its one automatic run."""
    return bool(
        int(getattr(cfg, "post_mutation_verification_gate_after", 0) or 0) > 0
        and state.has_mutated
        and state.post_mutation_verification_gate_armed
        and not state.post_mutation_automatic_verification_attempted
        and not state.formal_verification_passed_since_mutation
    )


def resolve_component_verification_target(
    state: GuardrailState,
    cwd: str | Path,
    *,
    ignore_policy: Any = None,
    runner: str = "auto",
) -> ComponentVerificationTarget | None:
    """Select a unique conventional component target without task knowledge."""
    from ..task_path import resolve_task_path
    root = resolve_task_path(cwd, '.')
    quirk = load_run_tests_quirk_object(root, runner=runner)
    sources = _safe_source_paths(root, state.post_mutation_source_paths)
    if not sources:
        return None

    if quirk.extra_fields.get("component_collection_plugin"):
        # The runner selects from its actual collection during execution.
        return ComponentVerificationTarget(
            path="", display="<runner-collected component>", runner=quirk.runner,
            source_path=sources[0].relative_to(root).as_posix(),
            native_sources=tuple(source.relative_to(root).as_posix() for source in sources),
        )

    for source in sources:
        names = _component_test_names(quirk.component_test_names, source)
        if not names:
            continue
        candidates = _pretest_candidates(root, state, names)
        if not candidates:
            candidates = _walk_named_files(
                root, names, ignore_policy=ignore_policy
            )
        selected = _select_unique_candidate(source, candidates)
        if selected is None:
            continue
        relative_test = selected.relative_to(root).as_posix()
        relative_source = source.relative_to(root).as_posix()
        parent = selected.parent.relative_to(root).as_posix() or "."
        try:
            target = quirk.component_target_template.format(
                test_path=relative_test,
                test_parent=parent,
                source_path=relative_source,
                source_parent=source.parent.relative_to(root).as_posix() or ".",
                stem=source.stem,
                suffix=source.suffix,
            )
        except (KeyError, ValueError):
            return None
        if target == "./.":
            target = "."
        if target.strip():
            return ComponentVerificationTarget(
                path=target,
                display=target,
                runner=quirk.runner,
                source_path=relative_source,
            )

    if quirk.component_fallback == "suite":
        return ComponentVerificationTarget(
            path="",
            display="<registered suite>",
            runner=quirk.runner,
            source_path=sources[0].relative_to(root).as_posix(),
        )
    return None


def mark_automatic_component_verification_attempted(
    state: GuardrailState,
    target: ComponentVerificationTarget | None,
) -> None:
    """Prevent another automatic run until a later source mutation."""
    state.post_mutation_automatic_verification_attempted = True
    state.post_mutation_verification_gate_armed = False
    state.post_mutation_automatic_verification_target = (
        target.display if target is not None else ""
    )


def verification_result_passed(tc_name: str, result: str, execution_metadata: dict | None = None, *, formal: bool = True) -> bool:
    """Use private runner facts; displayed test tags cannot grant a pass."""
    if tc_name in {"run_tests", "bash", "bash_poll"}:
        facts = execution_metadata or {}
        return bool(facts.get("executed")
                    and facts.get("exit_status_known")
                    and facts.get("exit_status") == 0
                    and not facts.get("timed_out")
                    and not facts.get("security_blocked_stage")
                    and facts.get("verification_status") in ({"passed"} if formal else {"passed", "custom_passed"}))
    return False


def verification_runner_unavailable(result: str, *, tc_name: str = "bash",
                                    execution_metadata: dict | None = None) -> bool:
    """Return whether a registered runner could not start."""
    if tc_name in {"run_tests", "bash", "bash_poll"}:
        facts = execution_metadata or {}
        return bool(facts.get("executed")
                    and not facts.get("security_blocked_stage")
                    and facts.get("verification_status") == "runner_unavailable")
    return False


def _safe_source_paths(root: Path, raw_paths: tuple[str, ...]) -> list[Path]:
    from ..task_path import resolve_task_path
    paths: list[Path] = []
    for raw in raw_paths:
        try:
            resolved = resolve_task_path(root, raw)
            if resolved.is_file() and resolved not in paths:
                paths.append(resolved)
        except (ValueError, OSError, RuntimeError):
            continue
    return paths


def _component_test_names(templates: tuple[str, ...], source: Path) -> set[str]:
    names: set[str] = set()
    for template in templates:
        try:
            name = template.format(
                name=source.name,
                stem=source.stem,
                suffix=source.suffix,
            )
        except (KeyError, ValueError):
            continue
        if name and "/" not in name and "\\" not in name:
            names.add(name)
    return names


def _pretest_candidates(
    root: Path,
    state: GuardrailState,
    names: set[str],
) -> list[Path]:
    candidates: list[Path] = []
    from ..task_path import resolve_task_path
    for test_id in sorted(
        state.pretest_failing_tests | state.pretest_passing_tests
    ):
        raw = test_id.split("::", 1)[0]
        try:
            candidate = resolve_task_path(root, raw)
            if candidate.name in names and candidate.is_file() and candidate not in candidates:
                candidates.append(candidate)
        except (ValueError, OSError, RuntimeError):
            continue
    return candidates


def _walk_named_files(
    root: Path,
    names: set[str],
    *,
    ignore_policy: Any,
) -> list[Path]:
    from ..task_path import TaskPath
    candidates: list[Path] = []
    traversal = root.walk() if isinstance(root, TaskPath) else os.walk(root, topdown=True, followlinks=False)
    for directory, dir_names, file_names in traversal:
        base = directory if isinstance(directory, TaskPath) else Path(directory)
        kept_dirs: list[str] = []
        for name in sorted(dir_names):
            path = base / name
            if name in {".git", ".solver", ".tool_output"} or path.is_symlink():
                continue
            if ignore_policy is not None:
                try:
                    if ignore_policy.is_model_hidden(path, is_dir=True):
                        continue
                except (OSError, ValueError):
                    continue
            kept_dirs.append(name)
        dir_names[:] = kept_dirs
        for name in sorted(file_names):
            if name not in names:
                continue
            path = base / name
            if ignore_policy is not None:
                try:
                    if ignore_policy.is_ignored(path, is_dir=False):
                        continue
                except (OSError, ValueError):
                    continue
            try:
                candidate = path.resolve(strict=False)
                candidate.relative_to(root)
                candidates.append(candidate)
            except (ValueError, OSError, RuntimeError):
                continue
            if len(candidates) > 64:
                return []
    return candidates


def _select_unique_candidate(source: Path, candidates: list[Path]) -> Path | None:
    """Preserve ambiguity; path spelling cannot establish source relevance.

    Callers supply resolved candidates. Uniqueness retains the existing
    conventional fallback, not proof of collection membership or coverage.
    """
    del source
    unique = set(candidates)
    return next(iter(unique)) if len(unique) == 1 else None


def post_mutation_verification_gate(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    tc_args: dict | None = None,
) -> Decision:
    """Keep the legacy registry slot non-blocking under mechanical H4."""
    del state, cfg, tc_name, tc_args
    return PASS


def record_verification_mutation(
    state: GuardrailState,
    *,
    tc_name: str,
    result: str,
    tc_args: dict | None = None,
    source_write_paths: tuple[str, ...] = (),
    execution_metadata: dict | None = None,
    cwd: str | Path | None = None,
) -> None:
    """Invalidate old verification and retain the changed input revisions."""
    # Accepted check outputs are effects, not newly edited inputs.
    check_outputs = bool(
        (execution_metadata or {}).get("_verification_inputs_unchanged")
        and verification_result_passed(tc_name, result, execution_metadata, formal=False)
        and not verification_changes_tree(tc_name, tc_args)
    )
    if cwd and not check_outputs:
        paths = set(state.verification_file_revisions)
        paths.update(normalize_trace_path(path) for path in source_write_paths
                     if is_workspace_path(path))
        state.verification_file_revisions = {
            path: _file_revision(cwd, path) for path in sorted(paths)
        }
    state.post_mutation_non_test_bash_count = 0
    state.post_mutation_verification_gate_armed = False
    state.formal_verification_passed_since_mutation = False
    state.post_mutation_automatic_verification_attempted = False
    state.post_mutation_automatic_verification_target = ""
    state.post_mutation_source_paths = tuple(source_write_paths)
    state.post_mutation_observed_runtime_family = ""
    state.post_mutation_observed_runtime_executable = ""
    state.post_mutation_observed_runtime_binding = {}
    state.post_mutation_automatic_verification_unavailable = False


def observe_post_mutation_verification(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
    source_write_paths: tuple[str, ...] = (),
    execution_metadata: dict | None = None,
    cwd: str | Path | None = None,
    **_: Any,
) -> None:
    """Arm one automatic component run after repeated custom checks."""
    if gate_blocked:
        return
    from ..file_changes import observed_mutation
    observed = observed_mutation(execution_metadata)
    changes = (execution_metadata or {}).get("file_changes")
    if changes is not None and changes.get("status") in {"unavailable", "incomplete"}:
        state.verified_since_mutation = False
        state.formal_verification_passed_since_mutation = False
        return
    accounted = bool((execution_metadata or {}).get("_mutation_accounted"))
    if accounted and not (execution_metadata or {}).get("_verification_inputs_unchanged"):
        return
    if (observed is True and not accounted) or (observed is None and (
            tc_name in MUTATION_TOOLS or _is_bash_write_like(tc_name, tc_args))):
        if observed is True or not is_error_result(result):
            record_verification_mutation(
                state, tc_name=tc_name, result=result, tc_args=tc_args,
                source_write_paths=source_write_paths,
                execution_metadata=execution_metadata, cwd=cwd,
            )
        return
    if not state.has_mutated:
        return
    if tc_name not in {"bash", "exec_cell", "run_tests", "bash_poll"}:
        return
    if (not verification_tree_matches(state, cwd)
            or verification_changes_tree(tc_name, tc_args)):
        return
    facts = execution_metadata or {}
    formal_attempt = (tc_name == "run_tests"
                      or (facts.get("runner_request") or {}).get("check_intent")
                      or facts.get("verification_status") in {"passed", "failed", "runner_unavailable"})
    if (facts.get("executed") and facts.get("exit_status_known") and
            formal_attempt):
        state.post_mutation_non_test_bash_count = 0
        state.post_mutation_verification_gate_armed = False
        passed = verification_result_passed(tc_name, result, execution_metadata)
        state.formal_verification_passed_since_mutation = passed
        state.verified_since_mutation = passed
        if not verification_runner_unavailable(result, tc_name=tc_name,
                                               execution_metadata=execution_metadata):
            state.post_mutation_automatic_verification_unavailable = False
        return
    if state.post_mutation_automatic_verification_attempted:
        return
    if tc_name != "bash" or not completed_custom_execution(facts, cwd):
        return
    observed_runtime = _observe_runtime_executable(facts)
    state.post_mutation_observed_runtime_family = ""
    state.post_mutation_observed_runtime_executable = ""
    state.post_mutation_observed_runtime_binding = {}
    if observed_runtime is not None:
        (
            state.post_mutation_observed_runtime_family,
            state.post_mutation_observed_runtime_executable,
        ) = observed_runtime
        state.post_mutation_observed_runtime_binding = dict(facts["shell_submission"])
    state.post_mutation_non_test_bash_count += 1
    threshold = int(
        getattr(cfg, "post_mutation_verification_gate_after", 0) or 0
    )
    if threshold > 0 and state.post_mutation_non_test_bash_count >= threshold:
        state.post_mutation_verification_gate_armed = True
