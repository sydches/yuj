"""Post-mutation component verification selection and state."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
from typing import Any

from ..._shared.classification import classify_outcome, is_error_result
from ...language_quirks import (
    load_run_tests_quirk_for_runner,
    load_run_tests_quirk_object,
)
from ..command_redirect import split_shell_fragments, strip_leading_assignments
from .extractors import MUTATION_TOOLS, _is_bash_write_like, _is_test_command
from .state import PASS, Decision, GuardrailState


@dataclass(frozen=True)
class ComponentVerificationTarget:
    """One mechanically selected target for the registered test runner."""

    path: str
    display: str
    runner: str
    source_path: str


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
) -> ComponentVerificationTarget | None:
    """Select a unique conventional component target without task knowledge."""
    root = Path(cwd).resolve()
    quirk = load_run_tests_quirk_object(root)
    sources = _safe_source_paths(root, state.post_mutation_source_paths)
    if not sources:
        return None

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


def verification_result_passed(tc_name: str, result: str) -> bool:
    """Read only harness-owned status markers for a verification verdict."""
    if tc_name == "run_tests":
        return '<test_results status="passed"' in result
    return classify_outcome(result) == "OK"


def verification_runner_unavailable(result: str) -> bool:
    """Return whether a registered runner could not start."""
    return '<test_results status="runner_unavailable"' in result


_RUNTIME_FAMILIES = {
    "bun", "cargo", "ctest", "deno", "dotnet", "go", "java", "make",
    "node", "npm", "npx", "php", "pnpm", "ruby", "yarn",
}
_NON_CHECK_EXECUTABLES = {
    "cat", "cp", "diff", "echo", "file", "find", "git", "grep", "head",
    "ls", "mkdir", "mv", "pwd", "rg", "sed", "stat", "tail", "tee",
    "touch", "wc", "which",
}
_UNAVAILABLE_EXIT_RE = re.compile(
    r'(?:\[exit code:\s*12[67]\]|\bexit_code="12[67]")'
)


def _runtime_family(executable: str) -> str:
    leaf = executable.rsplit("/", 1)[-1]
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", leaf):
        return "python"
    return leaf if leaf in _RUNTIME_FAMILIES else ""


def _observe_runtime_executable(
    tc_name: str,
    tc_args: dict | None,
    result: str,
) -> tuple[str, str] | None:
    """Find one runtime executable that just ran in a shell check."""
    if tc_name != "bash" or not isinstance(tc_args, dict):
        return None
    if _UNAVAILABLE_EXIT_RE.search(result):
        return None
    command = tc_args.get("cmd")
    if not isinstance(command, str):
        return None
    for fragment in split_shell_fragments(command):
        try:
            argv = shlex.split(
                strip_leading_assignments(fragment.text), posix=True
            )
        except ValueError:
            continue
        if not argv:
            continue
        family = _runtime_family(argv[0])
        if family:
            return family, argv[0]
    return None


def _runs_direct_executable(
    tc_name: str,
    tc_args: dict | None,
    result: str,
) -> bool:
    """Return whether a shell check ran an explicit executable path."""
    if tc_name != "bash" or not isinstance(tc_args, dict):
        return False
    if _UNAVAILABLE_EXIT_RE.search(result):
        return False
    command = tc_args.get("cmd")
    if not isinstance(command, str):
        return False
    for fragment in split_shell_fragments(command):
        try:
            argv = shlex.split(
                strip_leading_assignments(fragment.text), posix=True
            )
        except ValueError:
            continue
        if not argv:
            continue
        executable = argv[0]
        leaf = executable.rsplit("/", 1)[-1]
        if (
            executable.startswith(("/", "./", "../"))
            and leaf not in _NON_CHECK_EXECUTABLES
        ):
            return True
    return False


def observed_component_runner_base_cmd(
    state: GuardrailState,
    runner: str,
) -> str:
    """Reuse a runtime executable already demonstrated in this revision."""
    executable = state.post_mutation_observed_runtime_executable
    family = state.post_mutation_observed_runtime_family
    if not executable or not family:
        return ""
    quirk = load_run_tests_quirk_for_runner(runner)
    try:
        base = shlex.split(quirk.base_cmd, posix=True)
    except ValueError:
        return ""
    if not base or _runtime_family(base[0]) != family:
        return ""
    base[0] = executable
    return shlex.join(base)


def _safe_source_paths(root: Path, raw_paths: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_paths:
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            resolved = (root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file() and resolved not in paths:
            paths.append(resolved)
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
    for test_id in sorted(
        state.pretest_failing_tests | state.pretest_passing_tests
    ):
        raw = test_id.split("::", 1)[0]
        candidate = (root / raw).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if (
            candidate.name in names
            and candidate.is_file()
            and candidate not in candidates
        ):
            candidates.append(candidate)
    return candidates


def _walk_named_files(
    root: Path,
    names: set[str],
    *,
    ignore_policy: Any,
) -> list[Path]:
    candidates: list[Path] = []
    for directory, dir_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        base = Path(directory)
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
            candidates.append(path.resolve(strict=False))
            if len(candidates) > 64:
                return []
    return candidates


def _select_unique_candidate(source: Path, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    source_parts = source.parent.parts

    def score(candidate: Path) -> tuple[int, int, int]:
        common = 0
        for left, right in zip(source_parts, candidate.parent.parts):
            if left != right:
                break
            common += 1
        distance = len(source_parts) + len(candidate.parent.parts) - 2 * common
        test_dir = int(any("test" in part.lower() for part in candidate.parent.parts))
        return common, test_dir, -distance

    ranked = sorted(candidates, key=lambda path: (score(path), str(path)), reverse=True)
    best_score = score(ranked[0])
    best = [path for path in ranked if score(path) == best_score]
    return best[0] if len(best) == 1 else None


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


def observe_post_mutation_verification(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
    source_write_paths: tuple[str, ...] = (),
    **_: Any,
) -> None:
    """Arm one automatic component run after repeated custom checks."""
    if gate_blocked:
        return
    if tc_name in MUTATION_TOOLS or _is_bash_write_like(tc_name, tc_args):
        if not is_error_result(result):
            state.post_mutation_non_test_bash_count = 0
            state.post_mutation_verification_gate_armed = False
            state.formal_verification_passed_since_mutation = False
            state.post_mutation_automatic_verification_attempted = False
            state.post_mutation_automatic_verification_target = ""
            state.post_mutation_source_paths = tuple(source_write_paths)
            state.post_mutation_observed_runtime_family = ""
            state.post_mutation_observed_runtime_executable = ""
            state.post_mutation_automatic_verification_unavailable = False
        return
    if not state.has_mutated:
        return
    if _is_test_command(tc_name, tc_args):
        state.post_mutation_non_test_bash_count = 0
        state.post_mutation_verification_gate_armed = False
        passed = verification_result_passed(tc_name, result)
        state.formal_verification_passed_since_mutation = passed
        state.verified_since_mutation = passed
        if not verification_runner_unavailable(result):
            state.post_mutation_automatic_verification_unavailable = False
        return
    if state.post_mutation_automatic_verification_attempted:
        return
    if tc_name not in {"bash", "exec_cell"}:
        return
    observed_runtime = _observe_runtime_executable(tc_name, tc_args, result)
    if observed_runtime is None and not _runs_direct_executable(
        tc_name, tc_args, result
    ):
        return
    if observed_runtime is not None:
        (
            state.post_mutation_observed_runtime_family,
            state.post_mutation_observed_runtime_executable,
        ) = observed_runtime
    state.post_mutation_non_test_bash_count += 1
    threshold = int(
        getattr(cfg, "post_mutation_verification_gate_after", 0) or 0
    )
    if threshold > 0 and state.post_mutation_non_test_bash_count >= threshold:
        state.post_mutation_verification_gate_armed = True
