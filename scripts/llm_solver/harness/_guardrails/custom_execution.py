"""Eligibility and executable reuse from tool-owned completed shell records."""
from pathlib import Path
import re
import shlex

from ...language_quirks import load_run_tests_quirk_for_runner
from ..sandbox import container_mode
from ..task_path import active_task_host_root


_RUNTIME_FAMILIES = {
    "bun", "cargo", "ctest", "deno", "dotnet", "go", "java", "make",
    "node", "npm", "npx", "php", "pnpm", "ruby", "yarn",
}
_NON_CHECK_EXECUTABLES = {
    "cat", "cp", "diff", "echo", "env", "file", "find", "git", "grep", "head",
    "ls", "mkdir", "mv", "pwd", "rg", "sed", "stat", "tail", "tee",
    "touch", "wc", "which",
}


def _runtime_family(executable):
    leaf = executable.rsplit("/", 1)[-1]
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", leaf):
        return "python"
    return leaf if leaf in _RUNTIME_FAMILIES else ""


def completed_custom_execution(metadata, cwd=None):
    """Establish a completed custom invocation, not its semantic test coverage."""
    facts = metadata or {}
    submission = facts.get("shell_submission")
    if (not facts.get("executed") or not facts.get("exit_status_known")
            or not isinstance(facts.get("exit_status"), int)
            or facts["exit_status"] in {126, 127} or facts.get("timed_out")
            or facts.get("security_blocked_stage") or not isinstance(submission, dict)
            or not isinstance(submission.get("executable"), str)):
        return False
    if cwd and submission.get("task_cwd") != (active_task_host_root(cwd) or str(Path(cwd).resolve())):
        return False
    status = facts.get("verification_status")
    if status in {"custom_passed", "custom_failed"}:
        return True
    executable = submission["executable"]
    return bool(status == "not_a_check" and (
        _runtime_family(executable) or
        (executable.startswith(("/", "./", "../"))
         and executable.rsplit("/", 1)[-1] not in _NON_CHECK_EXECUTABLES)
    ))


def _observe_runtime_executable(metadata):
    """Retain a completed request's absolute spelling, not a PATH resolution."""
    record = metadata["shell_submission"]
    executable = record["executable"]
    family = _runtime_family(executable)
    return (family, executable) if family and record.get("reusable_spelling") else None


def observed_component_runner_base_cmd(state, runner, *, cwd=None, cfg=None):
    """Reuse the requested spelling only in its recorded task/backend binding."""
    executable = state.post_mutation_observed_runtime_executable
    family = state.post_mutation_observed_runtime_family
    binding = state.post_mutation_observed_runtime_binding
    from ..sandbox.policy import sandbox_execution_kwargs
    backend = sandbox_execution_kwargs(cfg)['sandbox_backend'] if cfg is not None else 'bwrap'
    if (not executable or not family or not binding or not cwd
            or binding.get("task_cwd") != (active_task_host_root(cwd) or str(Path(cwd).resolve()))
            or binding.get("container") != (container_mode() or "")
            or binding.get("backend") != backend):
        return ""
    quirk = load_run_tests_quirk_for_runner(runner)
    try:
        base = shlex.split(quirk.base_cmd)
    except ValueError:
        return ""
    if not base or _runtime_family(base[0]) != family:
        return ""
    base[0] = executable
    return shlex.join(base)
