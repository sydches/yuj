"""Observe entry changes across tool dispatch in the selected task namespace.

This is a metadata comparison, not a content hash or a filesystem watcher.
It does not attribute concurrent writes or prove that unchanged metadata
means unchanged bytes. No task language or command syntax selects the files.
"""
from functools import wraps
import functools
import os
import re
import subprocess
import time

from .action_metadata import action_metadata
from .operation_timing import measured
from .task_path import active_task_files
from .task_file_runtime import make_task_files
from .time_budget import BudgetExhausted, command_time_budget, execution_deadline, remaining_before
from .tool_specs import ACTION_WRITE_LIKE_TOOL_NAMES

OBSERVED_TOOLS = ACTION_WRITE_LIKE_TOOL_NAMES | {
    "bash", "bash_poll", "bash_kill", "exec_cell", "run_tests",
}


def _observed_check_attempt(name, arguments, metadata):
    """Recognize output creation without promoting a failed check to a pass."""
    if (name not in {'bash', 'run_tests'} or not metadata.get('executed')
            or not metadata.get('exit_status_known') or metadata.get('exit_status') is None
            or metadata.get('timed_out') or metadata.get('security_blocked_stage')):
        return False
    status = metadata.get('verification_status')
    if status not in {'passed', 'failed', 'shell_unresolved'}:
        return False
    if status == 'shell_unresolved' and metadata['exit_status'] == 0:
        return False
    from .shell_verification import shell_verification_status
    command = (metadata.get('runner_command', '') if name == 'run_tests'
               else str((arguments or {}).get('cmd') or ''))
    # This asks only which commands were submitted. It does not change the
    # actual result or assume that a nonzero pipeline identifies the failing stage.
    return shell_verification_status(command, 0, pipefail=True, allow_custom=False) == 'passed'


@functools.lru_cache(maxsize=1)
def _declared_output_patterns():
    from ..language_quirks import FORMATS_DIR, _load_runner_quirk_dict
    patterns = [pattern for path in sorted(FORMATS_DIR.glob('*.toml'))
                for pattern in _load_runner_quirk_dict(path.stem).get('artifact_command_patterns', ())]
    return re.compile('|'.join(f'(?:{pattern})' for pattern in patterns) or r'(?!)', re.I | re.X)


def _observed_output_attempt(name, arguments, metadata):
    if (name != 'bash' or not metadata.get('executed')
            or not metadata.get('exit_status_known') or metadata.get('exit_status') is None
            or metadata.get('timed_out') or metadata.get('security_blocked_stage')):
        return False
    from .shell_verification import shell_output_invocation
    return shell_output_invocation(str((arguments or {}).get('cmd') or ''), _declared_output_patterns())


@measured("changes.inventory")
def _inventory(files, cwd, owned_paths, *, include_directories=False):
    remaining_before(execution_deadline())
    excluded = [".git"]
    identities = files.observe_host_entries({path: os.path.join(cwd, path) for path in owned_paths})
    for path, identity in identities.items():
        if identity["relation"] == "same_entry":
            excluded.append(path)
    args = [files._utility("find"), "-P", ".", "("]
    for index, path in enumerate(excluded):
        if index:
            args.append("-o")
        # find -path takes a pattern, even when passed as a literal argv.
        literal = "".join("\\" + c if c in "\\*?[]" else c for c in path)
        args.extend(["-path", "./" + literal])
    args.extend([")", "-prune", "-o"])
    if not include_directories:
        args.extend(["!", "-type", "d"])
    args.extend(["-printf",
                 r"%p\0%y\0%m\0%s\0%T@\0%C@\0%i\0%l\0"])
    # Include the task index in the same inventory call. It is evidence for
    # tracked inputs, not a source change or permission to inspect Git config.
    script = '''cd -- "$1" && shift || exit
inventory=("$1" "$2" "$3"); shift 3
if [[ -d .git && ! -L .git && -f .git/index && ! -L .git/index ]]; then
    inventory+=(./.git/index)
fi
# reuse_inputs
exec "${inventory[@]}" "$@"'''
    if include_directories:
        script = script.replace('# reuse_inputs', '''for path in .git/config .git/info/exclude; do
    if [[ -e "$path" || -L "$path" ]]; then inventory+=("./$path"); fi
done''')
    result = files.run(script,
                       [str(files.root), *args], None)
    if result.returncode:
        raise OSError("native file inventory did not complete")
    fields = result.stdout.split(b"\0")
    if fields.pop() != b"" or len(fields) % 8:
        raise ValueError("native file inventory has an invalid shape")
    return {os.fsdecode(fields[i]).removeprefix('./'): tuple(fields[i + 1:i + 8])
            for i in range(0, len(fields), 8)}, excluded


def observed_dispatch(function):
    """Attach effects after execution, including commands that return errors."""
    @wraps(function)
    def wrapped(name, arguments, **kwargs):
        metadata = kwargs.get("execution_metadata")
        from .read_reuse import active_reuse, contained_search_paths, inventory_stamp, source_search
        reuse = active_reuse()
        owned = kwargs.pop("observation_owned_paths", (".tool_output", ".solver"))
        if (name not in OBSERVED_TOOLS and not (name == 'grep' and reuse)) or metadata is None:
            return function(name, arguments, **kwargs)
        cwd = kwargs["cwd"]
        record = {"basis": "native_entry_metadata_v1", "status": "unavailable",
                  "tool_call_id": kwargs.get("tool_call_id", ""),
                  "action_sha256": action_metadata(name, arguments)["action_sha256"],
                  "started_monotonic": time.monotonic(), "changed_paths": [],
                  "scope": "task_non_directory_entries_except_git_and_owned_artifacts"}
        metadata["file_changes"] = record
        files = None
        local_view = False
        before = None
        try:
            files = active_task_files(cwd)
            if files is None:
                # Only an intentionally local dispatcher has no bound reader.
                if kwargs["cfg"].sandbox_bash:
                    raise OSError("selected task file view is unavailable")
                local_view = True
                files = make_task_files(cwd, kwargs["cfg"],
                    environment=kwargs.get("effective_env"),
                    allow_login_shell=bool(kwargs.get("allow_login_shell")))
            record["task_binding"] = files.binding
            before, excluded = _inventory(files, cwd, owned, **(
                {'include_directories': True} if reuse is not None else {}))
            record["excluded_paths"] = excluded
            if reuse is not None:
                search = source_search(str(arguments.get('cmd', ''))) if name == 'bash' else None
                paths = search.paths if search else (str(arguments.get('path', '.')),)
                if (contained_search_paths(paths, (str(files.root), cwd), excluded)
                        and (name != 'bash' or search and search.directory in ('.', str(files.root), cwd))):
                    reuse.before = inventory_stamp(before, files.binding)
            before = {path: entry for path, entry in before.items()
                      if entry[0] != b'd' and path not in {'.git/config', '.git/info/exclude'}}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            record["error_kind"] = type(error).__name__
        try:
            try:
                remaining_before(execution_deadline())
            except BudgetExhausted:
                with command_time_budget() as allocation:
                    metadata.update(executed=False, verification_status="budget_exhausted",
                                    exit_status=None, exit_status_known=False, timed_out=False,
                                    execution_budget={**allocation.record, "status": "exhausted"})
                return "ERROR: tool did not start: execution time budget is exhausted."
            return function(name, arguments, **kwargs)
        finally:
            if before is not None:
                try:
                    after, excluded = _inventory(files, cwd, owned, **(
                        {'include_directories': True} if reuse is not None else {}))
                    if reuse is not None:
                        reuse.after = inventory_stamp(after, files.binding)
                    after = {path: entry for path, entry in after.items()
                             if entry[0] != b'd' and path not in {'.git/config', '.git/info/exclude'}}
                    # A registered artifact may first be created by this call.
                    def admitted(path):
                        return not any(path == p or path.startswith(p + "/")
                                       for p in excluded)
                    changed = sorted(path for path in before.keys() | after.keys()
                                     if admitted(path) and before.get(path) != after.get(path))
                    record.update(status="changed" if changed else "unchanged_metadata",
                                  changed_paths=changed, excluded_paths=excluded)
                    record['created_paths'] = [path for path in changed if path not in before]
                    if changed and (_observed_check_attempt(name, arguments, metadata)
                                    or _observed_output_attempt(name, arguments, metadata)):
                        from .project_artifacts import LocalProjectFiles, project_artifact_evidence
                        candidates = [path for path in changed
                                      if (after.get(path) or before[path])[0] == b'f']
                        record['project_artifact_evidence'] = project_artifact_evidence(
                            LocalProjectFiles(cwd) if local_view else files, before, after, candidates)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    record.update(status="unavailable", error_kind=type(error).__name__)
            record["finished_monotonic"] = time.monotonic()
            record["shell_submission"] = metadata.get("shell_submission")
            if name in {"bash", "bash_poll", "bash_kill", "exec_cell", "run_tests"}:
                record["terminal_observed"] = bool(metadata.get("exit_status_known"))
                if not record["terminal_observed"] and record["status"] == "unchanged_metadata":
                    record["status"] = "incomplete"
            if record['status'] in {'changed', 'unavailable', 'incomplete'}:
                # Equal stdout does not establish equal observations of an
                # effectful or unfinished command.
                metadata.pop('observation_receipt', None)

    return wrapped


def observed_mutation(execution_metadata):
    """Return None for old callers without observations; never guess unknowns."""
    record = (execution_metadata or {}).get("file_changes")
    return None if record is None else bool(
        execution_metadata.get("executed", True) and mutation_paths(record))


def mutation_paths(record):
    artifacts = set(record.get("test_artifact_paths", ()))
    return [path for path in record.get("changed_paths", ()) if path not in artifacts]


@functools.lru_cache(maxsize=1)
def _test_artifact_rules():
    from ..language_quirks import FORMATS_DIR, _load_runner_quirk_dict
    rules = []
    for path in sorted(FORMATS_DIR.glob("*.toml")):
        spec = _load_runner_quirk_dict(path.stem)
        patterns = spec.get("generated_artifact_patterns", ())
        if patterns:
            commands = (*spec.get("verification_patterns", ()),
                        *spec.get("custom_check_patterns", ()))
            if not commands:
                continue
            invocation = re.compile("|".join(commands), re.I | re.X)
            rules.append((path.stem, invocation, tuple(re.compile(p) for p in patterns)))
    return tuple(rules)


@measured("changes.classify_outputs")
def classify_test_artifacts(name, arguments, execution_metadata, input_paths, known_artifacts=None):
    """Keep runner outputs observable without making them edited test inputs."""
    record = execution_metadata.get("file_changes")
    if known_artifacts is None:
        known_artifacts = {}
    if not record:
        return
    previous = dict(known_artifacts)
    if record.get('status') in {'unavailable', 'incomplete'}:
        known_artifacts.clear()
    for path in record.get('changed_paths', ()):
        known_artifacts.pop(path, None)
    # Runtime caches can also come from custom checks. Project-specific outputs
    # still require the stricter registered-check evidence collected above.
    # Cache recognition needs a completed invocation, not an attributable test
    # verdict. A trailing wc/head can mask pytest's status while pytest still
    # updates its cache. The runner patterns below identify those outputs.
    if (record.get("status") != "changed" or name not in {"bash", "run_tests"}
            or not execution_metadata.get("executed")
            or not execution_metadata.get("exit_status_known")
            or execution_metadata.get("exit_status") is None
            or execution_metadata.get("timed_out")
            or execution_metadata.get("security_blocked_stage")):
        return
    from ._shell_patterns import matches_command
    command = str((arguments or {}).get("cmd") or "")
    family = (execution_metadata.get("runner_request") or {}).get("family")
    patterns = [pattern for runner, invocation, artifacts in _test_artifact_rules()
                if (name == "run_tests" and family == runner) or matches_command(command, invocation)
                for pattern in artifacts]
    record["test_artifact_paths"] = [
        path for path in record["changed_paths"] if path not in input_paths
        and any(pattern.fullmatch(path) for pattern in patterns)
    ]
    evidence = record.get('project_artifact_evidence') or {}
    if evidence.get('status') != 'available':
        return
    created = set(record.get('created_paths', ()))
    for path, receipt in evidence.get('matches', {}).items():
        if path in input_paths or (path not in created and previous.get(path) != receipt):
            continue
        if path not in record['test_artifact_paths']:
            record['test_artifact_paths'].append(path)
        known_artifacts[path] = receipt


def apply_observed_metadata(metadata, execution_metadata):
    record = (execution_metadata or {}).get("file_changes")
    if record is None:
        return
    metadata["predicted_write_like"] = metadata.get("write_like", False)
    metadata["predicted_source_write_like"] = metadata.get("source_write_like", False)
    metadata["write_like"] = metadata["source_write_like"] = observed_mutation(execution_metadata)
    metadata["source_write_paths"] = mutation_paths(record)
