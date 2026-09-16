"""Observe entry changes across tool dispatch in the selected task namespace.

This is a metadata comparison, not a content hash or a filesystem watcher.
It does not attribute concurrent writes or prove that unchanged metadata
means unchanged bytes. No task language or command syntax selects the files.
"""
from functools import wraps
import os
import subprocess
import time

from .action_metadata import action_metadata
from .task_path import active_task_files
from .task_file_runtime import make_task_files
from .time_budget import BudgetExhausted, command_time_budget, execution_deadline, remaining_before
from .tool_specs import ACTION_WRITE_LIKE_TOOL_NAMES

OBSERVED_TOOLS = ACTION_WRITE_LIKE_TOOL_NAMES | {
    "bash", "bash_poll", "bash_kill", "exec_cell", "run_tests",
}


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
        before = None
        try:
            files = active_task_files(cwd)
            if files is None:
                # Only an intentionally local dispatcher has no bound reader.
                if kwargs["cfg"].sandbox_bash:
                    raise OSError("selected task file view is unavailable")
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
                      if entry[0] != b'd' and path not in {'.git/index', '.git/config', '.git/info/exclude'}}
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
                             if entry[0] != b'd' and path not in {'.git/index', '.git/config', '.git/info/exclude'}}
                    # A registered artifact may first be created by this call.
                    def admitted(path):
                        return not any(path == p or path.startswith(p + "/")
                                       for p in excluded)
                    changed = sorted(path for path in before.keys() | after.keys()
                                     if admitted(path) and before.get(path) != after.get(path))
                    record.update(status="changed" if changed else "unchanged_metadata",
                                  changed_paths=changed, excluded_paths=excluded)
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
        execution_metadata.get("executed", True) and record.get("changed_paths"))


def apply_observed_metadata(metadata, execution_metadata):
    record = (execution_metadata or {}).get("file_changes")
    if record is None:
        return
    metadata["predicted_write_like"] = metadata.get("write_like", False)
    metadata["predicted_source_write_like"] = metadata.get("source_write_like", False)
    metadata["write_like"] = metadata["source_write_like"] = observed_mutation(execution_metadata)
    metadata["source_write_paths"] = record["changed_paths"]
