"""Describe requested runner interfaces from complete literal command syntax.

Descriptors supply CLI grammar, never a task's runner identity or result.
Unknown scripts/options stay unknown. No source files or programs are queried
while projecting a historical command.
"""
import functools
import hashlib
import re
import shlex
import tomllib

from ..language_quirks import FORMATS_DIR
from .command_redirect import split_shell_fragments, strip_leading_assignments
from .shell_verification import _literal_simple_command


@functools.lru_cache(maxsize=1)
def _grammars():
    rules = []
    for path in sorted(FORMATS_DIR.glob("*.toml")):
        raw = path.read_bytes()
        spec = tomllib.loads(raw.decode("utf-8"))
        for rule in spec.get("diagnostic_invocations", []):
            if not isinstance(rule.get("family"), str) or not rule["family"]:
                raise ValueError(f"{path}: diagnostic invocation requires a family")
            for key in ("prefix", "flags", "value_options", "non_targets"):
                values = rule.get(key, [])
                if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                    raise ValueError(f"{path}: diagnostic {key} must be a string list")
            if not rule.get("prefix"):
                raise ValueError(f"{path}: diagnostic invocation requires a prefix")
            rules.append((rule, [re.compile(part) for part in rule["prefix"]],
                          hashlib.sha256(raw).hexdigest()))
    return tuple(rules)


def _targets(args, rule):
    if args and rule.get("opaque_arguments", False):
        return [], "opaque_arguments"
    targets = []
    flags = set(rule.get("flags", []))
    value_options = set(rule.get("value_options", []))
    positional = False
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if not positional and arg == "--":
            if index < len(args) and not rule.get("literal_targets_after_separator", False):
                return [], "unsupported_forwarded_arguments"
            positional = True
        elif not positional and arg.startswith("-"):
            option, separator, _value = arg.partition("=")
            if option in value_options:
                if not separator:
                    if index == len(args) or args[index].startswith("-"):
                        return [], "unknown_option_value"
                    index += 1
            elif arg not in flags:
                return [], "unsupported_option"
        elif arg not in rule.get("non_targets", []):
            targets.append(arg)
    return targets, "recorded" if targets else "unspecified"


def describe_runner_invocations(command):
    """Return requested interfaces and target spellings, not execution claims."""
    from ._shell_patterns import _has_heredoc
    requests = []
    for fragment in split_shell_fragments(command):
        if _has_heredoc(fragment.text):
            break  # The remaining fragments may be heredoc data, not commands.
        if not _literal_simple_command(fragment.text):
            continue
        text = strip_leading_assignments(fragment.text)
        try:
            argv = shlex.split(text)
        except ValueError:
            continue
        if argv and argv[0].rsplit("/", 1)[-1] == "env":
            argv = argv[1:]
            while argv and (argv[0] in {"-i", "--ignore-environment"}
                            or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0])):
                argv = argv[1:]
            if argv and argv[0] == "--":
                argv = argv[1:]
        if not argv:
            continue
        argv[0] = argv[0].rsplit("/", 1)[-1]
        matches = [(rule, prefix, digest) for rule, prefix, digest in _grammars()
                   if len(argv) >= len(prefix)
                   and all(pattern.fullmatch(value) for pattern, value in zip(prefix, argv))]
        if len(matches) != 1:
            continue  # Ambiguous grammar cannot establish a family.
        rule, prefix, digest = matches[0]
        targets, status = _targets(argv[len(prefix):], rule)
        requests.append({
            "family": rule["family"], "basis": "invocation_syntax",
            "targets": targets, "target_status": status,
            "command_sha256": hashlib.sha256(fragment.text.encode()).hexdigest(),
            "descriptor_sha256": digest,
        })
    return requests


def recorded_runner_requests(event):
    """Prefer tool-owned selection; never parse identity out of tool output."""
    record = event.get("runner_request")
    if isinstance(record, dict) and 'requests' in record:
        records = record['requests']
        if not isinstance(records, list):
            return []
        return [request for item in records if isinstance(item, dict) and 'requests' not in item
                for request in recorded_runner_requests({'runner_request': item})]
    if isinstance(record, dict) and record.get("basis") in {
        "runtime_selection", "explicit_configuration", "custom_command", "invocation_syntax",
    }:
        if (isinstance(record.get("family"), str)
                and isinstance(record.get("targets"), list)
                and all(isinstance(t, str) for t in record["targets"])
                and isinstance(record.get("target_status"), str)
                and re.fullmatch(r"[0-9a-f]{64}", str(record.get("command_sha256", "")))):
            return [dict(record)]
        return []
    if event.get("tool_name") not in {"bash", "run"}:
        return []
    from ._shell_patterns import command_from_summary
    return describe_runner_invocations(command_from_summary(str(event.get("args_summary") or "")))


def runner_request_record(command):
    """Retain all parsed request facts without retaining another command copy."""
    requests = describe_runner_invocations(command)
    return requests[0] if len(requests) == 1 else {
        'requests': requests, 'command_sha256': hashlib.sha256(command.encode()).hexdigest(),
    }


def executed_runner_request(command):
    """Project only a single literal command actually submitted to the shell.

    Execution status is separate metadata. Compound shell fragments cannot
    establish which runner executed or the working directory it used.
    """
    if not _literal_simple_command(command):
        return None
    requests = describe_runner_invocations(command)
    return requests[0] if len(requests) == 1 else None


def bind_runner_workspace(request, command, cwd, *, sandbox=True, backend="bwrap"):
    """Record the selected view for a literal task command's path spellings.

    This describes the dispatcher view, not a runner's internal file selection
    or a guarantee that mounts stayed unchanged throughout execution.
    """
    from pathlib import Path
    from .sandbox import container_mode
    from .shell_verification import shell_verification_status
    from .task_path import active_task_files, active_task_host_root
    files = active_task_files(cwd) if sandbox else None
    if files is not None and _literal_simple_command(command):
        return {**request, 'workspace_cwd': active_task_host_root(cwd),
                'workspace_namespace': 'task_execution',
                'workspace_task_view': dict(files.binding),
                'check_intent': shell_verification_status(command, 0) == 'passed'}
    local = (container_mode() in {None, "ambient"}
             and (not sandbox or backend == "bwrap")
             and _literal_simple_command(command))
    return {**request, "workspace_cwd": str(Path(cwd).resolve()) if local else "",
            'workspace_namespace': 'local_filesystem' if local else 'unknown',
            "check_intent": (_literal_simple_command(command)
                             and shell_verification_status(command, 0) == "passed")}


def describe_shell_submission(command, cwd, *, backend="bwrap"):
    """Record the literal executable spelling submitted, not a resolved binary.

    Compound commands cannot establish which executable ran. Inline environment
    assignments may support a completed request but cannot authorize later reuse
    without preserving their environment. Arguments stay out of this receipt.
    """
    from pathlib import Path
    from .sandbox import container_mode
    from .task_path import active_task_host_root
    if not _literal_simple_command(command):
        return None
    stripped = strip_leading_assignments(command)
    try:
        argv = shlex.split(stripped)
    except ValueError:
        return None
    if not argv:
        return None
    return {"executable": argv[0], "task_cwd": active_task_host_root(cwd) or str(Path(cwd).resolve()),
            "container": container_mode() or "", "backend": backend,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "reusable_spelling": stripped.strip() == command.strip() and argv[0].startswith("/")}
