"""Advice grounded in runner requests and delivered, revision-bound excerpts."""
import hashlib
from pathlib import Path

from .state import PASS, Decision


def observe_test_file_read(state, cfg, *, gate_blocked, result="", execution_metadata=None, **kwargs):
    """Record actual file excerpts without guessing test names or understanding."""
    if not getattr(cfg, "test_read_warn_after", 0) or gate_blocked:
        return
    metadata = execution_metadata or {}
    record = metadata.get("inspection_evidence")
    if (not metadata.get("executed") or metadata.get("security_blocked_stage")
            or not isinstance(record, dict)
            or record.get("namespace") not in {'local_filesystem', 'task_execution'}):
        return
    path = record["path"]
    admitted_prefix = result[:record["admitted_output_chars"]]
    if hashlib.sha256(admitted_prefix.encode()).hexdigest() != record["admitted_output_sha256"]:
        record = {**record, "delivery": "unknown"}
    previous = state.inspected_files.get(path, {})
    same_revision = (previous.get("sha256") == record["sha256"]
                     and previous.get('namespace') == record['namespace']
                     and previous.get('task_view') == record.get('task_view'))
    intervals = list(previous.get("intervals", [])) if same_revision else []
    if record["delivery"] == "selected_excerpt" and record["line_count"]:
        intervals.append((record["start_line"], record["start_line"] + record["line_count"] - 1))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    state.inspected_files[path] = {**record, "intervals": merged}


def _workspace_coverage(state, cfg, cwd, request, target):
    """Observe a permitted task spelling, not a runner's per-test selection.

    A selector that is not a literal local path remains unknown. We never map
    container paths through the host, nor infer that a child covers its parent.
    """
    from ..task_path import active_task_host_root
    try:
        host_root = (active_task_host_root(cwd) or str(Path(cwd).resolve())) if cwd else None
    except (OSError, RuntimeError, ValueError):
        return "unknown", "workspace binding cannot be resolved", "", False
    if not cwd or request.get("workspace_cwd") != host_root:
        return "unknown", "execution namespace or working directory is unbound", "", False
    from ..task_path import active_task_files, resolve_task_path, TaskPath, NativeUnreadableMatcher
    files = active_task_files(cwd)
    namespace = request.get('workspace_namespace', 'local_filesystem')
    if (namespace == 'task_execution' and (files is None
            or request.get('workspace_task_view') != files.binding)) or (
            namespace == 'local_filesystem' and files is not None):
        return 'unknown', 'execution namespace differs from the current inspection view', '', False
    if namespace not in {'local_filesystem', 'task_execution'}:
        return 'unknown', 'execution namespace is unknown', '', False
    try:
        root = resolve_task_path(cwd, '.')
        path = resolve_task_path(cwd, target)
    except (OSError, ValueError, RuntimeError):
        return "unknown", "workspace path cannot be resolved", "", False
    if path != root and root not in path.parents:
        return "unknown", "selector is outside the task workspace", "", False
    from ..project_instructions import _UnreadableMatcher
    from ..sandbox.ignore_policy import active_ignore_policy
    from ..tools import _bash_unreadable_paths
    try:
        matcher = NativeUnreadableMatcher if isinstance(root, TaskPath) else _UnreadableMatcher
        if matcher(root, _bash_unreadable_paths(cwd, cfg)).blocks(path):
            return "unknown", "selector is not permitted for inspection", "", False
        policy = active_ignore_policy(cwd)
        if policy is not None:
            policy.require_visible(path, is_dir=path.is_dir())
        if path.is_dir():
            return "directory spelling", "file excerpts do not establish suite coverage", "", False
        if not path.is_file():
            return "unknown", "no matching local file; selector may be missing or non-file", "", False
        record = state.inspected_files.get(str(path))
        if not record:
            return "file spelling", "no matching recorded excerpt; other inspection is unknown", "", False
        if (record.get('namespace') != namespace or
                record.get('task_view') != request.get('workspace_task_view')):
            return 'file spelling', 'recorded excerpt belongs to a different inspection view', '', False
        # Hash only a previously inspected file, within the invocation's clock.
        with path.open("rb") as stream:
            revision = hashlib.file_digest(stream, "sha256").hexdigest()
        if revision != record["sha256"]:
            return "file spelling", "recorded excerpt is from a different file revision", revision, False
        intervals = record["intervals"]
        if not intervals:
            return "file spelling", "delivered content extent is unknown or empty", revision, False
        covered = sum(end - start + 1 for start, end in intervals)
        complete = covered == record["total_lines"]
        ranges = ", ".join(str(start) if start == end else f"{start}-{end}" for start, end in intervals)
        text = f"{'complete' if complete else 'partial'} recorded excerpt: lines {ranges} of {record['total_lines']} in the current revision"
        return "file spelling", text, revision, complete
    except (OSError, ValueError, RuntimeError):
        return "unknown", "workspace inspection unavailable", "", False


def test_read_ladder(state, cfg, *, tc_name, gate_blocked, execution_metadata=None,
                     cwd="", **kwargs):
    """Count completed invocations per requested selector and inspection state."""
    warn_after = int(getattr(cfg, "test_read_warn_after", 0) or 0)
    metadata = execution_metadata or {}
    if (warn_after <= 0 or tc_name not in {"bash", "run_tests"} or gate_blocked
            or not metadata.get("executed") or metadata.get("security_blocked_stage")
            or metadata.get("verification_status") not in {"passed", "failed"}
            or metadata.get("timed_out") or not metadata.get("exit_status_known")):
        return PASS
    request = metadata.get("runner_request")
    if (not isinstance(request, dict) or not request.get("family")
            or not request.get("check_intent")):
        return PASS
    targets = request.get("targets", [])
    if request.get("target_status") != "recorded" or not targets:
        return PASS
    messages = []
    for target in dict.fromkeys(targets):
        kind, coverage, revision, complete = _workspace_coverage(state, cfg, cwd, request, target)
        key = (request["family"], request.get("workspace_cwd", ""), target)
        signature = (kind, coverage, revision)
        previous = state.test_target_observations.get(key, {})
        same = previous.get("signature") == signature
        count = previous.get("count", 0) + 1 if same else 1
        warned = same and previous.get("warned", False)
        emit = not complete and not warned and count >= warn_after
        state.test_target_observations[key] = {
            "signature": signature, "count": count, "warned": warned or emit,
        }
        if emit:
            messages.append(cfg.test_read_nudge.format(
                count=count, target=target, kind=kind, coverage=coverage,
                runner=request["family"],
            ))
    return Decision.warn("\n".join(messages), reason="test_read_ladder") if messages else PASS
