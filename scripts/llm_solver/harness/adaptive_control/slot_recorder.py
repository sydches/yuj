"""Project the live session's recorded trace events into prefix-only slot facts.

`session._trace_events` already accumulates one `tool_call` event per dispatched
tool call during the run. This module projects those events into the same
diagnostic slot facts the predecessor turn/history instrument uses, prefix-only
(events with `turn_number <= observation_slot`). It is **read-only**: it adds no
recording to the live loop and changes nothing when adaptive control is off.

Reuse: mutation/error facts come from the harness's own `MUTATION_TOOLS` and
`is_error_result`; the action signature is the same primitive shape used for
duplicate detection. Imports fall back to stdlib-only approximations so this
module stays importable in the isolated test harness (no model-client deps).
"""
from __future__ import annotations

import hashlib
import re
import json

SLOT_PROJECTION_VERSION = "runner_facts_v1"

try:  # real harness constants in production
    from .._guardrails.extractors import MUTATION_TOOLS as _MUTATION_TOOLS
except Exception:  # noqa: BLE001 - keep import-safe under the stub-parent test harness
    _MUTATION_TOOLS = frozenset({
        "write", "edit", "notebook_edit", "structural_edit", "str_replace", "create",
        "apply_patch", "udiff", "insert",
    })

try:
    from ..._shared.classification import is_error_result as _is_error_result
except Exception:  # noqa: BLE001
    def _is_error_result(result) -> bool:  # type: ignore
        return isinstance(result, str) and result.startswith("ERROR:")

_OP_KIND = {
    "bash": "RUN", "run": "RUN",
    "read": "READ", "cat": "READ",
    "grep": "SEARCH", "glob": "SEARCH", "search": "SEARCH",
    "structural_search": "SEARCH",
    "edit": "EDIT", "notebook_edit": "EDIT", "structural_edit": "EDIT", "write": "EDIT",
    "str_replace": "EDIT", "create": "EDIT",
    "apply_patch": "EDIT", "udiff": "EDIT", "insert": "EDIT",
    "done": "SUBMIT", "submit": "SUBMIT",
}
from .._shell_patterns import command_from_summary, matches_command
# read-only commands that merely MENTION tests (grep a test name, cat a test
# file) are not verification; anchored at the command start.
_READ_ONLY_CMD_RE = re.compile(
    r"^\s*(?:cmd=)?[\'\"]?\s*(?:grep|cat|head|tail|ls|find|wc|rg|sed\s+-n)\b")
# Match a registered runner, then use the exit marker and runner summary.
# Treat output without a verdict as unknown.
_TEST_NON_RUN_FLAGS = ("--help", "--collect-only", "--version", "--co")
# Verdict markers. Python: "N failed"/FAILED/ERRORS, "N passed"/OK. Additive
# non-Python shapes: go `--- FAIL:`/`--- PASS:`, cargo `test result: FAILED/ok`,
# jest `✕`/`✓` and `Tests: N failed/passed`.
_TEST_FAIL_OUT_RE = re.compile(
    r"\b\d+ (?:failed|error)|\bFAILED\b|\bERRORS\b"
    r"|^--- FAIL:|^FAIL\b|test result:\s*FAILED|^\s*✕|Tests:.*\bfailed\b",
    re.MULTILINE)
_TEST_PASS_OUT_RE = re.compile(
    r"\b\d+ passed\b|^OK(?:\s*\(.*\))?\s*$"
    # go pass line is `ok <pkg> <duration>` (require the timing so prose
    # "ok ..." never matches); cargo `test result: ok`; jest summary.
    r"|^--- PASS:|^ok\s+\S+\s+[\d.]+m?s\b|test result:\s*ok|^\s*✓|Tests:.*\bpassed\b",
    re.MULTILINE)
_EXIT_CODE_RE = re.compile(r"\[exit code: (\d+)\]\s*$")

# --- Slot projection ---------------------------------------------------------
# Treat runner failure verdicts as fail evidence. Bare tracebacks, scaffold
# failures, and probe crashes do not open red-test state.
_V2_FAIL_RE = re.compile(
    r"\b\d+ failed\b|\bFAILED \(|^FAILED \S+|\S+ FAILED\b|^FAIL: "
    # Additive: go `--- FAIL: TestX`, go pkg `FAIL\tpkg`, cargo `test result: FAILED`.
    r"|^--- FAIL: \w+|^FAIL\s|test result:\s*FAILED", re.MULTILINE)
# Loader and collection errors may carry verdict tokens although no test ran.
_V3_LOADER_ERR_RE = re.compile(
    r"unittest\.loader\._FailedTest|Failed to import test module"
    r"|collected 0 items|no tests ran|file or directory not found")
# Match scratch-script paths.
_V3_SCRIPT_PATH_RE = re.compile(r"(/tmp/[\w./-]+\.py)")
# Match verbose green IDs from Django and unittest.
_V3_VERBOSE_OK_RE = re.compile(r"^(\w+) \([\w.]+\)\s*(?:\.\.\.\s*)?ok\b", re.MULTILINE)
_V3_VERBOSE_FAIL_RE = re.compile(
    r"^(\w+) \(([\w.]+)\)\s*(?:\.\.\.\s*)?FAIL\b",
    re.MULTILINE,
)
_V3_SUITE_GREEN_RE = re.compile(r"Ran \d+ tests?[\s\S]{0,200}?^OK(?:\s*\(.*\))?\s*$", re.MULTILINE)
# Self-declared simulations and text checks do not verify behavior.
_V3_MOCK_RE = re.compile(r"(?i)\bmock|without running|fix is present")
_V3_UNITTEST_RUN_RE = re.compile(
    r"Ran \d+ tests?[\s\S]{0,400}?(?:^OK(?:\s*\(.*\))?\s*$|^FAILED \()",
    re.MULTILINE,
)
# Match explicit pass markers, including inline executions.
_V2_PASS_RE = re.compile(
    r"\b\d+ passed\b|^OK(?:\s*\(.*\))?\s*$|All tests passed|ALL TESTS PASSED"
    r"|\bPASSED\b|^\s*(?:-\s*)?PASS\b|\bPASS:|Success!|✓"
    # Additive: go `--- PASS`/`ok <pkg> <duration>`, cargo `test result: ok`.
    r"|^--- PASS:|^ok\s+\S+\s+[\d.]+m?s\b|test result:\s*ok", re.MULTILINE)
# Static checks do not verify behavior.
_V2_STATIC_RE = re.compile(r"py_compile|Syntax OK|ast\.parse")
# Environment failures do not show a behavior failure. This keeps pytest import
# errors from opening red-test state on a verified run.
_V2_ENV_FAIL_RE = re.compile(
    r"ImportError|ModuleNotFoundError|No module named|Permission denied"
    r"|PermissionError|No matching distribution")
# Match test IDs.
_V2_TEST_ID = r"(?:[\w\[\]./-]+::)+[\w\[\]./-]+"
# ^FAIL: capture restricted to test-shaped names: "FAIL: Expected 1 auth.E003
# error..." prose must not become a named red test that cannot clear.
_V2_FAILED_ID_RE = re.compile(
    rf"FAILED ({_V2_TEST_ID})|({_V2_TEST_ID}) FAILED"
    r"|^FAIL: \w+\s+\(([\w.]+)\)|^FAIL: (test\w+)\b"
    r"|^_+\s+((?:\w+\.)?test\w+)\s+_+$"
    # Additive test-id capture: go `--- FAIL: TestX`, cargo `test mod::x ... FAILED`.
    r"|^--- FAIL: (\w+)"
    r"|^test (\S+) \.\.\. FAILED",
    re.MULTILINE)
_V2_PASSED_ID_RE = re.compile(
    rf"({_V2_TEST_ID}) PASSED|PASSED ({_V2_TEST_ID})"
    r"|^--- PASS: (\w+)"
    r"|^test (\S+) \.\.\. ok",
    re.MULTILINE)
# Match exclusion context.
_V2_EXCLUSION_RE = re.compile(r"-k\s+['\"]?not\b|--ignore=|--deselect|\bstash\b")
_GIT_STASH_RE = re.compile(r"\bgit\s+stash\b(?!\s+pop\b)")
_GIT_STASH_POP_RE = re.compile(r"\bgit\s+stash\s+pop\b|\bgit\s+stash\s+apply\b")
# Match mutation commands such as sed -i, cp, install, and subprocess calls
# when they target a path outside /tmp.
_V2_MUT_CMD_RE = re.compile(r"\bsed\s+-i\b|\binstall\b|\bcp\s|\bmv\s")
_V2_NON_TMP_PATH_RE = re.compile(r"(?:^|[\s='\"])/(?!tmp[/\s])[\w.-]+(?:/[\w.-]+)+\.\w+")
_V2_PERM_FAIL_RE = re.compile(r"PermissionError|Permission denied|Operation not permitted")
_TMP_HEREDOC_RE = re.compile(r"^\s*(?:cmd=)?[\"']?cat\s+>\s+/tmp/[^;\n]+<<")
# Multilingual crash-frame extraction (v3.1 inline-behavioral reds).
# A crash marker gates it; the frame regex captures source paths
# in any implementation language via the shared source-extension set.
from ..bash_write_classification import _SOURCE_EXT_RE as _SRC_EXT
_CRASH_MARKER_RE = re.compile(
    r"Traceback|panic:|thread '.*' panicked|^\s*at\s+\S|Exception|goroutine\s+\d+")
_CRASH_FRAME_RE = re.compile(
    r'(?:File "|\bat\s+(?:[\w.$<>]+\s+\()?|[\s\t])'          # py / js-at / bare-tab frame
    rf'(/[^\s":()]+\.(?:{_SRC_EXT}))'                       # absolute source path
    r'(?::\d+)?')                                            # optional :line
_NO_OUTPUT_MARKERS = {"", "(command produced no output)"}
_FORMAT_PATH_RE = r"format\s*=\s*['\"]{path}['\"]"


def _op_kind(tool_name: str) -> str:
    return _OP_KIND.get(tool_name, "RUN" if tool_name else "")


def action_signature(tool_name: str, args_summary: str) -> str:
    """Normalized (tool, args) action key — the repeat primitive, prefix-safe."""
    return hashlib.sha1(f"{tool_name}\x00{args_summary}".encode("utf-8")).hexdigest()[:16]


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _drop_bare_duplicate_ids(ids: set[str]) -> list[str]:
    out = set(ids)
    for item in list(ids):
        if "::" in item or "/" in item:
            continue
        pytest_tail = item.replace(".", "::")
        if any(
            other != item
            and (
                other.endswith("::" + item)
                or other.endswith("." + item)
                or other.endswith("::" + pytest_tail)
            )
            for other in ids
        ):
            out.discard(item)
    return sorted(out)


def _stale_bash_format_path_false_positive(args: str, paths: list[str]) -> bool:
    """True for old trace metadata that mistook a domain format for a file write.

    Earlier action metadata treated any Python `.write(...)` call as a file write.
    That incorrectly marked calls such as `tbl.write(sys.stdout,
    format='ascii.rst')` as source mutations. Keep this compatibility correction
    deliberately narrow so old, truncated trace summaries for real source writes
    are not downgraded.
    """
    if ".write(" not in (args or ""):
        return False
    if not paths:
        return False
    if ".write(sys." in args or ".write(stdout" in args:
        return True
    for path in paths:
        pattern = _FORMAT_PATH_RE.format(path=re.escape(str(path)))
        if not re.search(pattern, args or ""):
            return False
    return True


def project_tool_event(ev: dict) -> dict:
    """Project one `tool_call` trace event into an adaptive-control slot dict.

    Prefer recorded file effects independently of command success. Legacy
    events without observations retain action-metadata and tool-name fallbacks.
    """
    tool = ev.get("tool_name", "") or ""
    result = ev.get("result_summary", "") or ""
    args = ev.get("args_summary", "") or ""
    gate_blocked = bool(ev.get("gate_blocked"))
    explicit_pass_fail = str(ev.get("pass_fail") or "").strip().lower()
    explicit_outcome = str(ev.get("outcome") or "").strip().lower()
    try:
        explicit_exit = int(ev["exit_status"]) if ev.get("exit_status") not in (None, "") else None
    except (TypeError, ValueError):
        explicit_exit = None
    error = (
        explicit_pass_fail == "fail"
        or explicit_outcome == "error"
        or (explicit_exit is not None and explicit_exit != 0)
        or result.startswith("ERROR:")
        or bool(_is_error_result(result))
    )
    if ev.get("outcome_version") == "native_execution_v1":
        error = explicit_pass_fail == "fail" or explicit_outcome == "error"
        gate_blocked = gate_blocked or explicit_outcome in {"blocked", "not_executed", "unknown"}
    clean = not error and not gate_blocked

    sw_like = _truthy(ev.get("source_write_like"))
    write_like = _truthy(ev.get("write_like"))
    paths = ev.get("source_write_paths") or []
    if isinstance(paths, str):
        paths = [paths]
    if tool == "bash" and sw_like and _stale_bash_format_path_false_positive(args, paths):
        write_like = False
        sw_like = False
        paths = []

    # source mutation: emitted source-contact metadata, else tool-name fallback
    source_mutation = clean and (sw_like or tool in _MUTATION_TOOLS)
    from ..file_changes import observed_mutation
    observed = observed_mutation(ev)
    if observed is not None:
        source_mutation = observed
    elif ev.get("outcome_version") == "native_execution_v1" and tool in _MUTATION_TOOLS:
        source_mutation = False
    # a write-shaped action (incl. non-source writes) shows as an edit slot
    write_action = observed if observed is not None else source_mutation or (clean and write_like)

    op = _op_kind(tool)
    # gated on clean like test_like: a gate-blocked done never executed and
    # must not count as a submit-like action
    submit = clean and (op == "SUBMIT" or tool in ("done", "submit"))
    # A gate-blocked call did not run, so it cannot count as verification.
    # A read-only command that names a test also does not run that test.
    registered_command = matches_command(command_from_summary(args))
    test_like = clean and op == "RUN" and registered_command

    # Test-outcome projection (verification finish guards): bash appends
    # `[exit code: N]` to result_summary ONLY for nonzero exits, and summaries
    # keep the tail, so the marker survives truncation. Classification order:
    # nonzero marker -> fail; else runner summary line in the output tail
    # (fail pattern before pass pattern); else "" — pipes through `head`
    # swallow the exit code AND can cut the summary, so silence is unknown,
    # never green.
    m = _EXIT_CODE_RE.search(result)
    exit_code = str(explicit_exit) if explicit_exit is not None else (m.group(1) if m else "")
    if ev.get("outcome_version") == "native_execution_v1":
        exit_code = str(explicit_exit) if explicit_exit is not None else ""
    args_l = args.lower()
    from ..runner_invocations import recorded_runner_requests
    runner_requests = recorded_runner_requests(ev)
    runner_request = runner_requests[0] if len(runner_requests) == 1 else {}
    supported_request = bool(runner_request.get('family')) and runner_request.get('basis') in {
        'invocation_syntax', 'runtime_selection', 'explicit_configuration',
    }
    test_exec = (
        op == "RUN"
        and not write_action
        and (
            registered_command
            or (not _READ_ONLY_CMD_RE.match(args) and _V3_UNITTEST_RUN_RE.search(result))
        )
        and not any(f in args_l for f in _TEST_NON_RUN_FLAGS)
    )
    verification_status = str(ev.get("verification_status") or "")
    if tool == "run_tests":
        # A structured tool has no shell command to recognize. Its producer
        # records the runner status independently of rendered output.
        test_exec = supported_request and verification_status in {"passed", "failed", "timed_out"}
        test_like = clean and test_exec
    elif tool == "bash" and verification_status:
        # New traces carry the shell's attributed result. A recognized but
        # skipped/ambiguous invocation is not an executed test.
        test_exec = supported_request and verification_status in {"passed", "failed", "timed_out"}
        test_like = clean and test_exec
    elif tool == "bash_poll" and verification_status:
        test_exec = supported_request and verification_status in {"passed", "failed"}
        test_like = clean and test_exec
    if not test_exec:
        test_exit_status = ""
    elif tool == "run_tests" or (tool in {"bash", "bash_poll"} and verification_status):
        test_exit_status = (
            "pass" if verification_status == "passed" and explicit_exit == 0 and clean
            else "fail"
        )
    elif exit_code not in ("", "0"):
        test_exit_status = "fail"
    elif _TEST_FAIL_OUT_RE.search(result):
        test_exit_status = "fail"
    elif _TEST_PASS_OUT_RE.search(result) and not error and not gate_blocked:
        test_exit_status = "pass"
    else:
        test_exit_status = ""
    if error:
        slot_state = "tool_error"
    elif write_action:
        slot_state = "edit"
    elif submit:
        slot_state = "submit"
    else:
        slot_state = op.lower() if op else "other"
    # Additional verification and action fields.
    is_run = op == "RUN"
    nonzero_exit = exit_code not in ("", "0")
    static_check = bool(_V2_STATIC_RE.search(args) or _V2_STATIC_RE.search(result))
    pass_visible = bool(_V2_PASS_RE.search(result))
    fail_visible = bool(_V2_FAIL_RE.search(result))
    # Parsed per-node failure IDs take precedence over loader tokens.
    # pytest-repo test output CONTAINS pytest-output strings like "no tests
    # ran", which could otherwise hide real failing tests.
    has_failed_ids = bool(is_run and _V2_FAILED_ID_RE.search(result))
    if not is_run:
        exec_outcome = ""
    elif (test_exec and clean  # blocked/errored calls never executed
          and fail_visible
          and (has_failed_ids or not _V3_LOADER_ERR_RE.search(result))
          and exit_code != "141"):                       # SIGPIPE display
        exec_outcome = "fail"
    elif (not nonzero_exit and not error and not gate_blocked
          and not static_check and pass_visible
          and not (fail_visible and not test_exec)
          and not _V3_MOCK_RE.search(args) and not _V3_MOCK_RE.search(result)):
        exec_outcome = "pass"
    else:
        exec_outcome = ""
    # A nonzero-exit run with pass-only output does not open red-test state.
    if exec_outcome == "fail" and pass_visible and not _V2_FAIL_RE.search(result):
        exec_outcome = ""

    failed_ids = _drop_bare_duplicate_ids(
        {g for m2 in _V2_FAILED_ID_RE.finditer(result) for g in m2.groups() if g}
        | {f"{m2.group(2)}.{m2.group(1)}" for m2 in _V3_VERBOSE_FAIL_RE.finditer(result)}
    ) if exec_outcome == "fail" else []
    passed_ids = sorted({g for m2 in _V2_PASSED_ID_RE.finditer(result)
                         for g in m2.groups() if g}) if is_run else []
    if is_run:  # Django and unittest verbose greens count as green IDs.
        passed_ids = sorted(set(passed_ids) | {m2.group(1) for m2 in _V3_VERBOSE_OK_RE.finditer(result)})
    if ev.get("outcome_version") == "native_execution_v1":
        exec_outcome = explicit_pass_fail if explicit_pass_fail in {"pass", "fail"} else ""
    # Treat any "backup" substring as an exclusion marker, not only ".backup".
    # This covers restore commands such as `cp /tmp/autodoc_backup.py <src>`.
    exclusion = bool(_V2_EXCLUSION_RE.search(args)) or "backup" in args
    script_m = _V3_SCRIPT_PATH_RE.search(args)
    runner_family = runner_request.get("family", "")
    attributed_test_execution = (
        test_exec and not gate_blocked and ev.get("error_class") != "security_block"
        and verification_status in {"passed", "failed"}
        and supported_request
    )
    suite_green = bool(is_run and not error and not nonzero_exit
                       and _V3_SUITE_GREEN_RE.search(result))
    # Retain crash-frame paths only inside the discovered task workspace.
    from ..bash_write_classification import is_workspace_path
    traceback_paths = sorted({m2.group(1) for m2 in
                              _CRASH_FRAME_RE.finditer(result)
                              if is_workspace_path(m2.group(1))}) \
        if (is_run and (nonzero_exit or error) and _CRASH_MARKER_RE.search(result)) else []

    # Do not count a flagged write that failed or only touched /tmp. Count a
    # clean mutation-shaped command, including git apply, against other paths.
    tmp_only_paths = bool(paths) and all(str(p).startswith("/tmp") for p in paths)
    tmp_heredoc_prep = (
        bool(_TMP_HEREDOC_RE.search(args))
        and result.strip() in _NO_OUTPUT_MARKERS
        and not _V2_MUT_CMD_RE.search(args)
    )
    if observed is not None:
        effective_mutation = observed
    elif source_mutation and (_V2_PERM_FAIL_RE.search(result) or nonzero_exit or tmp_only_paths):
        effective_mutation = False
    elif source_mutation and tmp_heredoc_prep:
        effective_mutation = False
    elif source_mutation:
        effective_mutation = True
    else:
        # `git apply` targets repository files named inside the patch, so the
        # path outside /tmp does not need to appear in the command.
        effective_mutation = (clean and is_run
                              and (("git apply" in args)
                                   or (bool(_V2_MUT_CMD_RE.search(args))
                                       and bool(_V2_NON_TMP_PATH_RE.search(args)))))

    return {
        "slot_projection_version": SLOT_PROJECTION_VERSION,
        "execution_outcome_version": ev.get("outcome_version", "legacy_text_inference"),
        "slot_idx": int(ev.get("turn_number", 0) or 0),
        "slot_presence": "filled",
        "slot_state": slot_state,
        "op_kind": op,
        "obs_state": "tool_error" if error else "",
        "contact_state": "source_write" if source_mutation else "",
        "source_mutation": "true" if source_mutation else "false",
        "test_like_action": "true" if test_like else "false",
        "test_execution_action": "true" if attributed_test_execution else "false",
        "exit_code": exit_code,
        "test_exit_status": test_exit_status if attributed_test_execution else "",
        "exec_outcome": exec_outcome,
        "failed_test_ids": ";".join(failed_ids),
        "passed_test_ids": ";".join(passed_ids),
        "exclusion_context": "true" if exclusion else "false",
        "effective_source_mutation": "true" if effective_mutation else "false",
        "args_prefix": args[:80],
        "script_path": script_m.group(1) if script_m else "",
        "runner_family": runner_family,
        "runner_targets": ";".join(runner_request.get("targets", [])),
        "runner_identity_source": runner_request.get("basis", "unknown"),
        "runner_target_status": runner_request.get("target_status", "unknown"),
        "runner_requests": json.dumps(runner_requests, sort_keys=True),
        "runner_execution_source": "recorded_status" if attributed_test_execution else "unknown",
        "runner_execution_status": verification_status or "unknown",
        "suite_green": "true" if suite_green else "false",
        "traceback_paths": ";".join(traceback_paths),
        "submit_like_action": "true" if submit else "false",
        "target_ref": ";".join(str(p) for p in paths[:3]),
        "repeat_signature": action_signature(tool, args),
        "executed": "false" if (gate_blocked or error) else "true",
        "evidence_refs": f"turn={ev.get('turn_number', '')}",
    }


def recent_prefix_slots_from_events(trace_events, observation_slot: int) -> list[dict]:
    """Prefix-only projection: tool_call events with turn_number <= observation_slot.

    Never reads events beyond the observation slot, so the result for prefix k is
    independent of any later turns appended afterward (future_evidence_used=false).
    """
    out: list[dict] = []
    stash_active = False
    for ev in trace_events or []:
        if ev.get("event") != "tool_call":
            continue
        tn = ev.get("turn_number")
        if tn is None or int(tn) > int(observation_slot):
            continue
        slot = project_tool_event(ev)
        args = ev.get("args_summary", "") or ""
        if stash_active:
            slot["exclusion_context"] = "true"
            slot["control_context"] = "git_stash"
        if _GIT_STASH_RE.search(args):
            slot["exclusion_context"] = "true"
            slot["control_context"] = "git_stash"
            stash_active = True
        elif _GIT_STASH_POP_RE.search(args):
            slot["exclusion_context"] = "true"
            slot["control_context"] = "git_stash"
            stash_active = False
        out.append(slot)
    return out
