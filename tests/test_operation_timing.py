"""Operation costs preserve nesting, worker attribution and missing evidence."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import threading
import time

import pytest

from scripts.llm_solver.harness import operation_timing as timing
from scripts.llm_solver.analysis.timing_report import collect, render, trace_paths


def test_nested_costs_repeated_calls_and_exceptions(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    with timing.operation_profile() as profile, timing.operation_call("done-3"):
        with timing.operation("verification"):
            clock[0] = 1
            for _ in range(3):
                with timing.operation("files.resolve"):
                    clock[0] += 2
            with pytest.raises(OSError), timing.operation("files.read"):
                clock[0] += 1
                raise OSError("secret path must not enter the metric")
            clock[0] += 2
        rows = {tuple(r["operation"]): r for r in profile.take()}
        assert profile.take() == []
    assert rows[("verification",)]["elapsed_ms"] == 10000
    assert rows[("verification",)]["self_ms"] == 3000
    assert rows[("verification", "files.resolve")]["calls"] == 3
    assert rows[("verification", "files.resolve")]["max_ms"] == 2000
    assert rows[("verification", "files.read")]["raised"] == 1
    assert all(row["tool_call_id"] == "done-3" for row in rows.values())
    assert "secret" not in json.dumps(list(rows.values()))
    assert timing.take_operations() == []


def test_copied_workers_do_not_subtract_from_another_threads_parent():
    barrier = threading.Barrier(2)
    def work(call_id):
        with timing.operation_call(call_id), timing.operation("files.read"):
            barrier.wait(timeout=5)
            time.sleep(0.02)
    with timing.operation_profile() as profile, ThreadPoolExecutor(2) as pool:
        with timing.operation("parent_wait"):
            futures = [pool.submit(copy_context().run, work, str(i)) for i in range(2)]
            for future in futures:
                future.result()
        rows = profile.take()
    parent = next(row for row in rows if row["operation"] == ["parent_wait"])
    assert parent["self_ms"] == parent["elapsed_ms"]
    children = [row for row in rows if row["operation"] == ["files.read"]]
    assert {row["tool_call_id"] for row in children} == {"0", "1"}
    assert all(row["calls"] == 1 and row["self_ms"] >= 20 for row in children)


def test_profile_disabled_does_not_read_clock_or_labels(monkeypatch):
    monkeypatch.setattr(timing.time, "perf_counter", lambda: pytest.fail("timed outside profile"))
    @timing.measured(lambda *args: pytest.fail("built label outside profile"))
    def work(value):
        return value
    assert work("unchanged") == "unchanged"


def test_mutation_span_preserves_retained_inputs_and_subtracts_revision_child(monkeypatch):
    from scripts.llm_solver.harness._guardrails import verification
    from scripts.llm_solver.harness._guardrails.state import GuardrailState

    clock, calls = [0.0], []
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    original_normalize = verification.normalize_trace_path
    def normalize(path):
        clock[0] += 0.002
        return original_normalize(path)
    @timing.measured('verification.file_revisions')
    def revisions(cwd, paths):
        calls.append((cwd, paths))
        clock[0] += 0.007
        return {path: 'fresh' for path in paths}
    monkeypatch.setattr(verification, 'normalize_trace_path', normalize)
    monkeypatch.setattr(verification, '_file_revisions', revisions)
    state = GuardrailState(verification_file_revisions={'retained.py': 'old'},
                           formal_verification_passed_since_mutation=True)
    with timing.operation_profile() as profile, timing.operation_call('edit-7'):
        verification.record_verification_mutation(state, tc_name='write', result='ok',
            source_write_paths=('src/new.py', 'retained.py'), cwd='/fixture')
        rows = {tuple(row['operation']): row for row in profile.take()}
    assert calls == [('/fixture', ['retained.py', 'src/new.py'])]
    assert state.verification_file_revisions == {'retained.py': 'fresh', 'src/new.py': 'fresh'}
    assert not state.formal_verification_passed_since_mutation
    parent = rows[('verification.record_mutation',)]
    child = rows[('verification.record_mutation', 'verification.file_revisions')]
    assert parent['elapsed_ms'] == 11 and parent['self_ms'] == 4
    assert child['elapsed_ms'] == child['self_ms'] == 7
    assert all(row['calls'] == 1 and row['tool_call_id'] == 'edit-7' for row in rows.values())


def test_report_keeps_old_trace_gaps_and_does_not_sum_parents(tmp_path):
    path = tmp_path / "task.trace.jsonl"
    operation = {"operation": ["verification", "files.resolve"], "tool_call_id": "done-3",
                 "calls": 100, "elapsed_ms": 9000, "self_ms": 9000, "max_ms": 100, "raised": 0}
    parent = {**operation, "operation": ["verification"], "calls": 1,
              "self_ms": 1000, "elapsed_ms": 10000, "max_ms": 10000}
    rows = [
        {"event": "turn_timing", "session_number": 1, "turn_number": 3,
         "chat_call_ms": 2000, "tool_ms": 0, "harness_ms": 10000,
         "operation_timing_version": 1, "operations": [parent, operation]},
        {"event": "turn_timing", "session_number": 2, "turn_number": 1},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"event":')
    report = collect(trace_paths([tmp_path, path]), limit=2)
    assert report["traces"] == 1
    assert report["timed_turns"] == 2 and report["detailed_turns"] == 1
    assert report["invalid_json_lines"] == 1
    assert report["wall_ms"]["harness_ms"] == 10000
    assert report["operations"][0]["calls"] == 100
    assert report["slow_operations"][0]["tool_call_id"] == "done-3"
    text = render(report)
    assert "unknown" in text and "verification > files.resolve" in text
    assert "turn=3 call=done-3" in text


def test_real_local_command_records_process_launch_and_preserves_output(tmp_path):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    with timing.operation_profile() as profile:
        result = _run_in_sandbox("printf 'timing fixture'", cwd=str(tmp_path),
                                 timeout=5, sandbox=False, bwrap_bin="")
        rows = profile.take()
    assert result == ("timing fixture", 0, False)
    process = next(row for row in rows if row["operation"][-1] == "process.execute")
    assert process["calls"] == 1 and process["raised"] == 0
    assert process["operation"] == ["sandbox.command", "process.execute"]


def test_real_command_timeout_records_elapsed_and_raised(tmp_path):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    with timing.operation_profile() as profile:
        result = _run_in_sandbox("exec sleep 2", cwd=str(tmp_path), timeout=0.03,
                                 sandbox=False, bwrap_bin="")
        rows = profile.take()
    assert result == ("", None, True)
    process = next(row for row in rows if row["operation"][-1] == "process.execute")
    assert process["raised"] == 1 and process["elapsed_ms"] >= 20


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_container_exec_count_is_the_actual_launcher(tmp_path, monkeypatch, runtime):
    import subprocess
    from scripts.llm_solver.harness._tools._run_in_sandbox import _execute
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, "", ""))
    with timing.operation_profile() as profile:
        _execute([f"/usr/bin/{runtime}", "exec", "private-container", "true"], timeout=5)
        row, = profile.take()
    assert row["operation"] == ["process.container_exec"] and row["calls"] == 1
    assert "private-container" not in json.dumps(row)
