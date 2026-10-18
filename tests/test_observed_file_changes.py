"""Known audit 086 counterexamples use real commands and dispatch consumers."""
import gzip
import json
import os
import shlex
import shutil
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.file_changes import _inventory, observed_dispatch
from scripts.llm_solver.harness.task_files import NamespaceFiles
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def assert_projection_consumers(events, event, changed):
    from scripts.llm_solver.harness.state_writer import project
    from scripts.llm_solver.harness.context_strategies.salience_context import SalienceContext
    from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event

    state = project(events, max_result_chars=2000, imperative_projection=True)
    item = next(item for item in state["trace"] if item["turn"] == event["turn_number"])
    assert item["file_changes"] == event["file_changes"]
    assert state["process"]["last_mutation_step"] == (item["step"] if changed else None)
    assert state["process"]["last_failed_mutation_step"] is None
    assert state["process"]["phase"] == ("post_mutation_unverified" if changed else "pre_mutation_discovery")
    assert SalienceContext._is_mutation_item(item) is changed
    assert SalienceContext._is_successful_mutation_item(item) is changed
    slot = project_tool_event(event)
    assert slot["source_mutation"] == ("true" if changed else "false")
    assert slot["effective_source_mutation"] == ("true" if changed else "false")


CASES = [
    ("python", "from pathlib import Path; Path('module.py').write_text('after')", True),
    ("node", "require('fs').writeFileSync('module.py', 'after')", True),
    ("printed", 'print("Path(\'module.py\').write_text(\'after\')")', False),
    ("failed", "from pathlib import Path; Path('module.py').write_text('after'); raise SystemExit(1)", True),
    ("abort", "from pathlib import Path; Path('module.py').write_text('after'); raise SystemExit(1)", True),
    ("rename", "from pathlib import Path; Path('module.py').rename('renamed.py')", True),
    ("delete", "from pathlib import Path; Path('module.py').unlink()", True),
]


@pytest.mark.parametrize("case,program,changed", CASES)
def test_real_dispatch_consumers(tmp_path, case, program, changed, sandbox=False):
    task = tmp_path / "task"
    task.mkdir()
    (task / "module.py").write_text("before")
    executable = shutil.which("node") if case == "node" else sys.executable
    if executable is None:
        pytest.skip("Node is unavailable")
    cmd = shlex.join([executable, "-e" if case == "node" else "-c", program])
    cfg = make_config(max_turns=2, sandbox_bash=sandbox, turn_snapshots_enabled=True,
                      auto_commit=False, duplicate_abort=20,
                      error_abort_threshold=1 if case == "abort" else 0)
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(content=None, tool_calls=[ToolCall(id="c1", name="bash", arguments={"cmd": cmd})],
                   finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5)),
        TurnResult(content="finished", tool_calls=[], finish_reason="stop",
                   usage=Usage(prompt_tokens=10, completion_tokens=5)),
    ]
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = tmp_path / "trace.jsonl"
    with trace.open("a") as sink, patch(
        "scripts.llm_solver.harness.turn_snapshots.snapshot", return_value="fixture_snapshot"
    ) as snapshot:
        session = Session(cfg, client, "fixture", "run the fixture", str(task),
                          trace_file=sink, session_number=1)
        session._guards.verified_since_mutation = True
        session._guards.formal_verification_passed_since_mutation = True
        session.run()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    event = next(e for e in events if e.get("event") == "tool_call" and e.get("tool_call_id") == "c1")
    assert session._guards.has_mutated is changed
    assert snapshot.call_count == int(changed)
    assert event["source_write_like"] is changed
    assert (event["action_class"] == "source_write") is changed
    observation = event["file_changes"]
    assert observation["status"] == ("changed" if changed else "unchanged_metadata")
    assert observation["tool_call_id"] == "c1"
    assert observation["task_binding"]["working_directory"] == str(task)
    if changed:
        assert "module.py" in observation["changed_paths"]
        assert not session._guards.verified_since_mutation
        assert not session._guards.formal_verification_passed_since_mutation
        assert "module.py" in session._guards.post_mutation_source_paths
    if case == "rename":
        assert observation["changed_paths"] == ["module.py", "renamed.py"]
    if case == "printed":
        assert event["predicted_source_write_like"] is True
        assert (task / "module.py").read_text() == "before"
    if case in {"failed", "abort"}:
        assert event["exit_status"] == 1
    assert_projection_consumers(events, event, changed)


@pytest.mark.parametrize("status", ["unchanged_metadata", "unavailable", "incomplete"])
def test_present_observation_never_falls_back_to_predicted_write(status):
    from scripts.llm_solver.harness.action_metadata import action_metadata

    args = {"cmd": "python -c \"from pathlib import Path; Path('module.py').write_text('after')\""}
    event = {"event": "tool_call", "session_number": 1, "turn_number": 1,
        "tool_call_id": "fixture", "tool_name": "bash", "args_summary": repr(args),
        **action_metadata("bash", args), "outcome_version": "native_execution_v1",
        "outcome": "completed", "exit_status": 0, "pass_fail": "unknown",
        "file_changes": {"basis": "native_entry_metadata_v1", "status": status, "changed_paths": []}}
    assert_projection_consumers([event], event, False)


def test_native_dispatch_consumers(bwrap, tmp_path):
    test_real_dispatch_consumers(tmp_path, *CASES[1], sandbox=True)


def test_observation_uses_native_view(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / "file\nname").write_text("native")
    host = tmp_path / "view"
    (host / "hidden").write_text("host")
    metadata = {}

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        files.run('printf changed > "$1"', ["file\nname"], None)
        kwargs["execution_metadata"]["executed"] = True

    with activate_task_files(files, host_root=host):
        execute("bash", {"cmd": "fixture"}, cwd=str(host), cfg=make_config(),
                execution_metadata=metadata)
    assert metadata["file_changes"]["changed_paths"] == ["file\nname"]
    assert (host / "hidden").read_text() == "host"


def test_large_inventory_switches_to_compression_and_back(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness import file_changes

    source, files = namespace_files(bwrap, tmp_path)
    monkeypatch.setattr(file_changes, '_INVENTORY_GZIP_MIN_BYTES', 1024)
    calls = []
    native = files.run
    def observed(script, args, data):
        calls.append((script, args))
        return native(script, args, data)
    monkeypatch.setattr(files, 'run', observed)
    for index in range(40):
        (source / f'long-inventory-name-{index:04d}.txt').write_text('x')

    expected, _ = _inventory(files, str(source), ())
    assert files._inventory_compression
    first_calls = len(calls)
    assert _inventory(files, str(source), ())[0] == expected
    assert files._inventory_gzip and len(calls) == first_calls + 2
    assert files._inventory_gzip in calls[-1][1]

    for path in source.iterdir():
        path.unlink()
    assert _inventory(files, str(source), ())[0] == {}
    assert not files._inventory_compression
    assert files._inventory_gzip in calls[-1][1]
    assert _inventory(files, str(source), ())[0] == {}
    assert files._inventory_gzip not in calls[-1][1]


@pytest.mark.parametrize('gzip_kind', ['missing', 'unsupported'])
def test_unavailable_inventory_compressor_keeps_raw_inventory(tmp_path, gzip_kind):
    root, bins = tmp_path / 'task', tmp_path / 'bin'
    root.mkdir()
    bins.mkdir()
    for name in ('find',):
        (bins / name).symlink_to(shutil.which(name))
    if gzip_kind == 'unsupported':
        (bins / 'gzip').write_text('#!/bin/sh\nexit 1\n')
        (bins / 'gzip').chmod(0o700)
    calls = []
    def run(script, args, data):
        calls.append((script, args))
        return subprocess.run(['/bin/bash', '--noprofile', '--norc', '-c',
            script, 'inventory-compressor', *args], input=data, capture_output=True,
            env={'PATH': str(bins)}, timeout=15)
    files = NamespaceFiles(str(root), run, binding={'test': gzip_kind}, shares_host_kernel=True)
    (root / 'file').write_text('body')
    files._inventory_compression = True

    assert set(_inventory(files, str(root), ())[0]) == {'file'}
    assert files._inventory_gzip == ''
    assert len(calls) == 3
    assert set(_inventory(files, str(root), ())[0]) == {'file'}
    assert len(calls) == 4


def test_inventory_compression_rejects_upstream_failure(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    wrapper = tmp_path / 'failed-producer'
    wrapper.write_text('#!/bin/sh\nprintf partial\nexit 9\n')
    wrapper.chmod(0o755)
    files._utilities['find'] = str(wrapper)
    files._inventory_compression = True
    with pytest.raises(OSError, match='did not complete'):
        _inventory(files, str(source), ())


@pytest.mark.parametrize('failure', ['process', 'corrupt', 'truncated'])
def test_inventory_compression_fails_closed(bwrap, tmp_path, monkeypatch, failure):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'file').write_text('body')
    files._inventory_compression = True
    compressor = files.inventory_compressor()
    native = files.run
    def failed(script, args, data):
        if compressor not in args:
            return native(script, args, data)
        if failure == 'process':
            return subprocess.CompletedProcess(args, 1, b'', b'failed')
        payload = b'not a gzip stream' if failure == 'corrupt' else gzip.compress(b'payload')[:-2]
        return subprocess.CompletedProcess(args, 0, payload, b'')
    monkeypatch.setattr(files, 'run', failed)
    error = OSError if failure == 'process' else ValueError
    with pytest.raises(error):
        _inventory(files, str(source), ())


@pytest.mark.parametrize('reuse_enabled', [False, True])
def test_inventory_comparison_keeps_entry_changes_and_reuse_inputs(bwrap, tmp_path, monkeypatch, reuse_enabled):
    from types import SimpleNamespace
    from scripts.llm_solver.harness import read_reuse

    source, files = namespace_files(bwrap, tmp_path)
    (source / '.git').mkdir()
    (source / '.git/config').write_text('original config')
    (source / '.git/index').write_bytes(b'original index')
    (source / '.gitignore').write_text('*.cache\n')
    (source / 'mode').write_text('same bytes')
    (source / 'deleted').write_text('remove me')
    (source / 'alias').symlink_to('mode')
    (source / 'directory').mkdir()
    snapshots = []
    if reuse_enabled:
        reuse = SimpleNamespace()
        stamp = read_reuse.inventory_stamp
        def record_stamp(entries, binding):
            snapshots.append(dict(entries))
            return stamp(entries, binding)
        monkeypatch.setattr(read_reuse, 'active_reuse', lambda: reuse)
        monkeypatch.setattr(read_reuse, 'inventory_stamp', record_stamp)
    metadata = {}

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        result = files.run('chmod 700 mode; rm deleted; ln -sfn absent alias; '
                           'printf ignored > generated.cache; printf new > untracked; '
                           'printf changed > .git/config; printf changed > .git/index', [], None)
        assert result.returncode == 0
        kwargs['execution_metadata'].update(executed=True, exit_status_known=True, exit_status=0)

    with activate_task_files(files, host_root=source):
        execute('bash', {'cmd': 'grep mode mode'}, cwd=str(source), cfg=make_config(),
                execution_metadata=metadata)
    record = metadata['file_changes']
    assert record['changed_paths'] == ['alias', 'deleted', 'generated.cache', 'mode', 'untracked']
    assert record['created_paths'] == ['generated.cache', 'untracked']
    if reuse_enabled:
        assert len(snapshots) == 2
        assert snapshots[0]['directory'][0] == snapshots[1]['directory'][0] == b'd'
        assert snapshots[0]['.git/config'] != snapshots[1]['.git/config']
        assert snapshots[0]['.git/index'] != snapshots[1]['.git/index']


@pytest.mark.parametrize('transition', ['created_owned', 'lost_identity'])
def test_inventory_comparison_uses_fresh_owned_exclusions(bwrap, tmp_path, transition):
    import os
    from pathlib import Path

    source, files = namespace_files(bwrap, tmp_path)
    host = source
    if transition == 'lost_identity':
        host = Path(files.root)
        (source / '.tool_output').write_text('old retained bytes')
        os.link(source / '.tool_output', host / '.tool_output')
        command = 'rm .tool_output; printf new > .tool_output'
    else:
        command = 'mkdir .tool_output; printf retained > .tool_output/new'
    metadata = {}

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        assert files.run(command, [], None).returncode == 0
        kwargs['execution_metadata'].update(executed=True, exit_status_known=True, exit_status=0)

    with activate_task_files(files, host_root=host):
        execute('bash', {'cmd': 'fixture'}, cwd=str(host), cfg=make_config(), execution_metadata=metadata)
    record = metadata['file_changes']
    if transition == 'created_owned':
        assert record['status'] == 'unchanged_metadata'
        assert record['changed_paths'] == []
        assert '.tool_output' in record['excluded_paths']
    else:
        assert record['changed_paths'] == ['.tool_output']
        assert record['created_paths'] == ['.tool_output']
        assert '.tool_output' not in record['excluded_paths']
        assert (host / '.tool_output').read_text() == 'old retained bytes'


def test_unavailable_observation_does_not_invent_no_change(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import file_changes

    def unavailable(*args):
        raise OSError("fixture denied")

    monkeypatch.setattr(file_changes, "_inventory", unavailable)
    metadata = {}

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        (tmp_path / "file").write_text("changed")

    execute("bash", {"cmd": "fixture"}, cwd=str(tmp_path),
            cfg=make_config(sandbox_bash=False), execution_metadata=metadata)
    assert metadata["file_changes"]["status"] == "unavailable"
    assert (tmp_path / "file").read_text() == "changed"


def test_exhausted_budget_does_not_launch_tool(tmp_path):
    from scripts.llm_solver.harness.time_budget import run_time_budget

    @observed_dispatch
    def execute(name, arguments, **kwargs):
        pytest.fail("tool must not run after its allowance is exhausted")

    metadata = {}
    with run_time_budget(1e-9):
        result = execute("bash", {"cmd": "fixture"}, cwd=str(tmp_path),
                         cfg=make_config(sandbox_bash=False), execution_metadata=metadata)
    assert "budget is exhausted" in result
    assert metadata["executed"] is False
    assert metadata["file_changes"]["status"] == "unavailable"
