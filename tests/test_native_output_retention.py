"""Saved output and its pointer use the same task view as model reads."""
from dataclasses import replace
import hashlib
from pathlib import Path
import re
import subprocess

from tests.test_task_files import bwrap, namespace_files
from tests.test_structured_output_retention import projection
from scripts.llm_solver.harness import task_file_runtime
from scripts.llm_solver.harness._loop.state_projection import project_and_sink, sink_to_disk
from scripts.llm_solver.harness._loop.trace_output import _result_fields, _sink_trace_output
from scripts.llm_solver.harness.tools import dispatch


def bind_projection(projection, bwrap, tmp_path, monkeypatch, *, readonly=False):
    session, raw, ledger = projection
    source, files = namespace_files(bwrap, tmp_path, readonly=readonly)
    session.cwd = Path(str(files.root))
    session.cfg = replace(session.cfg, sandbox_bash=True, bwrap_bin=bwrap)
    session._effective_env = {'PATH': '/usr/bin:/bin', 'HOME': str(session.cwd)}
    session._allow_login_shell = False
    session._ignore_policy = None
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    monkeypatch.setattr(task_file_runtime, 'make_task_files', lambda *args, **kwargs: files)
    return session, raw, ledger, source, files


def test_native_saved_pointer_and_trace_read_the_same_bytes(projection, bwrap, tmp_path, monkeypatch):
    session, raw, ledger, source, files = bind_projection(projection, bwrap, tmp_path, monkeypatch)
    for directory in (source, session.cwd):
        subprocess.run(['git', 'init', '-q', str(directory)], check=True)
    host_exclude = session.cwd / '.git/info/exclude'
    host_exclude_before = host_exclude.read_bytes()
    hidden_dir = session.cwd / '.tool_output'
    hidden_dir.mkdir()
    hidden = hidden_dir / '1_0001_t3.log'
    hidden.write_text('HIDDEN_HOST_OUTPUT')
    result = project_and_sink(session, 'bash', 'check', raw, 3)
    pointer = re.search(r'full_path="([^"]+)"', result)[1]
    assert (source / pointer).read_text() == raw
    model_read = dispatch('read', {'path': pointer}, cwd=str(session.cwd), cfg=session.cfg)
    assert 'REQUIRED_DIAGNOSTIC' in model_read and 'HIDDEN_HOST_OUTPUT' not in model_read
    trace = _result_fields(session, result, 3)
    assert trace['output_retained'] is True
    assert trace['output_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
    assert hidden.read_text() == 'HIDDEN_HOST_OUTPUT'
    assert '/' + pointer in (source / '.git/info/exclude').read_text().splitlines()
    assert host_exclude.read_bytes() == host_exclude_before
    session._sink_counter = 0
    second = sink_to_disk(session, 'second result', 3)
    second_path = re.search(r'full_path="([^"]+)"', second)[1]
    assert second_path != pointer and (source / pointer).read_text() == raw
    assert (source / second_path).read_text() == 'second result'
    assert not list((source / '.tool_output').glob('.yuj-output-*'))


def test_readonly_sink_keeps_raw_result_and_does_not_write_the_host(projection, bwrap, tmp_path, monkeypatch):
    session, raw, ledger, source, files = bind_projection(
        projection, bwrap, tmp_path, monkeypatch, readonly=True,
    )
    assert project_and_sink(session, 'bash', 'check', raw, 3) == raw
    ledger.record_transform.assert_not_called()
    assert _sink_trace_output(session, raw, 3) == ''
    assert not (source / '.tool_output').exists()
    assert not (session.cwd / '.tool_output').exists()


def test_native_exclusive_creation_refuses_symlinks_and_directories(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'original').write_bytes(b'keep')
    (source / 'alias').symlink_to('original')
    (source / 'dangling').symlink_to('absent')
    (source / 'directory').mkdir()
    import pytest
    for name in ('original', 'alias', 'dangling', 'directory'):
        with pytest.raises(FileExistsError):
            files.create_bytes(name, b'new')
    assert (source / 'original').read_bytes() == b'keep'
    assert not (source / 'absent').exists()
    assert not list(source.glob('.yuj-output-*'))
