"""Telemetry retention preserves bytes without entering the task filesystem."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import retained_output
from scripts.llm_solver.harness._loop.trace_output import _result_fields
from scripts.llm_solver.harness.adaptive_control.llm_detector_utils import _trace_prefix_rows


@pytest.fixture
def session(tmp_path):
    workspace = tmp_path / 'task'
    workspace.mkdir()
    artifacts = tmp_path / 'custom-artifacts'
    return SimpleNamespace(cwd=workspace, _trace_path=artifacts / '.trace.jsonl',
        cfg=make_config(trace_result_summary_chars=30), _sink_counter=0, _session_number=1)


def test_trace_copy_bypasses_task_admission_and_preserves_bytes(session, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('trace retention entered task filesystem')
    monkeypatch.setattr(retained_output, 'output_file_scope', forbidden)
    text = 'Unicode π\n' * 20
    fields = _result_fields(session, text, 7)
    root = session._trace_path.parent
    assert (root / fields['output_full_path']).read_bytes() == text.encode()
    assert fields['output_full_path_scope'] == 'telemetry'
    assert fields['output_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert not list(session.cwd.iterdir())
    rows = _trace_prefix_rows([dict(fields, event='tool_call', turn_number=7)],
        observation_turn=7, max_field_chars=1000)
    assert rows[0]['output_full_path'] == ''


def test_existing_entries_and_symlink_targets_are_preserved(session):
    root = session._trace_path.parent
    output = root / 'tool_output'
    output.mkdir(parents=True)
    existing = output / '1_0001_t7_trace.log'
    existing.write_bytes(b'original')
    outside = root / 'outside'
    outside.write_bytes(b'keep')
    (output / '1_0002_t7_trace.log').symlink_to(outside)
    saved = retained_output.save_output(session, 'new', 7, trace=True)
    assert saved == 'tool_output/1_0003_t7_trace.log'
    assert (root / saved).read_bytes() == b'new'
    assert existing.read_bytes() == b'original'
    assert outside.read_bytes() == b'keep'


def test_symlinked_output_directory_is_unavailable(session):
    root = session._trace_path.parent
    root.mkdir()
    (root / 'tool_output').symlink_to(session.cwd, target_is_directory=True)
    fields = _result_fields(session, 'long output\n' * 20, 7)
    assert not fields['output_retained']
    assert fields['output_full_path'] == ''
    assert not list(session.cwd.iterdir())


def test_failed_host_write_preserves_result_hash_and_has_no_fallback(session, monkeypatch):
    writes = []
    def fail(*args, **kwargs):
        writes.append(args)
        raise OSError('disk full')
    monkeypatch.setattr(Path, 'open', fail)
    text = 'large result\n' * 20
    fields = _result_fields(session, text, 2)
    assert fields['output_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert not fields['output_retained']
    assert fields['output_full_path'] == ''
    assert len(writes) == 1
    assert not list(session.cwd.iterdir())


def test_named_trace_stream_and_default_telemetry_root(session):
    from scripts.llm_solver._shared.telemetry_paths import telemetry_dir
    del session._trace_path
    assert retained_output.trace_output_root(session) == telemetry_dir(session.cwd)
    session._trace_file = SimpleNamespace(name=session.cwd.parent / 'logs/trace.jsonl')
    assert retained_output.trace_output_root(session) == session.cwd.parent / 'logs'


def test_short_result_needs_no_retention(session):
    fields = _result_fields(session, 'short', 1)
    assert fields['output_snippet'] == 'short'
    assert not fields['output_retained']
    assert not session._trace_path.parent.exists()


def test_legacy_in_workspace_trace_does_not_create_untracked_output(session):
    session._trace_path = session.cwd / '.trace.jsonl'
    assert retained_output.save_output(session, 'large output', 1, trace=True) == ''
    assert not list(session.cwd.iterdir())
