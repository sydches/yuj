"""Tests for parallel read-only tool dispatch.

Direct Session.run() instrumentation is heavy; these tests exercise
the partition logic and the preexecuted-cache fallback separately to
keep the coverage focused.
"""
from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness import loop as loop_mod
from llm_solver.harness.tools import dispatch


class _FakeTC:
    def __init__(self, tc_id, name, arguments):
        self.id = tc_id
        self.name = name
        self.arguments = arguments


class TestReadonlyPartition:

    def test_readonly_set_is_stable(self):
        assert loop_mod._READONLY_TOOLS == frozenset(
            {"read", "glob", "grep", "structural_search"}
        )

    def test_write_not_in_readonly(self):
        assert "write" not in loop_mod._READONLY_TOOLS
        assert "edit" not in loop_mod._READONLY_TOOLS
        assert "bash" not in loop_mod._READONLY_TOOLS


class TestConcurrentDispatch:

    def test_two_reads_execute_concurrently(self, tmp_path):
        """Use ThreadPoolExecutor directly to verify dispatch is
        thread-safe for two reads on separate files."""
        (tmp_path / "a.txt").write_text("content-a")
        (tmp_path / "b.txt").write_text("content-b")
        cfg = make_config()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(
                dispatch, "read", {"path": "a.txt"},
                cwd=str(tmp_path), cfg=cfg,
            )
            fb = ex.submit(
                dispatch, "read", {"path": "b.txt"},
                cwd=str(tmp_path), cfg=cfg,
            )
            ra = fa.result()
            rb = fb.result()
        assert "content-a" in ra
        assert "content-b" in rb

    def test_session_read_workers_see_the_persistent_shell_files(self, tmp_path):
        import shutil
        import subprocess
        from llm_solver.server.types import TurnResult, Usage, ToolCall
        binary = shutil.which('bwrap')
        if not binary:
            pytest.skip('bwrap is unavailable')
        probe = subprocess.run([binary, '--ro-bind', '/', '/', '--', 'true'], capture_output=True)
        if probe.returncode:
            pytest.skip('mount namespaces are unavailable')
        hooks = tmp_path / '.git' / 'hooks'
        hooks.mkdir(parents=True)
        (tmp_path / '.yujignore').write_text('secret\n')
        (tmp_path / 'secret').write_text('ignored host content')
        for number in range(2):
            (hooks / f'marker-{number}').write_text('hidden host marker')
        cfg = make_config(sandbox_bash=True, bwrap_bin=binary, max_turns=4,
                          parallel_readonly_enabled=True, unreadable_paths=())
        client = MagicMock()
        calls = [
            [ToolCall(id='create', name='bash', arguments={
                'cmd': "printf 'selected marker 0' > .git/hooks/marker-0; "
                       "printf 'selected marker 1' > .git/hooks/marker-1",
            })],
            [ToolCall(id=f'read-{number}', name='read', arguments={
                'path': f'.git/hooks/marker-{number}',
            }) for number in range(2)],
            [],
        ]
        client.chat.side_effect = [TurnResult(
            content=None if group else 'done', tool_calls=group,
            finish_reason='tool_calls' if group else 'stop',
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        ) for group in calls]
        client.build_assistant_message.return_value = {'role': 'assistant', 'content': ''}
        with patch.object(loop_mod.Session, '_get_server_ctx', return_value=0):
            session = loop_mod.Session(cfg, client, 'sys', 'prompt', str(tmp_path))
            result = session.run()
        assert result.done
        observed = [message['content'] for message in session.context.get_messages()
                    if message.get('role') == 'tool'
                    and message.get('tool_call_id', '').startswith('read-')]
        assert len(observed) == 2
        assert all('selected marker' in content for content in observed), '\n'.join(
            message['content'] for message in session.context.get_messages()
            if message.get('role') == 'tool')
        assert all('hidden host marker' not in content for content in observed)
        assert all((hooks / f'marker-{number}').read_text() == 'hidden host marker'
                   for number in range(2))


class TestConfigDefaults:

    def test_parallel_disabled_by_default(self):
        cfg = make_config()
        assert cfg.parallel_readonly_enabled is False
        assert cfg.parallel_max_workers == 4

    def test_enabling_via_make_config(self):
        cfg = make_config(parallel_readonly_enabled=True, parallel_max_workers=8)
        assert cfg.parallel_readonly_enabled is True
        assert cfg.parallel_max_workers == 8


@pytest.mark.parametrize('parallel', [False, True])
def test_tool_timings_cover_preparation_postprocessing_and_shared_turn_work(tmp_path, monkeypatch, parallel):
    from llm_solver.harness._loop import run_step
    from llm_solver.server.types import TurnResult, Usage, ToolCall

    clock = [100.0]
    monkeypatch.setattr(run_step.time, 'perf_counter', lambda: clock[0])
    cfg = make_config(max_turns=2, parallel_readonly_enabled=parallel)
    client = MagicMock()
    count = 2 if parallel else 1
    calls = [ToolCall(id=f'read-{index}', name='read', arguments={'path': f'{index}.txt'})
             for index in range(count)]
    client.chat.side_effect = [
        TurnResult(content=None, tool_calls=calls, finish_reason='tool_calls',
                   usage=Usage(prompt_tokens=10, completion_tokens=5)),
        TurnResult(content='done', tool_calls=[], finish_reason='stop',
                   usage=Usage(prompt_tokens=10, completion_tokens=5)),
    ]
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': ''}
    session = loop_mod.Session(cfg, client, 'sys', 'prompt', str(tmp_path))
    pre_models = [0]
    original_hook = session._run_hook
    def hook(event, **kwargs):
        if event == 'pre_tool':
            clock[0] += 2
        if event == 'pre_model':
            pre_models[0] += 1
            if pre_models[0] == 2:
                clock[0] += 13
        return original_hook(event, **kwargs)
    monkeypatch.setattr(session, '_run_hook', hook)
    def dispatch(*args, **kwargs):
        clock[0] += 3
        return 'task result'
    monkeypatch.setattr(loop_mod, 'dispatch', dispatch)
    original_add = session.context.add_tool_result
    def add(*args, **kwargs):
        clock[0] += 5
        return original_add(*args, **kwargs)
    monkeypatch.setattr(session.context, 'add_tool_result', add)
    original_observe = session._observe_harness_tool_result
    def observe(**kwargs):
        clock[0] += 7
        return original_observe(**kwargs)
    monkeypatch.setattr(session, '_observe_harness_tool_result', observe)
    original_post = run_step._run_post_turn_hooks
    def post(*args, **kwargs):
        clock[0] += 11
        return original_post(*args, **kwargs)
    monkeypatch.setattr(run_step, '_run_post_turn_hooks', post)
    result = session.run()
    assert result.done
    rows = session._trace_events
    tool_rows = [row for row in rows if row['event'] == 'tool_call']
    timings = [row for row in rows if row['event'] == 'tool_timing']
    assert len(timings) == len(tool_rows) == count
    assert all(row['tool_dispatch_ms'] == 3000 for row in tool_rows)
    assert all(row['duration_ms'] == 3000 and row['dispatch_executed'] for row in tool_rows)
    ends = [row for row in rows if row['event'] == 'tool_end']
    assert len(ends) == count
    assert all(row['duration_ms'] == 3000 and not row['includes_queue_wait'] for row in ends)
    for call, timing in zip(tool_rows, timings):
        assert timing['tool_call_id'] == call['tool_call_id']
        assert timing['completed'] and timing['includes_queue_wait']
        assert timing['duration_ms'] >= 17000
        assert rows.index(timing) > rows.index(call)
    shared = next(row for row in rows if row['event'] == 'turn_timing')
    assert shared['post_turn_ms'] == 11000
    assert shared['next_model_preparation_ms'] == 13000
    assert shared['tool_phase_ms'] == count * 17000
    assert shared['duration_ms'] == shared['tool_phase_ms'] + 24000
    assert shared['tool_ms'] == count * 3000
    assert shared['model_to_boundary_ms'] == shared['chat_call_ms'] + shared['tool_ms'] + shared['harness_ms']
    assert shared['post_ms'] >= 36000
    assert rows[-1]['event'] == 'turn_timing'
    assert rows[-1]['boundary'] == 'session_end'
    assert rows[-1]['next_turn_number'] is None
    assert rows[-1]['post_ms'] is None


@pytest.mark.parametrize('failure', ['schema_reject', 'sandbox_unavailable'])
def test_tool_timing_distinguishes_rejected_and_interrupted_calls(tmp_path, monkeypatch, failure):
    from llm_solver.harness._loop import run_step
    from llm_solver.harness._tools._run_in_sandbox import SandboxUnavailableError
    from llm_solver.server.types import TurnResult, Usage, ToolCall

    clock = [100.0]
    monkeypatch.setattr(run_step.time, 'perf_counter', lambda: clock[0])
    cfg = make_config(max_turns=1, tools_schema_validation='reject')
    client = MagicMock()
    client.chat.return_value = TurnResult(content=None, finish_reason='tool_calls',
        tool_calls=[ToolCall(id='read', name='read', arguments=(
            {} if failure == 'schema_reject' else {'path': 'a.txt'}))],
        usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': ''}
    session = loop_mod.Session(cfg, client, 'sys', 'prompt', str(tmp_path))
    original_hook = session._run_hook
    def hook(event, **kwargs):
        if event == 'pre_tool':
            clock[0] += 2
        return original_hook(event, **kwargs)
    monkeypatch.setattr(session, '_run_hook', hook)
    def dispatch(*args, **kwargs):
        assert failure == 'sandbox_unavailable', 'rejected tool was executed'
        clock[0] += 3
        raise SandboxUnavailableError('fixture unavailable')
    monkeypatch.setattr(loop_mod, 'dispatch', dispatch)
    session.run()
    timing = next(row for row in session._trace_events if row['event'] == 'tool_timing')
    assert timing['completed'] is (failure == 'schema_reject')
    assert timing['duration_ms'] == (2000 if failure == 'schema_reject' else 5000)
    ends = [row for row in session._trace_events if row['event'] == 'tool_end']
    terminal = next(row for row in session._trace_events if row['event'] == 'turn_timing')
    assert terminal['boundary'] == 'session_end'
    if failure == 'schema_reject':
        assert ends == []
        call = next(row for row in session._trace_events if row['event'] == 'tool_call')
        assert call['duration_ms'] == 0 and call['dispatch_executed'] is False
        assert terminal['post_ms'] is None
    else:
        assert len(ends) == 1 and ends[0]['completed'] is False
        assert ends[0]['duration_ms'] == terminal['tool_ms'] == 3000


class TestPartitionConditions:
    """Replicates the entry-condition logic used in Session.run() to
    ensure the parallelism only activates when all three conditions
    hold (flag + >1 call + all read-only)."""

    def _should_parallelize(self, cfg, tcs):
        return (
            cfg.parallel_readonly_enabled
            and len(tcs) > 1
            and all(tc.name in loop_mod._READONLY_TOOLS for tc in tcs)
        )

    def test_disabled_flag_blocks(self):
        cfg = make_config(parallel_readonly_enabled=False)
        tcs = [_FakeTC("1", "read", {"path": "a"}),
               _FakeTC("2", "read", {"path": "b"})]
        assert self._should_parallelize(cfg, tcs) is False

    def test_single_call_blocks(self):
        cfg = make_config(parallel_readonly_enabled=True)
        tcs = [_FakeTC("1", "read", {"path": "a"})]
        assert self._should_parallelize(cfg, tcs) is False

    def test_any_mutating_tool_blocks(self):
        cfg = make_config(parallel_readonly_enabled=True)
        tcs = [_FakeTC("1", "read", {"path": "a"}),
               _FakeTC("2", "write", {"path": "b", "content": "x"})]
        assert self._should_parallelize(cfg, tcs) is False

    def test_multiple_readonly_enabled_passes(self):
        cfg = make_config(parallel_readonly_enabled=True)
        tcs = [_FakeTC("1", "read", {"path": "a"}),
               _FakeTC("2", "grep", {"pattern": "x"})]
        assert self._should_parallelize(cfg, tcs) is True
