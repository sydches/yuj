"""Repeated inspections save executions, while changed inputs remain visible."""
import io
import json
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness import tools
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


@pytest.mark.parametrize('kind', ['read', 'grep', 'bash'])
@pytest.mark.parametrize('change', ['none', 'bytes', 'new_file', 'deleted_file'])
def test_only_unchanged_inspections_reuse_the_third_call(tmp_path, monkeypatch, kind, change):
    source = tmp_path / 'src'
    source.mkdir()
    path = source / 'sample.py'
    path.write_text('alpha = 1\n')
    args = {'path': 'src/sample.py'} if kind == 'read' else (
        {'pattern': 'alpha', 'path': 'src'} if kind == 'grep' else
        {'cmd': 'grep -r "alpha" --include="*.py" src/ | head -20'})
    target = {'read': 'read', 'grep': 'grep_files', 'bash': 'bash'}[kind]
    original = getattr(tools, target)
    executions = []
    def execute(*a, **kw):
        executions.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(tools, target, execute)
    cfg = make_config(max_turns=4, sandbox_bash=False, duplicate_guard_enabled=True,
                      duplicate_warn_count=2, duplicate_abort=2, loop_detect_enabled=False,
                      tools_output_dedup_enabled=False, done_guard_enabled=False,
                      guardrails_arm_after_turn=0)
    client = MagicMock()
    client.is_replay = False
    turns = []
    def chat(*a, **kw):
        turn = len(turns)
        turns.append(1)
        if turn == 2:
            if change == 'bytes':
                path.write_text('alpha = 2\n')
            elif change == 'new_file':
                (source / 'new.py').write_text('alpha = 3\n')
            elif change == 'deleted_file':
                path.unlink()
        return TurnResult(content=None, tool_calls=[ToolCall(str(turn), kind, args)],
                          finish_reason='tool_calls', usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': None}
    trace = io.StringIO()
    session = Session(cfg, client, 'Follow the task.', 'Inspect the source.', str(tmp_path), trace_file=trace)
    result = session.run()
    assert result.finish_reason == 'max_turns'  # The legacy abort cannot end repetition.
    unchanged = change == 'none' or kind == 'read' and change == 'new_file'
    assert len(executions) == (2 if unchanged else 4)
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    reuse = [row for row in rows if row.get('event') == 'tool_call' and row.get('observation_reuse')]
    assert len(reuse) == (2 if unchanged else 0)
    warnings = [row for row in rows if row.get('event') == 'duplicate_observation_check'
                and row.get('action') == 'warn']
    assert len(warnings) == (1 if unchanged else 2)


@pytest.mark.parametrize('command', [
    'printf x >> changes.txt',
    'grep alpha src/sample.py > result.txt',
    'grep alpha src/sample.py && printf x >> changes.txt',
    'rg --pre "touch marker" alpha src/',
    'python -c "print(1)"',
    'grep alpha ../outside.py',
    'grep -if/tmp/patterns src/',
    './grep alpha src/',
    '/tmp/grep alpha src/',
])
def test_effects_and_unbound_searches_are_not_reusable(command):
    from scripts.llm_solver.harness.read_reuse import source_search
    parsed = source_search(command)
    assert parsed is None or '..' in parsed.paths[0]


@pytest.mark.parametrize('mode', ['disabled', 'pending', 'batch', 'symlink', 'quiet', 'leave_quiet'])
def test_reuse_requires_enabled_single_completed_bound_inspections(tmp_path, monkeypatch, mode):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'a.py').write_text('alpha = 1\n')
    if mode == 'symlink':
        (source / 'alias.py').symlink_to(source / 'a.py')
    original = tools.grep_files
    executions = []
    def execute(*args, **kwargs):
        executions.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(tools, 'grep_files', execute)
    cfg = make_config(max_turns=4, sandbox_bash=False, done_guard_enabled=False,
                      tools_background_enabled=mode == 'pending',
                      duplicate_guard_enabled=mode != 'disabled', duplicate_warn_count=2,
                      guardrails_arm_after_turn=10 if mode == 'quiet' else 2 if mode == 'leave_quiet' else 0)
    client = MagicMock()
    client.is_replay = False
    calls = [ToolCall('g', 'grep', {'pattern': 'alpha', 'path': 'src'})]
    if mode == 'batch':
        calls.append(ToolCall('r', 'read', {'path': 'src/a.py'}))
    client.chat.return_value = TurnResult(content=None, tool_calls=calls,
        finish_reason='tool_calls', usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': None}
    session = Session(cfg, client, 'system', 'task', str(tmp_path))
    if mode == 'pending':
        monkeypatch.setattr(session._process_manager, 'has_pending_observations', lambda: True)
    assert session.run().finish_reason == 'max_turns'
    assert len(executions) == 4
    if mode == 'leave_quiet':
        assert session._guards.duplicate_warned


def test_a_change_during_reuse_validation_runs_the_search_again(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.read_reuse import ReuseCall
    original = ReuseCall.lookup
    changes = []
    def lookup(self, cwd, cfg):
        result = original(self, cwd, cfg)
        if result is not None and not changes:
            (tmp_path / 'src' / 'sample.py').write_text('alpha = 2\n')
            changes.append(1)
        return result
    monkeypatch.setattr(ReuseCall, 'lookup', lookup)
    # Reuse is attempted on turn 2, invalidated by the second inventory,
    # and the fresh result is returned by the ordinary dispatcher.
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'sample.py').write_text('alpha = 1\n')
    cfg = make_config(max_turns=3, sandbox_bash=False, duplicate_guard_enabled=True,
                      duplicate_warn_count=2, guardrails_arm_after_turn=0)
    client = MagicMock()
    client.is_replay = False
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall('g', 'grep', {'pattern': 'alpha', 'path': 'src'})],
        finish_reason='tool_calls', usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': None}
    trace = io.StringIO()
    Session(cfg, client, 'system', 'task', str(tmp_path), trace_file=trace).run()
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    calls = [row for row in rows if row.get('event') == 'tool_call']
    assert changes == [1]
    assert not calls[-1].get('observation_reuse')
    assert 'alpha = 2' in calls[-1]['output_snippet']


@pytest.mark.parametrize('retained', [True, False])
def test_context_projection_must_retain_the_answer_before_reuse(tmp_path, monkeypatch, retained):
    answer = 'first line\nsecond line\nthird line'
    (tmp_path / 'a.py').write_text(answer)
    original_read = tools.read
    executions = []
    def read(*args, **kwargs):
        executions.append(1)
        return original_read(*args, **kwargs)
    monkeypatch.setattr(tools, 'read', read)
    cfg = make_config(max_turns=3, duplicate_guard_enabled=True,
                      duplicate_warn_count=2, guardrails_arm_after_turn=0,
                      tools_output_dedup_enabled=False)
    client = MagicMock()
    client.is_replay = False
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall('r', 'read', {'path': 'a.py'})],
        finish_reason='tool_calls', usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': None}
    session = Session(cfg, client, 'system', 'task', str(tmp_path))
    render = session.context.get_messages
    def projected():
        messages = render()
        if not retained and len(executions) == 2:
            return [{**m, 'content': '[older result shortened]'} if m.get('role') == 'tool'
                    else m for m in messages]
        return messages
    monkeypatch.setattr(session.context, 'get_messages', projected)
    assert session.run().finish_reason == 'max_turns'
    assert len(executions) == (2 if retained else 3)
