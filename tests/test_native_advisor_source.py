"""Advisor priorities come from the same task view as tool reads."""
from pathlib import Path

import pytest

from tests.test_task_files import bwrap, namespace_files
from test_advisor import _ScriptedClient, _advisor_config, _record_primary_turn, _turn
from scripts.llm_solver.harness._advisor_support import SYSTEM_PROMPT, advisor_system_prompt
from scripts.llm_solver.harness.loop import Session


@pytest.mark.parametrize('source_kind', ['regular', 'internal_link', 'missing',
                                        'denied', 'ignored', 'ignored_link', 'external_link', 'reserved_link'])
def test_native_watchdog_reaches_scripted_advisor_request(
    bwrap, tmp_path, monkeypatch, source_kind,
):
    from scripts.llm_solver.harness import task_file_runtime
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    target = Path(str(files.root))
    (target / 'WATCHDOG.md').write_text('HIDDEN_HOST_PRIORITIES')
    native = source / 'WATCHDOG.md'
    body = 'Native priority 文é\n' * 2000 + 'FINAL_NATIVE_RULE'
    if source_kind in {'regular', 'denied', 'ignored'}:
        native.write_text(body)
        if source_kind == 'denied':
            native.chmod(0)
    elif source_kind in {'internal_link', 'ignored_link', 'reserved_link'}:
        name = 'transcript.log' if source_kind == 'reserved_link' else 'review.md'
        (source / name).write_text(body)
        native.symlink_to(name)
    elif source_kind == 'external_link':
        outside = tmp_path / 'external.md'
        outside.write_text('OUTSIDE_PRIORITIES')
        native.symlink_to(outside)
    if source_kind in {'ignored', 'ignored_link'}:
        (source / '.yujignore').write_text('WATCHDOG.md\n')

    calls = []

    def selected_files(cwd, cfg, **options):
        assert str(cwd) == str(target)
        calls.append(options['environment'])
        return files

    monkeypatch.setattr(task_file_runtime, 'make_task_files', selected_files)
    client = _ScriptedClient(advisor=[_turn(content='NO_ADVISORY', reason='stop')])
    session = Session(_advisor_config(sandbox_bash=True, bwrap_bin=bwrap),
                      client, 'system', 'task', str(target))
    _record_primary_turn(session, 0)
    session._capture_advisor_turn(0, 'Observed a tool result.', [])
    assert not session._maybe_run_advisor(0)
    assert len(client.advisor_calls) == 1
    prompt = client.advisor_calls[0][0][0]['content']
    expected = SYSTEM_PROMPT
    if source_kind in {'regular', 'internal_link'}:
        expected += '\nRepository-specific review priorities from WATCHDOG.md:\n' + body
    assert prompt == expected
    assert 'HIDDEN_HOST_PRIORITIES' not in prompt and 'OUTSIDE_PRIORITIES' not in prompt
    assert calls and dict(calls[-1]) == dict(session._effective_env)


def test_advisor_does_not_read_after_execution_allowance_expires(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness.time_budget import BudgetExhausted, run_time_budget
    (tmp_path / 'WATCHDOG.md').write_text('NATIVE_PRIORITY')
    session = Session(_advisor_config(sandbox_bash=True, bwrap_bin=bwrap),
                      _ScriptedClient(), 'system', 'task', str(tmp_path))
    clock = [10.0]
    monkeypatch.setattr('scripts.llm_solver.harness.time_budget.time.monotonic', lambda: clock[0])
    with run_time_budget(1):
        clock[0] = 12.0
        with pytest.raises(BudgetExhausted):
            advisor_system_prompt(session, ignore_policy=session._advisor._ignore_policy)
