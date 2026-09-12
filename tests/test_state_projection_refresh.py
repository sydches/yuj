"""Live synchronous state remains identical to the canonical full projection."""
import json
from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness._loop import state_projection
from scripts.llm_solver.harness.state_writer import project
from tests._config_helpers import make_config


def session(tmp_path, imperative=True):
    return SimpleNamespace(_state_path=tmp_path / 'state.json', _trace_events=[],
        cfg=make_config(state_imperative_projection_enabled=imperative, tools_think_keep_turns=1))


def assert_current(value):
    state_projection.refresh_state(value)
    expected = project(value._trace_events, max_result_chars=value.cfg.max_output_chars,
                       imperative_projection=value.cfg.state_imperative_projection_enabled,
                       think_keep_turns=value.cfg.tools_think_keep_turns)
    actual = json.loads(value._state_path.read_text())
    assert actual == expected
    return actual


@pytest.mark.parametrize('imperative', [False, True])
def test_every_published_state_matches_full_projection(tmp_path, imperative):
    value = session(tmp_path, imperative)
    events = [
        dict(event='session_start', session_number=0, edit_format='edit'),
        dict(event='tool_call', session_number=0, turn_number=0, tool_name='think',
             args_summary='retained thought', reasoning='thought reasoning'),
        dict(event='tool_timing', session_number=0, turn_number=100),
        dict(event='turn_timing', session_number=0, turn_number=101),
        dict(event='tool_end', session_number=0, turn_number=102),
        dict(event='turn', session_number=0, turn_number=103),
        dict(event='tool_call', session_number=0, turn_number=2, tool_name='read', args_summary='file.py'),
        dict(event='rewind', session_number=0, from_turn=2, to_turn=0),
        dict(event='tool_timing', session_number=0, turn_number=1),
        dict(event='tool_call', session_number=0, turn_number=1, tool_name='edit', args_summary='file.py'),
        dict(event='rewind', session_number=0, from_turn=1, to_turn=2),
        dict(event='turn_timing', session_number=0, turn_number=3),
        dict(event='session_start', session_number=1, edit_format='apply_patch'),
        dict(event='tool_timing', session_number=1, turn_number=0),
        dict(event='unknown_future_event', session_number=2, turn_number=0),
        dict(event='tool_end', session_number=3, turn_number=0),
    ]
    for index, event in enumerate(events):
        value._trace_events.append(event)
        actual = assert_current(value)
        if 1 <= index <= 5:
            assert actual['trace'][0]['reasoning'] == 'thought reasoning'
        if index == 6:
            assert actual['trace'][0]['reasoning'] == ''


def test_only_safe_append_reuses_projection(tmp_path, monkeypatch):
    value = session(tmp_path)
    calls = []
    def recording_project(*args, **kwargs):
        calls.append(len(args[0]))
        return project(*args, **kwargs)
    monkeypatch.setattr(state_projection, 'project', recording_project)
    value._trace_events.append(dict(event='session_start', session_number=0))
    assert_current(value)
    value._trace_events.append(dict(event='tool_timing', session_number=0, turn_number=0))
    assert_current(value)
    assert calls == [1]
    value.cfg = make_config(state_imperative_projection_enabled=False, tools_think_keep_turns=0)
    value._trace_events.append(dict(event='tool_end', session_number=0))
    assert_current(value)
    assert calls == [1, 3]
    value._trace_events = list(value._trace_events)
    value._trace_events.append(dict(event='turn_timing', session_number=0))
    assert_current(value)
    assert calls == [1, 3, 4]
    value._trace_events[:] = value._trace_events[:1]
    assert_current(value)
    assert calls == [1, 3, 4, 1]
    value._trace_events[0] = dict(event='session_start', session_number=2)
    value._trace_events.append(dict(event='tool_timing', session_number=2))
    assert_current(value)
    assert calls == [1, 3, 4, 1, 2]


def test_disabled_projection_does_not_build_or_write(tmp_path, monkeypatch):
    value = session(tmp_path)
    value._state_path = None
    def unexpected(*args, **kwargs):
        pytest.fail('disabled projection performed work')
    monkeypatch.setattr(state_projection, 'project', unexpected)
    state_projection.refresh_state(value)
    assert not hasattr(value, '_state_projection_cache')
