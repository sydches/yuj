"""Tests for keyword- and path-triggered conditional injection rules."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from llm_solver.harness.injections import (
    Injection,
    InjectionState,
    fire_candidates,
    fire_path_candidates,
    load_injections,
    match,
    parse_injection,
)
from llm_solver.harness.loop import Session
from llm_solver.server.types import ToolCall, TurnResult, Usage
from _config_helpers import make_config


def _write(dir_: Path, name: str, body: str) -> Path:
    p = dir_ / name
    p.write_text(body)
    return p


# ── parse_injection ─────────────────────────────────────────────────────

class TestParseInjection:

    def test_parses_keyword_fragment(self):
        text = (
            '+++\n'
            'name = "pytest-hint"\n'
            'trigger = "keyword"\n'
            'keywords = ["pytest"]\n'
            'fire_once = true\n'
            '+++\n'
            '\n'
            'Use pytest -q for terse output.\n'
        )
        inj = parse_injection(text, source_path="x.md")
        assert inj.name == "pytest-hint"
        assert inj.trigger == "keyword"
        assert inj.keywords == ("pytest",)
        assert inj.fire_once is True
        assert "terse output" in inj.body

    def test_parses_always_fragment(self):
        text = (
            '+++\n'
            'name = "git-note"\n'
            'trigger = "always"\n'
            '+++\n'
            'git is available via bash.\n'
        )
        inj = parse_injection(text, source_path="x.md")
        assert inj.trigger == "always"
        assert inj.keywords == ()
        assert inj.fire_once is True  # default

    def test_missing_frontmatter_raises(self):
        with pytest.raises(ValueError, match="missing"):
            parse_injection("just body text", source_path="x.md")

    def test_missing_name_raises(self):
        text = '+++\ntrigger = "always"\n+++\nbody\n'
        with pytest.raises(ValueError, match="name"):
            parse_injection(text, source_path="x.md")

    def test_invalid_trigger_raises(self):
        text = (
            '+++\n'
            'name = "x"\n'
            'trigger = "sometimes"\n'
            '+++\n'
            'body\n'
        )
        with pytest.raises(ValueError, match="invalid trigger"):
            parse_injection(text, source_path="x.md")

    def test_keyword_trigger_without_keywords_raises(self):
        text = (
            '+++\n'
            'name = "x"\n'
            'trigger = "keyword"\n'
            '+++\n'
            'body\n'
        )
        with pytest.raises(ValueError, match="non-empty keywords"):
            parse_injection(text, source_path="x.md")

    def test_parses_paths_keywords_and_repeat(self):
        text = (
            '+++\n'
            'name = "python-guidance"\n'
            'paths = ["./src/**/*.py"]\n'
            'keywords = ["pytest"]\n'
            'repeat = true\n'
            '+++\n'
            'Use the repository Python conventions.\n'
        )
        inj = parse_injection(text, source_path="x.md")
        assert inj.paths == ("src/**/*.py",)
        assert inj.keywords == ("pytest",)
        assert inj.repeat is True
        assert inj.fire_once is False

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ('paths = "src/*.py"', "paths"),
            ('paths = [""]', "paths"),
            ('paths = [1]', "paths"),
            ('keywords = "pytest"', "keywords"),
            ('keywords = [1]', "keywords"),
            ('repeat = "once"', "repeat"),
            ('repeat = 1', "repeat"),
        ],
    )
    def test_conditional_fields_reject_wrong_types(self, field, expected):
        text = f'+++\nname = "bad"\n{field}\n+++\nbody\n'
        with pytest.raises(ValueError, match=expected):
            parse_injection(text, source_path="bad.md")

    @pytest.mark.parametrize("pattern", ["/etc/*.conf", "../secret/*.md"])
    def test_paths_must_stay_project_relative(self, pattern):
        text = (
            '+++\n'
            'name = "bad-path"\n'
            f'paths = ["{pattern}"]\n'
            '+++\nbody\n'
        )
        with pytest.raises(ValueError, match="project root"):
            parse_injection(text, source_path="bad.md")

    def test_repeat_and_legacy_fire_once_are_ambiguous(self):
        text = (
            '+++\nname = "bad"\nkeywords = ["pytest"]\n'
            'repeat = true\nfire_once = false\n+++\nbody\n'
        )
        with pytest.raises(ValueError, match="not both"):
            parse_injection(text, source_path="bad.md")

    def test_explicit_path_trigger_requires_paths(self):
        text = '+++\nname = "bad"\ntrigger = "path"\n+++\nbody\n'
        with pytest.raises(ValueError, match="non-empty paths"):
            parse_injection(text, source_path="bad.md")


# ── LEAKAGE_RULES guard ─────────────────────────────────────────────────

class TestLeakageGuard:

    def test_task_id_in_body_rejected(self):
        text = (
            '+++\n'
            'name = "bad"\n'
            'trigger = "always"\n'
            '+++\n'
            'See pypa__packaging.013f3b03 for the fix pattern.\n'
        )
        with pytest.raises(ValueError, match="task-id"):
            parse_injection(text, source_path="x.md")

    def test_task_id_in_keyword_rejected(self):
        text = (
            '+++\n'
            'name = "bad"\n'
            'trigger = "keyword"\n'
            'keywords = ["django__django"]\n'
            '+++\n'
            'body\n'
        )
        with pytest.raises(ValueError, match="task-id"):
            parse_injection(text, source_path="x.md")

    def test_framework_name_alone_accepted(self):
        text = (
            '+++\n'
            'name = "ok"\n'
            'trigger = "keyword"\n'
            'keywords = ["django", "pytest", "sympy"]\n'
            '+++\n'
            'django admin: use manage.py. pytest: -q is terse.\n'
        )
        inj = parse_injection(text, source_path="x.md")
        assert inj.keywords == ("django", "pytest", "sympy")

    def test_single_underscore_accepted(self):
        text = (
            '+++\n'
            'name = "ok"\n'
            'trigger = "always"\n'
            '+++\n'
            'dunder methods like __init__ and test_name are fine.\n'
        )
        inj = parse_injection(text, source_path="x.md")
        assert "test_name" in inj.body


# ── match ───────────────────────────────────────────────────────────────

class TestMatch:

    def _mk(self, trigger="keyword", keywords=("pytest",)):
        return Injection(
            name="x", trigger=trigger, keywords=keywords,
            fire_once=True, body="body", source_path="x.md",
        )

    def test_keyword_substring_case_insensitive(self):
        inj = self._mk()
        assert match(inj, "I will run PyTest now") is True
        assert match(inj, "pytest run complete") is True

    def test_keyword_miss(self):
        inj = self._mk()
        assert match(inj, "nothing to see here") is False

    def test_always_always_matches(self):
        inj = self._mk(trigger="always", keywords=())
        assert match(inj, "") is True
        assert match(inj, "anything") is True


# ── fire_candidates ─────────────────────────────────────────────────────

class TestFireCandidates:

    def test_fire_once_respects_state(self):
        inj = Injection(
            name="p", trigger="keyword", keywords=("pytest",),
            fire_once=True, body="b", source_path="x",
        )
        state = InjectionState()
        out1 = fire_candidates([inj], text="pytest run", state=state)
        assert len(out1) == 1
        assert "p" in state.fired_names
        out2 = fire_candidates([inj], text="pytest again", state=state)
        assert out2 == []

    def test_fire_always_refires_when_fire_once_false(self):
        inj = Injection(
            name="p", trigger="keyword", keywords=("pytest",),
            fire_once=False, body="b", source_path="x",
        )
        state = InjectionState()
        out1 = fire_candidates([inj], text="pytest", state=state)
        out2 = fire_candidates([inj], text="pytest again", state=state)
        assert len(out1) == 1
        assert len(out2) == 1

    def test_always_fires_once_per_session(self):
        inj = Injection(
            name="always", trigger="always", keywords=(),
            fire_once=True, body="b", source_path="x",
        )
        state = InjectionState()
        assert fire_candidates([inj], text="", state=state) == [inj]
        assert fire_candidates([inj], text="", state=state) == []


# ── path-triggered candidates ───────────────────────────────────────────

def _path_rule(*, pattern="src/**/*.py", repeat=None, name="python-rule"):
    return Injection(
        name=name,
        trigger="path",
        keywords=(),
        fire_once=True,
        body="Use the Python conventions.",
        source_path="rule.md",
        paths=(pattern,),
        repeat=repeat,
    )


class TestPathCandidates:

    @pytest.mark.parametrize(
        ("tool_name", "arguments", "operations"),
        [
            ("read", {"path": "src/main.py"}, ()),
            (
                "edit",
                {"path": "src/main.py", "old_str": "x", "new_str": "y"},
                (),
            ),
            ("write", {"path": "src/main.py", "content": "x"}, ()),
            (
                "apply_patch",
                {"patch": "*** Begin Patch\n*** End Patch"},
                (("update", "src/main.py"),),
            ),
            ("bash", {"cmd": "sed -n '1,20p' src/main.py"}, ()),
        ],
    )
    def test_file_tools_and_single_file_bash_fire(
        self, tmp_path, tool_name, arguments, operations,
    ):
        state = InjectionState()
        fired = fire_path_candidates(
            [_path_rule()],
            tool_name=tool_name,
            arguments=arguments,
            cwd=str(tmp_path),
            state=state,
            applied_operations=operations,
        )
        assert [(item.injection.name, item.path) for item in fired] == [
            ("python-rule", "src/main.py")
        ]

    def test_nonmatching_path_does_not_fire(self, tmp_path):
        fired = fire_path_candidates(
            [_path_rule()],
            tool_name="read",
            arguments={"path": "docs/guide.md"},
            cwd=str(tmp_path),
            state=InjectionState(),
        )
        assert fired == []

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat src/a.py src/b.py",
            "cat src/a.py | wc -l",
            "find src -name '*.py'",
        ],
    )
    def test_non_single_file_bash_does_not_fire(self, tmp_path, cmd):
        fired = fire_path_candidates(
            [_path_rule()],
            tool_name="bash",
            arguments={"cmd": cmd},
            cwd=str(tmp_path),
            state=InjectionState(),
        )
        assert fired == []

    def test_symlink_matches_resolved_target_and_traces_canonical_path(
        self, tmp_path,
    ):
        source = tmp_path / "src" / "main.py"
        source.parent.mkdir()
        source.write_text("print('ok')\n")
        (tmp_path / "alias.py").symlink_to(source.relative_to(tmp_path))
        fired = fire_path_candidates(
            [_path_rule()],
            tool_name="read",
            arguments={"path": "alias.py"},
            cwd=str(tmp_path),
            state=InjectionState(),
        )
        assert [item.path for item in fired] == ["src/main.py"]

    def test_default_once_and_per_rule_repeat(self, tmp_path):
        state = InjectionState()
        once = _path_rule()
        first = fire_path_candidates(
            [once], tool_name="read", arguments={"path": "src/main.py"},
            cwd=str(tmp_path), state=state,
        )
        second = fire_path_candidates(
            [once], tool_name="edit", arguments={"path": "src/main.py"},
            cwd=str(tmp_path), state=state,
        )
        assert len(first) == 1
        assert second == []

        repeating = _path_rule(repeat=True, name="repeating")
        repeating_state = InjectionState()
        assert len(fire_path_candidates(
            [repeating], tool_name="read", arguments={"path": "src/main.py"},
            cwd=str(tmp_path), state=repeating_state,
        )) == 1
        assert len(fire_path_candidates(
            [repeating], tool_name="edit", arguments={"path": "src/main.py"},
            cwd=str(tmp_path), state=repeating_state,
        )) == 1

    def test_global_path_repeat_applies_when_rule_omits_repeat(self, tmp_path):
        state = InjectionState()
        rule = _path_rule()
        for _ in range(2):
            fired = fire_path_candidates(
                [rule],
                tool_name="read",
                arguments={"path": "src/main.py"},
                cwd=str(tmp_path),
                state=state,
                path_rule_repeat=True,
            )
            assert len(fired) == 1


# ── load_injections ─────────────────────────────────────────────────────

class TestLoadInjections:

    def test_empty_dir_returns_empty_list(self, tmp_path):
        d = tmp_path / "injections"
        d.mkdir()
        assert load_injections(d) == []

    def test_missing_dir_returns_empty_list(self, tmp_path):
        assert load_injections(tmp_path / "does_not_exist") == []

    def test_loads_multiple_files_sorted(self, tmp_path):
        d = tmp_path / "inj"
        d.mkdir()
        _write(d, "b.md",
               '+++\nname = "B"\ntrigger = "always"\n+++\nbody-B\n')
        _write(d, "a.md",
               '+++\nname = "A"\ntrigger = "always"\n+++\nbody-A\n')
        loaded = load_injections(d)
        assert [i.name for i in loaded] == ["A", "B"]

    def test_malformed_conditional_file_errors_during_load(self, tmp_path):
        d = tmp_path / "injections"
        d.mkdir()
        _write(
            d,
            "bad.md",
            '+++\nname = "bad"\npaths = "src/*.py"\n+++\nbody\n',
        )
        with pytest.raises(ValueError, match="paths"):
            load_injections(d)

    def test_registry_logs_armed_rule_and_trigger(self, tmp_path, caplog):
        d = tmp_path / "injections"
        d.mkdir()
        _write(
            d,
            "python.md",
            '+++\nname = "python"\npaths = ["**/*.py"]\n+++\nbody\n',
        )
        caplog.set_level("INFO", logger="llm_solver.harness.injections")
        load_injections(d)
        assert "injection armed: rule=python triggers=path" in caplog.text


# ── format_block ────────────────────────────────────────────────────────

class TestFormatBlock:

    def test_envelope_shape(self):
        inj = Injection(
            name="pytest-hint", trigger="keyword", keywords=("pytest",),
            fire_once=True, body="Use -q", source_path="x",
        )
        out = inj.format_block()
        assert out.startswith('<injected-fragment source="pytest-hint">')
        assert out.endswith("</injected-fragment>")
        assert "Use -q" in out


# ── Session wiring (integration at the _apply_injections level) ─────────

class _FakeContext:
    """Minimal ContextManager stand-in for wiring tests."""
    def __init__(self, messages):
        self._messages = list(messages)
        self.added_user = []

    def get_messages(self):
        return list(self._messages)

    def add_user(self, text):
        self.added_user.append(text)
        self._messages.append({"role": "user", "content": text})


class _FakeSession:
    """Stand-in that borrows Session._apply_injections verbatim."""
    def __init__(self, injections, context):
        self._injections = list(injections)
        self._injection_state = InjectionState()
        self.context = context

    # Import the real method under test.
    from llm_solver.harness.loop import Session
    _apply_injections = Session._apply_injections


class TestSessionWiring:

    def test_keyword_match_triggers_add_user_call(self):
        inj = Injection(
            name="pytest-hint", trigger="keyword", keywords=("pytest",),
            fire_once=True, body="Use -q", source_path="x",
        )
        ctx = _FakeContext([
            {"role": "user", "content": "please run pytest"},
        ])
        s = _FakeSession([inj], ctx)
        s._apply_injections()
        assert len(ctx.added_user) == 1
        assert '<injected-fragment source="pytest-hint">' in ctx.added_user[0]
        assert "Use -q" in ctx.added_user[0]

    def test_no_match_no_addition(self):
        inj = Injection(
            name="pytest-hint", trigger="keyword", keywords=("pytest",),
            fire_once=True, body="Use -q", source_path="x",
        )
        ctx = _FakeContext([
            {"role": "user", "content": "build the project"},
        ])
        s = _FakeSession([inj], ctx)
        s._apply_injections()
        assert ctx.added_user == []

    def test_fire_once_across_multiple_calls(self):
        inj = Injection(
            name="pytest-hint", trigger="keyword", keywords=("pytest",),
            fire_once=True, body="Use -q", source_path="x",
        )
        ctx = _FakeContext([
            {"role": "user", "content": "pytest please"},
        ])
        s = _FakeSession([inj], ctx)
        s._apply_injections()
        s._apply_injections()
        s._apply_injections()
        assert len(ctx.added_user) == 1

    def test_scans_last_tool_message_when_newer_than_user(self):
        inj = Injection(
            name="pytest-hint", trigger="keyword", keywords=("pytest",),
            fire_once=True, body="Use -q", source_path="x",
        )
        ctx = _FakeContext([
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "listing"},
            {"role": "tool", "content": "found test_pytest.py"},
        ])
        s = _FakeSession([inj], ctx)
        s._apply_injections()
        assert len(ctx.added_user) == 1

    def test_empty_injections_list_noop(self):
        ctx = _FakeContext([{"role": "user", "content": "anything"}])
        s = _FakeSession([], ctx)
        s._apply_injections()
        assert ctx.added_user == []

    def test_always_fragment_fires_once_regardless_of_content(self):
        inj = Injection(
            name="always-on", trigger="always", keywords=(),
            fire_once=True, body="Session notice.", source_path="x",
        )
        ctx = _FakeContext([{"role": "user", "content": ""}])
        s = _FakeSession([inj], ctx)
        s._apply_injections()
        s._apply_injections()
        assert len(ctx.added_user) == 1


# ── full Session dispatch + trace/state artifact wiring ─────────────────

def _turn(*, tool_calls=(), content="", finish_reason="tool_calls"):
    return TurnResult(
        content=content,
        tool_calls=list(tool_calls),
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=10, completion_tokens=3),
    )


def _client(*turns):
    client = MagicMock()
    client.chat.side_effect = turns
    client.build_assistant_message.side_effect = lambda content, tool_calls: {
        "role": "assistant",
        "content": content,
    }
    return client


class TestConditionalRuntimeArtifacts:

    def test_session_startup_rejects_malformed_enabled_rule(self, tmp_path):
        injection_dir = tmp_path / ".harness" / "injections"
        injection_dir.mkdir(parents=True)
        _write(
            injection_dir,
            "bad.md",
            '+++\nname = "bad"\nrepeat = "yes"\n+++\nbody\n',
        )
        cfg = make_config(injections_enabled=True)
        with pytest.raises(ValueError, match="repeat"):
            Session(
                cfg,
                _client(),
                "system",
                "task",
                str(tmp_path),
                trace_file=StringIO(),
            )

    def test_path_rules_remain_dormant_without_path_switch(self, tmp_path):
        target = tmp_path / "src" / "main.py"
        target.parent.mkdir()
        target.write_text("VALUE = 1\n")
        client = _client(
            _turn(tool_calls=[ToolCall(
                id="read-1", name="read", arguments={"path": "src/main.py"},
            )]),
            _turn(content="done", finish_reason="stop"),
        )
        cfg = make_config(
            max_turns=2,
            injections_enabled=True,
            turn_snapshots_enabled=False,
        )
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=StringIO(),
            injections=[_path_rule()],
        )
        captured = []
        original_add = session.context.add_tool_result

        def capture(tool_call_id, result, **kwargs):
            captured.append(result)
            return original_add(tool_call_id, result, **kwargs)

        session.context.add_tool_result = capture
        with patch.object(
            session, "_get_server_ctx", return_value=cfg.context_size,
        ):
            session.run()

        assert "<injected-fragment" not in captured[0]
        assert not any(
            event.get("event") == "injection"
            for event in session._trace_events
        )

    def test_global_repeat_knob_refires_path_rule(self, tmp_path):
        target = tmp_path / "src" / "main.py"
        target.parent.mkdir()
        target.write_text("VALUE = 1\n")
        client = _client(
            _turn(tool_calls=[ToolCall(
                id="read-1", name="read", arguments={"path": "src/main.py"},
            )]),
            _turn(tool_calls=[ToolCall(
                id="read-2", name="read", arguments={"path": "src/main.py"},
            )]),
            _turn(content="done", finish_reason="stop"),
        )
        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            injections_enabled=True,
            injections_path_rules_enabled=True,
            injections_path_rule_repeat=True,
            turn_snapshots_enabled=False,
        )
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=StringIO(),
            injections=[_path_rule()],
        )
        captured = []
        original_add = session.context.add_tool_result

        def capture(tool_call_id, result, **kwargs):
            captured.append(result)
            return original_add(tool_call_id, result, **kwargs)

        session.context.add_tool_result = capture
        with patch.object(
            session, "_get_server_ctx", return_value=cfg.context_size,
        ):
            session.run()

        assert len(captured) == 2
        assert all('trigger="path"' in result for result in captured)
        assert sum(
            event.get("event") == "injection"
            for event in session._trace_events
        ) == 2

    def test_matching_read_appends_fragment_and_emits_raw_trace(self, tmp_path):
        target = tmp_path / "src" / "main.py"
        target.parent.mkdir()
        target.write_text("VALUE = 1\n")
        client = _client(
            _turn(tool_calls=[ToolCall(
                id="read-1", name="read", arguments={"path": "src/main.py"},
            )]),
            _turn(content="done", finish_reason="stop"),
        )
        cfg = make_config(
            max_turns=2,
            duplicate_abort=10,
            injections_enabled=True,
            injections_path_rules_enabled=True,
            turn_snapshots_enabled=False,
        )
        state_path = tmp_path / ".solver" / "state.json"
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=StringIO(),
            state_path=state_path,
            injections=[_path_rule()],
        )
        tool_results = []
        original_add = session.context.add_tool_result

        def capture(tool_call_id, result, **kwargs):
            tool_results.append(result)
            return original_add(tool_call_id, result, **kwargs)

        session.context.add_tool_result = capture
        with patch.object(
            session, "_get_server_ctx", return_value=cfg.context_size,
        ):
            result = session.run()

        assert result.finish_reason == "stop"
        assert 'rule="python-rule"' in tool_results[0]
        assert 'trigger="path" path="src/main.py"' in tool_results[0]
        assert tool_results[0].endswith("</injected-fragment>")

        injection_events = [
            event for event in session._trace_events
            if event.get("event") == "injection"
        ]
        assert [{
            key: event[key] for key in ("rule", "trigger", "path")
        } for event in injection_events] == [{
            "rule": "python-rule",
            "trigger": "path",
            "path": "src/main.py",
        }]
        tool_event = next(
            event for event in session._trace_events
            if event.get("event") == "tool_call"
        )
        assert "<injected-fragment" in tool_event["result_summary"]

        projected = json.loads(state_path.read_text())
        assert "injection" not in projected
        assert "<injected-fragment" in projected["trace"][0]["result"]

    @pytest.mark.parametrize(
        ("tool_name", "arguments", "expected_path"),
        [
            (
                "edit",
                {"path": "src/main.py", "old_str": "1", "new_str": "2"},
                "src/main.py",
            ),
            (
                "write",
                {"path": "src/new.py", "content": "VALUE = 2\n"},
                "src/new.py",
            ),
            (
                "apply_patch",
                {"patch": (
                    "*** Begin Patch\n"
                    "*** Update File: src/main.py\n"
                    "@@\n"
                    "-VALUE = 1\n"
                    "+VALUE = 2\n"
                    "*** End Patch"
                )},
                "src/main.py",
            ),
            ("bash", {"cmd": "cat src/main.py"}, "src/main.py"),
        ],
    )
    def test_other_supported_tools_append_in_real_dispatch(
        self, tmp_path, tool_name, arguments, expected_path,
    ):
        work = tmp_path / tool_name
        target = work / "src" / "main.py"
        target.parent.mkdir(parents=True)
        target.write_text("VALUE = 1\n")
        client = _client(
            _turn(tool_calls=[ToolCall(
                id=f"{tool_name}-1", name=tool_name, arguments=arguments,
            )]),
            _turn(content="done", finish_reason="stop"),
        )
        cfg = make_config(
            max_turns=2,
            duplicate_abort=10,
            injections_enabled=True,
            injections_path_rules_enabled=True,
            tools_apply_patch_enabled=True,
            tools_stale_guard_mode="off",
            turn_snapshots_enabled=False,
        )
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(work),
            trace_file=StringIO(),
            injections=[_path_rule()],
        )
        captured = []
        original_add = session.context.add_tool_result

        def capture(tool_call_id, result, **kwargs):
            captured.append(result)
            return original_add(tool_call_id, result, **kwargs)

        session.context.add_tool_result = capture
        with patch.object(
            session, "_get_server_ctx", return_value=cfg.context_size,
        ):
            session.run()

        assert 'trigger="path"' in captured[0]
        assert f'path="{expected_path}"' in captured[0]

    def test_nonmatching_read_stays_byte_free_of_fragment(self, tmp_path):
        target = tmp_path / "docs" / "guide.md"
        target.parent.mkdir()
        target.write_text("guide\n")
        client = _client(
            _turn(tool_calls=[ToolCall(
                id="read-doc", name="read", arguments={"path": "docs/guide.md"},
            )]),
            _turn(content="done", finish_reason="stop"),
        )
        cfg = make_config(
            max_turns=2,
            injections_enabled=True,
            injections_path_rules_enabled=True,
            turn_snapshots_enabled=False,
        )
        session = Session(
            cfg, client, "system", "task", str(tmp_path),
            trace_file=StringIO(), injections=[_path_rule()],
        )
        captured = []
        original_add = session.context.add_tool_result

        def capture(tool_call_id, result, **kwargs):
            captured.append(result)
            return original_add(tool_call_id, result, **kwargs)

        session.context.add_tool_result = capture
        with patch.object(
            session, "_get_server_ctx", return_value=cfg.context_size,
        ):
            session.run()
        assert "<injected-fragment" not in captured[0]
        assert not any(
            event.get("event") == "injection"
            for event in session._trace_events
        )

    def test_keyword_fire_emits_trace_with_empty_path(self, tmp_path):
        keyword = Injection(
            name="pytest-hint",
            trigger="keyword",
            keywords=("pytest",),
            fire_once=True,
            body="Use pytest -q.",
            source_path="keyword.md",
        )
        cfg = make_config(max_turns=1, injections_enabled=True)
        session = Session(
            cfg,
            _client(_turn(content="done", finish_reason="stop")),
            "system",
            "Please run pytest.",
            str(tmp_path),
            trace_file=StringIO(),
            injections=[keyword],
        )
        with patch.object(
            session, "_get_server_ctx", return_value=cfg.context_size,
        ):
            session.run()
        event = next(
            event for event in session._trace_events
            if event.get("event") == "injection"
        )
        assert {
            key: event[key] for key in ("rule", "trigger", "path")
        } == {
            "rule": "pytest-hint",
            "trigger": "keyword",
            "path": "",
        }

    def test_trace_schema_declares_injection_fields(self):
        from llm_solver.harness._loop.trace_schema import (
            TRACE_EVENT_REQUIRED_FIELDS,
        )
        assert TRACE_EVENT_REQUIRED_FIELDS["injection"] == frozenset({
            "session_number", "turn_number", "rule", "trigger", "path",
        })
