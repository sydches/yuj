"""Tests for the keyword-triggered injection subsystem."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from llm_solver.harness.injections import (
    Injection,
    InjectionState,
    fire_candidates,
    load_injections,
    match,
    parse_injection,
)


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
