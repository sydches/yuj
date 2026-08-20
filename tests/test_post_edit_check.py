"""Tests for the declarative post-edit check subsystem.

Covers predicate safe-eval, trigger matching, and the three on_fail
modes (append / warn / block). Validator commands run unsandboxed so
the tests don't need bwrap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver._shared.post_edit_spec import (
    PostEditCheckSpec,
    parse_post_edit_check_spec,
)
from llm_solver.harness.post_edit import (
    PredicateError,
    eval_when,
    run_post_edit_checks,
)
from llm_solver.harness.tools import edit, write


# ── Predicate safe-eval ──────────────────────────────────────────────────

class TestEvalWhen:

    def test_empty_predicate_is_true(self):
        assert eval_when("", path="f.py", ext=".py") is True

    def test_ext_equality(self):
        assert eval_when("ext == '.py'", path="f.py", ext=".py") is True
        assert eval_when("ext == '.py'", path="f.txt", ext=".txt") is False

    def test_path_startswith(self):
        assert eval_when(
            "path.startswith('src/')", path="src/a.py", ext=".py",
        ) is True

    def test_boolean_ops(self):
        assert eval_when(
            "ext == '.py' and not path.startswith('tests/')",
            path="src/a.py", ext=".py",
        ) is True
        assert eval_when(
            "ext == '.py' and not path.startswith('tests/')",
            path="tests/a.py", ext=".py",
        ) is False

    def test_rejects_disallowed_name(self):
        with pytest.raises(PredicateError):
            eval_when("__import__('os')", path="f.py", ext=".py")

    def test_rejects_function_call_on_non_whitelist_method(self):
        with pytest.raises(PredicateError):
            eval_when("path.format(0)", path="f.py", ext=".py")

    def test_rejects_arithmetic(self):
        with pytest.raises(PredicateError):
            eval_when("1 + 1 == 2", path="f.py", ext=".py")


# ── Check list semantics ─────────────────────────────────────────────────

def _cfg(checks: list, **extra) -> object:
    base = dict(
        post_edit_check_enabled=True,
        post_edit_check_timeout=5,
        post_edit_checks=checks,
        sandbox_bash=False,
    )
    base.update(extra)
    return make_config(**base)


PY_SYNTAX = "python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' {path}"


class TestRunPostEditChecks:

    def test_disabled_is_noop(self, tmp_path):
        cfg = make_config(post_edit_check_enabled=False, sandbox_bash=False)
        result = write("f.py", "oops(:", cwd=str(tmp_path), cfg=cfg)
        assert "post-edit check" not in result

    def test_empty_list_is_noop(self, tmp_path):
        cfg = _cfg([])
        result = write("f.py", "oops(:", cwd=str(tmp_path), cfg=cfg)
        assert "post-edit check" not in result

    def test_trigger_filter_skips_non_matching(self, tmp_path):
        """A write-only check should not fire on edit (and vice versa)."""
        cfg = _cfg([{
            "name": "write-only-check",
            "trigger": "write",
            "when": "",
            "cmd": "false",
            "on_fail": "append",
        }])
        (tmp_path / "f.py").write_text("ok\n")
        result = edit("f.py", "ok", "still_ok", cwd=str(tmp_path), cfg=cfg)
        assert "post-edit check" not in result

    def test_when_predicate_skips_non_matching(self, tmp_path):
        cfg = _cfg([{
            "name": "only-txt",
            "trigger": "edit|write",
            "when": "ext == '.txt'",
            "cmd": "false",
            "on_fail": "append",
        }])
        result = write("f.py", "fine", cwd=str(tmp_path), cfg=cfg)
        assert "post-edit check" not in result


class TestAppendMode:

    def test_append_on_failure(self, tmp_path):
        cfg = _cfg([{
            "name": "py-syntax",
            "trigger": "edit|write",
            "when": "ext == '.py'",
            "cmd": PY_SYNTAX,
            "on_fail": "append",
        }])
        result = write("f.py", "def f(:\n    pass\n",
                       cwd=str(tmp_path), cfg=cfg)
        assert "post-edit check 'py-syntax' failed" in result
        assert result.startswith("OK")

    def test_append_success_is_silent(self, tmp_path):
        cfg = _cfg([{
            "name": "py-syntax",
            "trigger": "edit|write",
            "when": "ext == '.py'",
            "cmd": PY_SYNTAX,
            "on_fail": "append",
        }])
        result = write("f.py", "def f():\n    pass\n",
                       cwd=str(tmp_path), cfg=cfg)
        assert "post-edit check" not in result
        assert result.startswith("OK")


class TestBlockMode:

    def test_block_reverts_new_file_creation(self, tmp_path):
        cfg = _cfg([{
            "name": "py-syntax",
            "trigger": "edit|write",
            "when": "ext == '.py'",
            "cmd": PY_SYNTAX,
            "on_fail": "block",
        }])
        result = write("f.py", "def f(:", cwd=str(tmp_path), cfg=cfg)
        assert result.startswith("ERROR:")
        assert "blocked by post-edit check" in result
        assert not (tmp_path / "f.py").exists()

    def test_block_reverts_overwrite_to_prior_content(self, tmp_path):
        (tmp_path / "f.py").write_text("def original():\n    pass\n")
        cfg = _cfg([{
            "name": "py-syntax",
            "trigger": "edit|write",
            "when": "ext == '.py'",
            "cmd": PY_SYNTAX,
            "on_fail": "block",
        }])
        result = write("f.py", "def broken(:", cwd=str(tmp_path), cfg=cfg)
        assert result.startswith("ERROR:")
        assert (tmp_path / "f.py").read_text() == "def original():\n    pass\n"

    def test_block_reverts_edit_to_prior_content(self, tmp_path):
        (tmp_path / "f.py").write_text("def f():\n    return 1\n")
        cfg = _cfg([{
            "name": "py-syntax",
            "trigger": "edit|write",
            "when": "ext == '.py'",
            "cmd": PY_SYNTAX,
            "on_fail": "block",
        }])
        result = edit("f.py", "return 1", "return 1 +",
                      cwd=str(tmp_path), cfg=cfg)
        assert result.startswith("ERROR:")
        assert (tmp_path / "f.py").read_text() == "def f():\n    return 1\n"


class TestWarnMode:

    def test_warn_is_append_shape_with_different_ledger_mechanism(self, tmp_path):
        """For now `warn` appends like `append` at the tool level — the
        distinction is carried on the ledger, not in the user-visible
        output. Verifies appended text shape is unchanged."""
        cfg = _cfg([{
            "name": "py-syntax",
            "trigger": "edit|write",
            "when": "ext == '.py'",
            "cmd": PY_SYNTAX,
            "on_fail": "warn",
        }])
        result = write("f.py", "oops(:", cwd=str(tmp_path), cfg=cfg)
        assert "post-edit check 'py-syntax' failed" in result
        assert result.startswith("OK")


class TestSchemaValidation:

    def test_parse_check_spec_returns_mechanical_object(self):
        spec = parse_post_edit_check_spec({
            "name": "py-syntax",
            "trigger": "edit|write|apply_patch",
            "when": "ext == '.py'",
            "cmd": "python -m py_compile {path}",
            "on_fail": "block",
        })
        assert isinstance(spec, PostEditCheckSpec)
        assert spec.name == "py-syntax"
        assert spec.triggers == frozenset({"edit", "write", "apply_patch"})
        assert spec.when == "ext == '.py'"
        assert spec.cmd == "python -m py_compile {path}"
        assert spec.on_fail == "block"

    def test_missing_required_key_raises(self, tmp_path):
        cfg = _cfg([{
            "name": "bad", "trigger": "edit|write",
            # missing cmd and on_fail
            "when": "",
        }])
        with pytest.raises(ValueError, match="missing required key"):
            run_post_edit_checks(
                str(tmp_path / "x.py"), cwd=str(tmp_path),
                cfg=cfg, trigger="write",
            )

    def test_invalid_on_fail_raises(self, tmp_path):
        (tmp_path / "f.py").write_text("x")
        cfg = _cfg([{
            "name": "bad",
            "trigger": "edit|write",
            "when": "",
            "cmd": "false",
            "on_fail": "shrug",
        }])
        with pytest.raises(ValueError, match="invalid on_fail"):
            run_post_edit_checks(
                "f.py", cwd=str(tmp_path),
                cfg=cfg, trigger="write",
            )
