"""Test bash_quirks/forbidden.toml refusal layer (O2)."""
import re
from pathlib import Path

import pytest

from scripts.llm_solver.bash_quirks.transforms import (
    ForbiddenRule, apply_forbidden, load_forbidden_rules, rewrite_command,
)


def _is_refusal(out: str) -> bool:
    """A refusal reports its reason and keeps a machine marker."""
    return out.startswith('echo "[HARNESS refused') and "false  # [HARNESS:" in out


def _rule(name: str, pattern: str, reason: str) -> ForbiddenRule:
    return ForbiddenRule(name=name, pattern=re.compile(pattern, re.IGNORECASE), reason=reason)


def test_apply_forbidden_no_match_returns_unchanged():
    rules = [_rule("cd_root", r'^\s*cd\s+/\s*(?:&&|;|$)', "no")]
    assert apply_forbidden("ls -la", rules) == "ls -la"


def test_apply_forbidden_match_replaces_with_refusal():
    rules = [_rule("cd_root", r'^\s*cd\s+/\s*(?:&&|;|$)', "cd / not permitted")]
    out = apply_forbidden("cd / && python -c 'x'", rules)
    assert _is_refusal(out)
    assert "cd / not permitted" in out


def test_apply_forbidden_first_match_wins():
    rules = [
        _rule("a", r'^\s*cd\s+/(?:home|root)/', "outside cwd"),
        _rule("b", r'^\s*cd\s+/\s*&&', "cd / loud"),
    ]
    out = apply_forbidden("cd /home/example && ls", rules)
    assert "outside cwd" in out


def test_apply_forbidden_empty_rules_passthrough():
    assert apply_forbidden("cd / && ls", None) == "cd / && ls"
    assert apply_forbidden("cd / && ls", []) == "cd / && ls"


def test_load_forbidden_rules_from_real_file():
    rules = load_forbidden_rules()
    assert len(rules) >= 3
    names = {r.name for r in rules}
    assert "cd_root" in names
    assert "cd_home_other" in names


def test_real_rules_catch_known_leak_patterns():
    rules = load_forbidden_rules()
    # Refuse commands that leave the task directory.
    assert apply_forbidden("cd / && python -c 'import mlflow'", rules).startswith('echo "[HARNESS refused')
    assert apply_forbidden("cd /home/example/work && ls", rules).startswith('echo "[HARNESS refused')
    # And do NOT catch /tmp scratch use:
    assert apply_forbidden("cd /tmp && mkdir test", rules) == "cd /tmp && mkdir test"
    # Or normal commands:
    assert apply_forbidden("python3 -m pytest", rules) == "python3 -m pytest"


def test_rewrite_command_runs_forbidden_first():
    """When a forbidden rule fires, the rewrite skips quiet-flag injection."""
    rules = [_rule("cd_root", r'^\s*cd\s+/\s*&&', "cd / not permitted")]
    out = rewrite_command("cd / && pip install foo", oc=None, universal_rewrites=None,
                          forbidden_rules=rules)
    assert _is_refusal(out)
    assert "pip install" not in out  # original command body discarded


def test_rewrite_command_passes_through_when_no_forbidden_match():
    rules = [_rule("cd_root", r'^\s*cd\s+/\s*&&', "cd / not permitted")]
    out = rewrite_command("ls -la", oc=None, universal_rewrites=None, forbidden_rules=rules)
    assert out == "ls -la"


# ─── O8: bash file-write as covert mutation channel ─────────────────────

def test_real_rules_refuse_cat_redirect():
    rules = load_forbidden_rules()
    out = apply_forbidden("cat > ./mlflow/x.py", rules)
    assert _is_refusal(out)
    assert "mutation tracking" in out


def test_real_rules_refuse_sed_inplace():
    rules = load_forbidden_rules()
    out = apply_forbidden("sed -i 's/foo/bar/g' ./mlflow/x.py", rules)
    assert _is_refusal(out)


def test_real_rules_refuse_tee_to_relative_path():
    rules = load_forbidden_rules()
    out = apply_forbidden("echo content | tee ./mlflow/y.py", rules)
    assert _is_refusal(out)


def test_real_rules_refuse_shell_redirect_to_codefile():
    rules = load_forbidden_rules()
    for redirect in (
        "echo 'x = 1' > ./mlflow/z.py",
        "cat foo >> ./pyproject.toml",
        "printf 'a' > setup.cfg",
    ):
        out = apply_forbidden(redirect, rules)
        assert _is_refusal(out), f"failed to refuse: {redirect}"


def test_real_rules_refuse_python_heredoc_writing_to_file():
    rules = load_forbidden_rules()
    cmd = (
        "python3 << EOF\n"
        "with open('./mlflow/x.py', 'w') as f:\n"
        "    f.write('def foo(): pass')\n"
        "EOF"
    )
    out = apply_forbidden(cmd, rules)
    assert _is_refusal(out), f"failed to refuse python heredoc: {out[:80]}"


def test_real_rules_pass_through_python_heredoc_without_file_write():
    """`python3 -c` or `python3 << EOF` that only computes / prints
    without opening files in 'w'/'a' mode is fine — we only refuse the
    file-write channel."""
    rules = load_forbidden_rules()
    cmd = "python3 -c 'print(1+1)'"
    assert apply_forbidden(cmd, rules) == cmd
    cmd = (
        "python3 << EOF\n"
        "x = 1 + 1\n"
        "print(x)\n"
        "EOF"
    )
    assert apply_forbidden(cmd, rules) == cmd


def test_real_rules_pass_through_legitimate_pytest_invocations():
    rules = load_forbidden_rules()
    for ok in (
        "python3 -m pytest tests/",
        "ls -la tests/",
        "cat ./mlflow/x.py",
        "grep -r 'foo' ./mlflow/",
    ):
        assert apply_forbidden(ok, rules) == ok, f"false-positive on: {ok}"
