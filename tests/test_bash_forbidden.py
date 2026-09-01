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
    names = {r.name for r in rules}
    assert names == {"cd_root", "cd_home_other"}


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


@pytest.mark.parametrize(
    "cmd",
    [
        "cat > ./mlflow/x.py",
        "sed -i 's/foo/bar/g' ./mlflow/x.py",
        "echo content | tee ./mlflow/y.py",
        "echo 'x = 1' > ./mlflow/z.py",
        "cat foo >> ./pyproject.toml",
        "printf 'a' > setup.cfg",
        (
            "python3 << EOF\n"
            "with open('./mlflow/x.py', 'w') as f:\n"
            "    f.write('def foo(): pass')\n"
            "EOF"
        ),
    ],
)
def test_real_rules_allow_sandboxed_file_writes(cmd):
    rules = load_forbidden_rules()
    assert apply_forbidden(cmd, rules) == cmd


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
