"""Regression tests for refusals and safe command rewrites."""
import re
import subprocess

from scripts.llm_solver.bash_quirks._forbidden import load_forbidden_rules
from scripts.llm_solver.bash_quirks._rewrites import load_universal_rewrites
from scripts.llm_solver.bash_quirks.transforms import OutputControl, rewrite_command

UR = load_universal_rewrites()
FR = load_forbidden_rules()

HEREDOC_WRITE = """python << 'EOF'
# make the edit and write it back
with open('/tmp/x.py', 'w') as f:
    f.write('x')
EOF"""

HEREDOC_READ = """python << 'EOF'
# make the edit in memory only
with open('/tmp/x.py', 'r') as f:
    content = f.read()
print(content)
EOF"""


def test_refusal_surfaces_reason_to_model():
    out = rewrite_command(HEREDOC_WRITE, None, universal_rewrites=UR,
                          forbidden_rules=FR)
    assert out.startswith('echo "[HARNESS refused')
    # and the refusal actually prints the reason and exits nonzero
    r = subprocess.run(out, shell=True, capture_output=True, text=True)
    assert r.returncode != 0
    assert "HARNESS refused" in r.stderr


def test_refusal_shell_safe_despite_quotes_in_reason():
    out = rewrite_command(HEREDOC_WRITE, None, universal_rewrites=UR,
                          forbidden_rules=FR)
    r = subprocess.run(out, shell=True, capture_output=True, text=True)
    # no shell parse error text
    assert "unexpected" not in r.stderr.lower()


def test_multiline_commands_never_get_flag_appends():
    out = rewrite_command(HEREDOC_READ, None, universal_rewrites=UR,
                          forbidden_rules=FR)
    assert out == HEREDOC_READ  # 'make' in prose must not append -s


def test_single_line_rewrites_still_work():
    out = rewrite_command("pip install requests", None,
                          universal_rewrites=UR, forbidden_rules=FR)
    assert out != "pip install requests"  # -q (or similar) appended


def test_rewrite_targets_matching_compound_fragment():
    assert rewrite_command(
        "pip install demo && echo done", None, universal_rewrites=UR
    ) == "pip install demo -q && echo done"
    assert rewrite_command(
        "echo ok | pip install demo", None, universal_rewrites=UR
    ) == "echo ok | pip install demo -q"


def test_quoted_command_mentions_are_not_rewritten():
    for command in (
        'echo "pip install demo"',
        'grep "npm install" README.md',
        'echo "make all"',
        "which make",
        'echo "x && pip install demo"',
    ):
        assert rewrite_command(command, None, universal_rewrites=UR) == command


def test_rewrites_do_not_inject_pipelines_or_result_caps():
    assert rewrite_command(
        "npx tsc --noEmit", None, universal_rewrites=UR
    ) == "npx tsc --noEmit --pretty false"
    assert rewrite_command(
        "cmake --build build", None, universal_rewrites=UR
    ) == "cmake --build build"
    assert rewrite_command("rg pattern .", None, universal_rewrites=UR) == "rg pattern ."


def test_task_flag_targets_test_fragment_only():
    oc = OutputControl(
        failure_only_flag="--tb=short -q",
        passed_marker="PASSED",
        failed_marker="FAILED",
        verification_patterns=(re.compile(r"(?:^|[\s/'\"])(?:pytest)\b"),),
    )
    assert rewrite_command("cd /repo && pytest tests", oc) == (
        "cd /repo && pytest tests --tb=short -q"
    )
    command = 'echo "pytest tests && echo done"'
    assert rewrite_command(command, oc) == command


def test_rule_log_records_kind():
    log = []
    rewrite_command(HEREDOC_WRITE, None, universal_rewrites=UR,
                    forbidden_rules=FR, rule_log=log)
    assert log == [{"kind": "forbidden"}]
    log = []
    rewrite_command("pip install requests", None, universal_rewrites=UR,
                    forbidden_rules=FR, rule_log=log)
    assert log and log[0]["kind"] == "universal"
