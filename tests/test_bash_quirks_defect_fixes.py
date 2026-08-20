"""Regression tests for refusals and multi-line command rewrites."""
import subprocess

from scripts.llm_solver.bash_quirks._forbidden import load_forbidden_rules
from scripts.llm_solver.bash_quirks._rewrites import load_universal_rewrites
from scripts.llm_solver.bash_quirks.transforms import rewrite_command

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


def test_rule_log_records_kind():
    log = []
    rewrite_command(HEREDOC_WRITE, None, universal_rewrites=UR,
                    forbidden_rules=FR, rule_log=log)
    assert log == [{"kind": "forbidden"}]
    log = []
    rewrite_command("pip install requests", None, universal_rewrites=UR,
                    forbidden_rules=FR, rule_log=log)
    assert log and log[0]["kind"] == "universal"
