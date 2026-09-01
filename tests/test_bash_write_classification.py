"""Shared bash write/mutation classifier behavior."""
from __future__ import annotations

from scripts.llm_solver.harness.bash_write_classification import (
    classify_bash_write,
    extract_source_write_paths,
    extract_workspace_write_paths,
    is_bash_action_write_like,
    is_bash_legacy_mutation_like,
    is_bash_workspace_mutation_like,
)


def test_python_read_heredoc_is_not_action_write_like():
    cmd = (
        "python3 << 'PY'\n"
        "with open('/testbed/src/app.py') as f:\n"
        "    print(f.read())\n"
        "PY"
    )

    classification = classify_bash_write(cmd)

    assert classification.action_write_like is False
    assert classification.legacy_mutation_like is False
    assert classification.workspace_write_like is False
    assert classification.workspace_write_paths == ()
    assert classification.source_write_like is False
    assert classification.source_write_paths == ()


def test_python_temp_replace_extracts_source_write_path():
    cmd = (
        "cd /testbed && python3 << 'PY'\n"
        "from pathlib import Path\n"
        "import os\n"
        "path = Path('src/app.py')\n"
        "tmp = path.with_name(path.name + '.tmp-write')\n"
        "tmp.write_text('x')\n"
        "os.replace(tmp, path)\n"
        "PY"
    )

    classification = classify_bash_write(cmd)

    assert classification.action_write_like is True
    assert classification.workspace_write_like is True
    assert classification.workspace_write_paths == ("src/app.py",)
    assert classification.source_write_like is True
    assert classification.source_write_paths == ("src/app.py",)


def test_test_paths_are_not_source_write_paths():
    assert extract_source_write_paths(
        "sed -i 's/a/b/' tests/test_app.py"
    ) == ()
    assert extract_workspace_write_paths(
        "sed -i 's/a/b/' tests/test_app.py"
    ) == ("tests/test_app.py",)


def test_non_python_source_extensions_are_source_write_paths():
    classification = classify_bash_write(
        "sed -i 's/old/new/' src/lib.rs pkg/server.go web/Button.tsx"
    )

    assert classification.action_write_like is True
    assert classification.source_write_like is True
    assert classification.source_write_paths == (
        "src/lib.rs",
        "pkg/server.go",
        "web/Button.tsx",
    )


def test_heredoc_redirect_to_non_python_source_is_write_like():
    cmd = "cat <<'EOF' > src/main.go\npackage main\nEOF"

    classification = classify_bash_write(cmd)

    assert classification.action_write_like is True
    assert classification.source_write_like is True
    assert classification.source_write_paths == ("src/main.go",)


def test_non_python_test_paths_are_not_source_write_paths():
    assert extract_source_write_paths(
        "sed -i 's/a/b/' tests/foo_test.go src/widget.test.ts __tests__/panel.tsx"
    ) == ()


def test_project_config_paths_are_not_source_write_paths():
    assert extract_source_write_paths(
        "sed -i 's/a/b/' Cargo.toml package.json go.mod CMakeLists.txt"
    ) == ()


def test_external_scratch_write_is_not_workspace_mutation():
    cmd = "cat > /tmp/new_methods.py"

    classification = classify_bash_write(cmd)

    assert classification.action_write_like is True
    assert classification.legacy_mutation_like is True
    assert classification.workspace_write_like is False
    assert classification.workspace_write_paths == ()
    assert classification.source_write_like is False
    assert classification.source_write_paths == ()
    assert is_bash_workspace_mutation_like(cmd) is False


def test_relative_write_is_workspace_and_source_mutation():
    cmd = "cat > new_methods.py"

    classification = classify_bash_write(cmd)

    assert classification.workspace_write_like is True
    assert classification.workspace_write_paths == ("new_methods.py",)
    assert classification.source_write_like is True
    assert classification.source_write_paths == ("new_methods.py",)
    assert is_bash_workspace_mutation_like(cmd) is True


def test_plain_tee_write_is_workspace_and_source_mutation():
    classification = classify_bash_write(
        "printf x | tee src/generated.py"
    )

    assert classification.workspace_write_like is True
    assert classification.workspace_write_paths == ("src/generated.py",)
    assert classification.source_write_like is True


def test_test_write_is_workspace_but_not_source_mutation():
    classification = classify_bash_write(
        "sed -i 's/a/b/' tests/test_app.py"
    )

    assert classification.workspace_write_like is True
    assert classification.workspace_write_paths == ("tests/test_app.py",)
    assert classification.source_write_like is False
    assert classification.source_write_paths == ()


def test_testbed_path_is_normalized_to_workspace_relative():
    classification = classify_bash_write(
        "sed -i 's/a/b/' /testbed/src/app.py"
    )

    assert classification.workspace_write_like is True
    assert classification.workspace_write_paths == ("src/app.py",)
    assert classification.source_write_paths == ("src/app.py",)


def test_legacy_mutation_policy_keeps_python_dash_heredoc_gate_shape():
    cmd = "python3 - <<'PY'\nprint('no write call here')\nPY"

    assert is_bash_action_write_like(cmd) is False
    assert is_bash_legacy_mutation_like(cmd) is True


def test_action_policy_sees_python_c_replace_but_legacy_policy_does_not():
    cmd = (
        "python3 -c \"import os; "
        "os.replace('src/app.py.tmp', 'src/app.py')\""
    )

    assert is_bash_action_write_like(cmd) is True
    assert is_bash_legacy_mutation_like(cmd) is False


def test_domain_write_method_is_not_file_write():
    cmd = (
        "cd /testbed && python -c \""
        "from astropy.table import QTable; "
        "import sys; "
        "tbl = QTable({'wave': [350]}); "
        "tbl.write(sys.stdout, format='ascii.rst')"
        "\""
    )

    classification = classify_bash_write(cmd)

    assert classification.action_write_like is False
    assert classification.source_write_like is False
    assert classification.source_write_paths == ()
