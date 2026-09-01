from __future__ import annotations

from scripts.llm_solver.harness.action_metadata import action_metadata


def test_python_read_heredoc_is_not_source_write_like():
    meta = action_metadata("bash", {
        "cmd": "python3 << 'PY'\nwith open('/testbed/src/app.py') as f:\n    print(f.read())\nPY"
    })

    assert meta["write_like"] is False
    assert meta["source_write_like"] is False
    assert meta["source_write_paths"] == []


def test_python_temp_replace_is_source_write_like():
    meta = action_metadata("bash", {
        "cmd": (
            "cd /testbed && python3 << 'PY'\n"
            "from pathlib import Path\n"
            "import os\n"
            "path = Path('src/app.py')\n"
            "tmp = path.with_name(path.name + '.tmp-write')\n"
            "tmp.write_text('x')\n"
            "os.replace(tmp, path)\n"
            "PY"
        )
    })

    assert meta["write_like"] is True
    assert meta["source_write_like"] is True
    assert meta["source_write_paths"] == ["src/app.py"]


def test_non_python_source_write_is_source_write_like():
    meta = action_metadata("bash", {
        "cmd": "cat <<'EOF' > src/main.rs\nfn main() {}\nEOF"
    })

    assert meta["write_like"] is True
    assert meta["source_write_like"] is True
    assert meta["source_write_paths"] == ["src/main.rs"]


def test_test_file_write_is_not_source_write_like():
    meta = action_metadata("bash", {
        "cmd": "cd /testbed && sed -i 's/a/b/' tests/test_app.py"
    })

    assert meta["write_like"] is True
    assert meta["source_write_like"] is False
    assert meta["source_write_paths"] == []


def test_external_scratch_write_is_not_source_write_like():
    meta = action_metadata("bash", {
        "cmd": "cat > /tmp/new_methods.py"
    })

    assert meta["write_like"] is True
    assert meta["source_write_like"] is False
    assert meta["source_write_paths"] == []
