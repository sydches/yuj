"""Tests for absolute paths that are already inside the working directory."""
from __future__ import annotations

import os
import sys
import tempfile
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.llm_solver.harness._tools._common import _resolve


def test_abs_path_already_inside_cwd_resolves_to_itself():
    """The bug case: abs path under cwd must resolve correctly."""
    with tempfile.TemporaryDirectory() as cwd:
        target = os.path.join(cwd, "foo.py")
        Path(target).write_text("# hello\n")

        # An absolute path inside cwd resolves to itself.
        resolved = _resolve(cwd, target)
        assert resolved == Path(target).resolve()
        assert resolved.read_text() == "# hello\n"


def test_abs_path_under_cwd_nested_subdir():
    """Nested abs path under cwd resolves correctly."""
    with tempfile.TemporaryDirectory() as cwd:
        sub = os.path.join(cwd, "astropy", "coordinates", "builtin_frames")
        os.makedirs(sub, exist_ok=True)
        target = os.path.join(sub, "itrs.py")
        Path(target).write_text("class ITRS:\n    pass\n")

        # A nested path also stays inside cwd.
        resolved = _resolve(cwd, target)
        assert resolved.exists()
        assert "class ITRS" in resolved.read_text()


def test_abs_path_outside_cwd_is_refused_even_with_a_local_collision():
    with tempfile.TemporaryDirectory() as cwd:
        collision = Path(cwd) / 'etc/passwd'
        collision.parent.mkdir()
        collision.write_text('different task file')
        with pytest.raises(ValueError, match='escapes cwd'):
            _resolve(cwd, '/etc/passwd')
        assert collision.read_text() == 'different task file'


def test_dotdot_escape_still_refused():
    """Path containment is preserved: `..` escapes raise ValueError."""
    import pytest as _pytest
    with tempfile.TemporaryDirectory() as cwd:
        with _pytest.raises(ValueError, match="escapes cwd"):
            _resolve(cwd, "../../../etc/passwd")


@pytest.mark.parametrize("tool,args", [
    ("read", {"path": "/outside/file.txt"}),
    ("write", {"path": "/outside/file.txt", "content": "REPLACED"}),
    ("edit", {"path": "/outside/file.txt", "old_str": "TASK COLLISION", "new_str": "REPLACED"}),
    ("apply_patch", {"patch": "*** Begin Patch\n*** Update File: /outside/file.txt\n@@\n-TASK COLLISION\n+REPLACED\n*** End Patch"}),
])
def test_dispatch_refuses_outside_name_without_touching_collision(tmp_path, tool, args):
    from _config_helpers import make_config
    from scripts.llm_solver.harness.tools import dispatch

    collision = tmp_path / "outside/file.txt"
    collision.parent.mkdir()
    collision.write_text("TASK COLLISION\n")
    result = dispatch(tool, args, cwd=str(tmp_path), cfg=make_config(sandbox_bash=False))
    assert "ERROR" in result
    assert "TASK COLLISION" not in result
    assert collision.read_text() == "TASK COLLISION\n"


def test_relative_path_unchanged():
    """Existing relative-path behavior preserved."""
    with tempfile.TemporaryDirectory() as cwd:
        target = os.path.join(cwd, "bar.py")
        Path(target).write_text("# rel\n")

        resolved = _resolve(cwd, "bar.py")
        assert resolved.read_text() == "# rel\n"


def test_symlink_under_cwd_resolves():
    """Symlinks under cwd that point inside cwd are handled."""
    with tempfile.TemporaryDirectory() as cwd:
        real = os.path.join(cwd, "real.py")
        Path(real).write_text("# real\n")
        link = os.path.join(cwd, "link.py")
        os.symlink(real, link)

        # Absolute path of the symlink, which is inside cwd
        resolved = _resolve(cwd, link)
        assert resolved.read_text() == "# real\n"


def test_cwd_with_trailing_slash():
    """cwd argument robust to trailing slash."""
    with tempfile.TemporaryDirectory() as cwd:
        target = os.path.join(cwd, "foo.py")
        Path(target).write_text("ok")

        resolved = _resolve(cwd + "/", target)
        assert resolved.read_text() == "ok"
