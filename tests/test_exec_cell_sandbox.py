"""Live sandbox escape and timeout checks for issue #7 exec_cell."""
from __future__ import annotations

from pathlib import Path
import time

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness.sandbox._preflight import bwrap_preflight
from scripts.llm_solver.harness.tools import dispatch


BWRAP = Path("/usr/bin/bwrap")
BWRAP_AVAILABLE, BWRAP_FAILURE = bwrap_preflight(str(BWRAP))
requires_bwrap = pytest.mark.skipif(
    not BWRAP_AVAILABLE,
    reason=f"operational bwrap is required: {BWRAP_FAILURE or 'unavailable'}",
)


def _cfg(**overrides):
    values = dict(
        tools_exec_cell_enabled=True,
        tools_exec_cell_timeout=5,
        sandbox_bash=True,
        sandbox_required=True,
        tools_unified_envelope_enabled=False,
    )
    values.update(overrides)
    return make_config(**values)


@pytest.mark.parametrize(
    "source,target_kind",
    [
        (
            "from pathlib import Path\n"
            "p = Path.home() / '.yuj_exec_cell_escape'\n"
            "try:\n p.write_text('escape')\n"
            "except Exception as exc:\n print(type(exc).__name__, exc)",
            "home",
        ),
        (
            "from pathlib import Path\n"
            "p = Path('../../../../../yuj_exec_cell_relative_escape')\n"
            "try:\n p.write_text('escape')\n"
            "except Exception as exc:\n print(type(exc).__name__, exc)",
            "relative",
        ),
        (
            "from pathlib import Path\n"
            "p = Path('/usr/local/lib/yuj_exec_cell_escape')\n"
            "try:\n p.write_text('escape')\n"
            "except Exception as exc:\n print(type(exc).__name__, exc)",
            "absolute",
        ),
    ],
)
@requires_bwrap
def test_exec_cell_blocks_docs_sandbox_escape_attempts(
    tmp_path: Path, source: str, target_kind: str,
):
    targets = {
        "home": Path.home() / ".yuj_exec_cell_escape",
        "relative": (tmp_path / "../../../../../yuj_exec_cell_relative_escape").resolve(),
        "absolute": Path("/usr/local/lib/yuj_exec_cell_escape"),
    }
    target = targets[target_kind]
    if target.exists():
        target.unlink()
    result = dispatch(
        "exec_cell", {"source": source}, cwd=str(tmp_path), cfg=_cfg()
    )
    assert "read-only" in result.lower() or "Read-only" in result
    assert not target.exists()


@requires_bwrap
def test_exec_cell_allows_task_directory_write(tmp_path: Path):
    result = dispatch(
        "exec_cell",
        {
            "source": (
                "from pathlib import Path\n"
                "Path('inside.txt').write_text('inside')\n"
                "print(Path('inside.txt').read_text())"
            )
        },
        cwd=str(tmp_path),
        cfg=_cfg(),
    )
    assert "inside" in result
    assert (tmp_path / "inside.txt").read_text() == "inside"


@requires_bwrap
def test_exec_cell_timeout_bounds_model_code(tmp_path: Path):
    started = time.monotonic()
    result = dispatch(
        "exec_cell",
        {"source": "while True:\n    pass"},
        cwd=str(tmp_path),
        cfg=_cfg(tools_exec_cell_timeout=1),
    )
    elapsed = time.monotonic() - started
    assert "timed out after 1 seconds" in result
    assert elapsed < 5


@requires_bwrap
def test_exec_cell_timeout_bounds_inner_calls(tmp_path: Path):
    started = time.monotonic()
    result = dispatch(
        "exec_cell",
        {"source": 'print(bash("sleep 10"))'},
        cwd=str(tmp_path),
        cfg=_cfg(tools_exec_cell_timeout=1),
    )
    elapsed = time.monotonic() - started
    assert "timed out" in result.lower()
    assert elapsed < 5


def test_exec_cell_uses_explicit_unsandboxed_execution(tmp_path: Path):
    result = dispatch(
        "exec_cell",
        {"source": "print('explicit host execution')"},
        cwd=str(tmp_path),
        cfg=_cfg(
            sandbox_backend="none",
            sandbox_bash=False,
            sandbox_required=False,
        ),
    )
    assert result.strip() == "explicit host execution"
