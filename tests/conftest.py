"""Project-wide pytest fixtures.

Centralised so individual test modules don't have to repeat the same
setUp boilerplate. Anything autouse=True here applies to EVERY test
unless explicitly overridden.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))


def pytest_configure(config):
    # Register custom marks used by the suite so pytest doesn't emit
    # PytestUnknownMarkWarning. test_smoke.py uses pytestmark =
    # pytest.mark.smoke as the module-level "smoke run" tag.
    config.addinivalue_line("markers", "smoke: smoke-test subset")


@pytest.fixture(autouse=True)
def _default_task_format():
    """Load `pytest` as the analysis task format for every test.

    The analysis layer's detector functions raise RuntimeError ("Task
    format not loaded. Analysis CLI must call set_current(...)") when
    they are invoked without a current task format. The CLI calls
    set_current() at startup; tests don't go through the CLI, so they
    each get hit by the same error. Loading pytest as the autouse
    default mirrors what every analysis CLI invocation does in production
    and unblocks ~70 tests across coherence / denorm_audit /
    denorm_discover that exercise these detectors.

    Tests that need a different format can call `set_current(load_task_format(...))`
    inside the test body to override; the ContextVar scope means the
    override applies only to that test.
    """
    # Import via BOTH paths the test suite uses ("llm_solver…" via the
    # tests/scripts sys.path entry AND "scripts.llm_solver…" via the
    # project-root entry). They resolve to DIFFERENT module objects with
    # DIFFERENT ContextVar instances; setting one doesn't affect the
    # other. Tests vary in which path they use, so set both.
    # Tests import the analysis layer via BOTH "llm_solver…" (via the
    # tests/scripts sys.path entry) and "scripts.llm_solver…" (via the
    # project-root entry). These resolve to DIFFERENT module objects
    # with DIFFERENT ContextVar instances; setting one doesn't affect
    # the other. Set both.
    setters = []
    for mod_path in ("llm_solver.analysis._task_format",
                     "scripts.llm_solver.analysis._task_format"):
        try:
            mod = __import__(mod_path, fromlist=["set_current", "load_task_format"])
            setters.append((mod.set_current, mod.load_task_format))
        except ImportError:
            continue
    for set_current, load_task_format in setters:
        set_current(load_task_format("pytest"))
    yield
