"""Project-wide pytest fixtures.

Centralised so individual test modules don't have to repeat the same
setUp boilerplate. Anything autouse=True here applies to EVERY test
unless explicitly overridden.

Analysis starts without a task-format descriptor. Tests whose inputs require
one must select it explicitly; Python trace modules use python_analysis_format.
Context restoration isolates those choices without inventing startup facts.
Discovery tests must supply task inputs and let the harness obtain the answer.
Keep recorded benchmark regressions explicit; add ordinary task contrasts
instead of treating their passing results as general coverage.
"""
from __future__ import annotations

import os
import sys
import importlib.util
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


@pytest.fixture(scope="session", autouse=True)
def _isolated_yuj_auth_home(tmp_path_factory):
    """Keep every test process away from a user's real provider credentials."""
    prior = os.environ.get("YUJ_AUTH_HOME")
    os.environ["YUJ_AUTH_HOME"] = str(tmp_path_factory.mktemp("yuj-auth-home"))
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("YUJ_AUTH_HOME", None)
        else:
            os.environ["YUJ_AUTH_HOME"] = prior


@pytest.fixture(autouse=True)
def _isolated_task_formats():
    """Start unconfigured and restore both analysis import contexts afterward."""
    modules = []
    for mod_path in ("llm_solver.analysis._task_format",
                     "scripts.llm_solver.analysis._task_format"):
        # These are separate modules and ContextVars in this test suite.
        if importlib.util.find_spec(mod_path) is None:
            # The public snapshot omits offline analysis tools.
            continue
        modules.append(__import__(mod_path, fromlist=["_current"]))
    tokens = [(mod, mod._current.set(None)) for mod in modules]
    try:
        yield modules
    finally:
        for mod, token in reversed(tokens):
            mod._current.reset(token)


@pytest.fixture
def analysis_task_format(_isolated_task_formats):
    """Explicitly select a descriptor for a test's declared input format."""
    def select(name):
        for mod in _isolated_task_formats:
            mod.set_current(mod.load_task_format(name))
    return select


@pytest.fixture
def python_analysis_format(analysis_task_format):
    """Python-analysis cases opt in to the descriptor their traces require."""
    analysis_task_format("pytest")


@pytest.fixture
def bound_container_environment(monkeypatch):
    """Isolate environment-policy tests from Docker discovery and transport."""
    import importlib
    import os
    from scripts.llm_solver.harness import task_environment
    from scripts.llm_solver.harness.process_identity import ProcessIdentity
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')

    def discover(cwd, **kwargs):
        return task_environment.TaskEnvironment(
            str(cwd), '/selected/task', ('/selected/task',),
            container=os.environ['YUJ_CONTAINER'], container_id='a' * 64,
            process_identity=ProcessIdentity((1000,) * 4, (1000,) * 4, (1000,), ('0',) * 5),
        )

    monkeypatch.setattr(task_environment, 'discover_task_environment', discover)
    return execution


@pytest.fixture
def native_process_identity():
    """Observe the local entry process for real subprocess contract checks."""
    import subprocess
    from scripts.llm_solver.harness.process_identity import OBSERVE_SCRIPT, parse_process_identity
    result = subprocess.run(['bash', '--noprofile', '--norc', '-p', '-c', OBSERVE_SCRIPT],
                            capture_output=True, check=True)
    return parse_process_identity(result.stdout)
