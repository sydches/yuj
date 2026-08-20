"""Self-contained import bootstrap for adaptive-control tests.

Import this module FIRST, before importing ``llm_solver.harness.adaptive_control``.

In a shell without the model-client deps (openai/orjson), it seeds stub parent
packages so the stdlib-only adaptive_control submodules import without running
``llm_solver.harness.__init__`` (which imports the live loop). When the deps are
present (CI / real harness), it does nothing and the real package is used, so it
never pollutes other harness tests.
"""
import sys
import types
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:  # real harness deps available -> use the real package, no stubbing
    import openai  # noqa: F401
    import orjson  # noqa: F401
    _DEPS = True
except Exception:  # noqa: BLE001
    _DEPS = False

if not _DEPS and "llm_solver.harness" not in sys.modules:
    _ll = sys.modules.setdefault("llm_solver", types.ModuleType("llm_solver"))
    if not hasattr(_ll, "__path__"):
        _ll.__path__ = [str(_SCRIPTS / "llm_solver")]
    _h = types.ModuleType("llm_solver.harness")
    _h.__path__ = [str(_SCRIPTS / "llm_solver" / "harness")]
    sys.modules["llm_solver.harness"] = _h
