"""Shared context-strategy fixture builders.

Lifted out of the original test_harness_and_pipeline.py monolith
(commit 780bb2e split it into 15 sibling files). The helpers are used
across at least four test modules — duplicating the defaults dict in
each was causing drift, so they live here as the single source of truth.

Each `_make_<strategy>(**overrides)` factory wires the strategy class
with sensible test defaults; pass `original_prompt=` and any specific
tunable as overrides. Defaults match the production config.toml shape
closely enough that tests assert on real behaviour, not contrived edge
cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))


_COMPACT_DEFAULTS = dict(
    recent_results_chars=30000,
    trace_reasoning_chars=150,
    min_turns=2,
    args_summary_chars=80,
)


_SOLVER_STATE_DEFAULTS = dict(
    trace_lines=50,
    evidence_lines=30,
    inference_lines=20,
    recent_tool_results_chars=30000,
    trace_stub_chars=200,
    min_turns=2,
    suffix=(
        "Continue working. Your progress is tracked in .solver/state.json — "
        "read it to see what you've already done."
    ),
)


def _make_compact(**overrides):
    from llm_solver.harness.context_strategies import CompactTranscript
    kwargs = dict(_COMPACT_DEFAULTS)
    kwargs.update(overrides)
    return CompactTranscript(**kwargs)


def _make_solver_state(**overrides):
    from llm_solver.harness.context_strategies import SolverStateContext
    kwargs = dict(_SOLVER_STATE_DEFAULTS)
    kwargs.update(overrides)
    return SolverStateContext(**kwargs)


def _make_yuj_transcript(**overrides):
    from llm_solver.harness.context_strategies import YujTranscript
    kwargs = dict(_COMPACT_DEFAULTS)
    kwargs.update(overrides)
    return YujTranscript(**kwargs)


__all__ = [
    "_COMPACT_DEFAULTS",
    "_SOLVER_STATE_DEFAULTS",
    "_make_compact",
    "_make_solver_state",
    "_make_yuj_transcript",
]
