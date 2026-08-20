"""Harness layer — agentic loop, tool dispatch, task orchestration."""
from .context import ContextManager, FullTranscript
from .context_strategies import (
    CompactTranscript,
    CompoundContext,
    SolverStateContext,
    YujTranscript,
)
from .loop import (
    MODEL_STUCK,
    NORMAL_LIFECYCLE,
    Session,
    SessionResult,
    TaskSpec,
    build_resume_prompt,
    solve_task,
)
from .solver import build_system_prompt, collect_pending, write_checkpoint
