"""Importable compaction-hook fixture used by runtime acceptance tests."""
from scripts.llm_solver.harness.compaction_hooks import Cancel, Compaction


seen = []

SUMMARY = """\
## Long-term goal
Finish the requested integration.
## Mid-term goal
Integrate context compaction.
## Near-term goal
Run the focused compaction tests.
## Constraints
Keep deterministic fallback behavior.
## Progress
Done: updated src/changed.py.
In progress: integration proof.
Blocked: none.
## Key decisions
Use the digest on validation failure because it is deterministic.
## Critical context
Modified path: src/changed.py
"""


def use_default(preparation):
    seen.append(preparation)
    return None


def cancel(preparation):
    seen.append(preparation)
    return Cancel()


def replace(preparation):
    seen.append(preparation)
    return Compaction(SUMMARY, preparation.first_kept_turn)


def replace_one_turn_earlier(preparation):
    seen.append(preparation)
    return Compaction(SUMMARY, preparation.first_kept_turn - 1)


def invalid_replace(preparation):
    seen.append(preparation)
    return Compaction(
        "## Long-term goal\nmissing", preparation.first_kept_turn
    )


def fail(preparation):
    seen.append(preparation)
    raise RuntimeError("stub hook failed")


async def async_hook(preparation):
    return None


not_callable = 42
