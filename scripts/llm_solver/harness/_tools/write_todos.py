"""Session-scoped todo-list validation for the ``write_todos`` tool.

The handler is deliberately side-effect free.  A successful result carries a
canonical copy of the submitted list back to the session loop, which owns the
durable ``todos`` trace event.  The state writer then projects that event into
``.solver/state.json``; this module never writes model-visible state directly.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


TODO_STATUSES = (
    "pending",
    "in_progress",
    "completed",
    "cancelled",
    "blocked",
)


class TodoValidationError(ValueError):
    """A submitted todo list does not satisfy the public tool contract."""


@dataclass(frozen=True)
class TodoWriteResult:
    """Model-facing success text paired with canonical trace payload data."""

    todos: tuple[dict[str, str], ...]

    def __str__(self) -> str:
        count = len(self.todos)
        noun = "item" if count == 1 else "items"
        return f"OK: replaced todo list with {count} {noun}"


def validate_todos(
    todos: object,
    *,
    max_items: int,
) -> tuple[dict[str, str], ...]:
    """Validate and copy one whole-list replacement payload.

    Validation runs in the handler as well as in the optional generic JSON
    schema layer.  That keeps the tool safe when
    ``tools.schema_validation = "off"`` and enforces the cross-item rule that
    at most one entry may be ``in_progress``.
    """
    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or max_items < 1
    ):
        raise TodoValidationError("tools.todos_max_items must be an integer >= 1")
    if not isinstance(todos, list):
        raise TodoValidationError("todos must be an array")
    if len(todos) > max_items:
        raise TodoValidationError(
            f"todos has {len(todos)} items; tools.todos_max_items is {max_items}"
        )

    canonical: list[dict[str, str]] = []
    in_progress = 0
    for index, item in enumerate(todos):
        if not isinstance(item, Mapping):
            raise TodoValidationError(f"todos[{index}] must be an object")
        if set(item) != {"description", "status"}:
            raise TodoValidationError(
                f"todos[{index}] must contain only description and status"
            )
        description = item.get("description")
        status = item.get("status")
        if not isinstance(description, str) or not description.strip():
            raise TodoValidationError(
                f"todos[{index}].description must be a non-empty string"
            )
        if not isinstance(status, str) or status not in TODO_STATUSES:
            allowed = "|".join(TODO_STATUSES)
            raise TodoValidationError(
                f"todos[{index}].status must be one of {allowed}"
            )
        if status == "in_progress":
            in_progress += 1
            if in_progress > 1:
                raise TodoValidationError(
                    "todos may contain at most one in_progress item"
                )
        canonical.append({"description": description, "status": status})
    return tuple(canonical)


def write_todos(todos: object, *, max_items: int) -> TodoWriteResult | str:
    """Return a canonical replacement payload or a repairable tool error."""
    try:
        canonical = validate_todos(todos, max_items=max_items)
    except TodoValidationError as exc:
        return f"ERROR: write_todos validation failed: {exc}"
    return TodoWriteResult(canonical)


__all__ = [
    "TODO_STATUSES",
    "TodoValidationError",
    "TodoWriteResult",
    "validate_todos",
    "write_todos",
]
