"""Bookkeeping identity from declarations and selected native trace records.

Workspace names and contents are not task identifiers. These records describe
lineage asserted by a selected trace, not permission or revision equivalence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from uuid import UUID, uuid4


@dataclass(frozen=True)
class TaskIdentity:
    task_id: str
    invocation_id: str
    declared_instance_id: str = ""
    parent_invocation_id: str = ""
    linkage: str = "new"
    version: int = 1

    @property
    def instance_id(self) -> str:
        return self.declared_instance_id or f"local-task:{self.task_id}"

    def attempt_id(self, session_number: int) -> str:
        return f"invocation:{self.invocation_id}:session{int(session_number)}"

    def record(self) -> dict:
        return asdict(self)


def _read_record(raw) -> TaskIdentity:
    if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] != 1:
        raise ValueError("unsupported task identity record")
    try:
        if set(raw) != {field.name for field in fields(TaskIdentity)}:
            raise ValueError("incomplete identity record")
        identity = TaskIdentity(**raw)
        for value in (identity.task_id, identity.invocation_id):
            if not isinstance(value, str) or str(UUID(value)) != value:
                raise ValueError("noncanonical identity")
        if not isinstance(identity.parent_invocation_id, str):
            raise ValueError("invalid parent identity")
        if identity.parent_invocation_id:
            if str(UUID(identity.parent_invocation_id)) != identity.parent_invocation_id:
                raise ValueError("noncanonical parent identity")
        if not isinstance(identity.declared_instance_id, str):
            raise ValueError("invalid declared identity")
        if identity.linkage not in {"new", "selected_trace", "legacy_trace_unknown", "transcript_unknown"}:
            raise ValueError("invalid identity linkage")
        if (bool(identity.parent_invocation_id) != (identity.linkage == "selected_trace")
                or identity.parent_invocation_id == identity.invocation_id):
            raise ValueError("inconsistent parent identity")
        return identity
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid task identity record") from exc


def resolve_task_identity(
    cfg, *, prior_events: list[dict] | None = None, transcript_resume: bool = False,
) -> TaskIdentity:
    """Allocate once per invocation; only explicit trace resume carries lineage.

    A direct Session is a standalone invocation unless its caller supplies this
    shared object. An old trace or a wire transcript has no native identity;
    retain that uncertainty instead of searching adjacent files or task text.
    """
    declared = str(getattr(cfg, "adaptive_control_source_instance_id", "") or "")
    previous = None
    linkage = "transcript_unknown" if transcript_resume else "new"
    if prior_events is not None:
        latest = next((event for event in reversed(prior_events)
                       if event.get("event") == "session_start"), None)
        linkage = "legacy_trace_unknown"
        if latest is not None and "task_identity" in latest:
            previous = _read_record(latest["task_identity"])
            number = latest.get("session_number")
            if (type(number) is not int or number < 0
                    or latest.get("instance_id") != previous.instance_id
                    or latest.get("attempt_id") != previous.attempt_id(number)):
                raise ValueError("task identity disagrees with selected session record")
            linkage = "selected_trace"
    if previous is not None:
        if declared and declared != previous.declared_instance_id:
            raise ValueError("declared task identity conflicts with selected resume trace")
        declared = previous.declared_instance_id
    return TaskIdentity(
        task_id=previous.task_id if previous else str(uuid4()),
        invocation_id=str(uuid4()),
        declared_instance_id=declared,
        parent_invocation_id=previous.invocation_id if previous else "",
        linkage=linkage,
    )
