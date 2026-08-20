"""Context strategies — swappable conversation layouts.

Each strategy decides which prior turns the model sees on the next call.
All implement the ``ContextManager`` protocol defined in
``harness/context.py``.

Discovery contract:
- Any sibling module can register a mode by exporting:
    CONTEXT_MODE = "<name>"
    CONTEXT_CLASS = <ContextManager subclass>
    CONTEXT_METADATA = ContextModeMetadata(...)

Adding a new strategy is file-drop only, as long as the module follows that
contract. The registry, CLI order, and artifact contract metadata all come
from the discovered mode record.
"""
from __future__ import annotations

import importlib
import pkgutil

from ._metadata import ContextMode, ContextModeMetadata


def _discover_context_modes() -> dict[str, ContextMode]:
    """Discover context mode records from strategy modules in this package."""
    discovered: dict[str, ContextMode] = {}
    prefix = __name__ + "."
    for mod_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if mod_info.name.startswith("_"):
            continue
        mod = importlib.import_module(prefix + mod_info.name)
        mode = getattr(mod, "CONTEXT_MODE", None)
        klass = getattr(mod, "CONTEXT_CLASS", None)
        if mode is None and klass is None:
            continue
        metadata = getattr(mod, "CONTEXT_METADATA", None)
        if not isinstance(mode, str) or not mode or klass is None:
            raise RuntimeError(
                f"Invalid context strategy registration in {mod.__name__}: "
                "expected CONTEXT_MODE and CONTEXT_CLASS"
            )
        if not isinstance(metadata, ContextModeMetadata):
            raise RuntimeError(
                f"Invalid context strategy registration in {mod.__name__}: "
                "expected CONTEXT_METADATA = ContextModeMetadata(...)"
            )
        if mode in discovered:
            raise RuntimeError(f"Duplicate context mode registration: {mode}")
        discovered[mode] = ContextMode(name=mode, cls=klass, metadata=metadata)
    return discovered


def _order_modes(discovered: dict[str, ContextMode]) -> tuple[ContextMode, ...]:
    """Return discovered modes in stable CLI order."""
    seen_orders: dict[int, str] = {}
    ordered = sorted(
        discovered.values(),
        key=lambda context_mode: (context_mode.metadata.cli_order, context_mode.name),
    )
    for context_mode in ordered:
        existing = seen_orders.get(context_mode.metadata.cli_order)
        if existing is not None:
            raise RuntimeError(
                "Duplicate context mode cli_order "
                f"{context_mode.metadata.cli_order}: {existing}, {context_mode.name}"
            )
        seen_orders[context_mode.metadata.cli_order] = context_mode.name
    return tuple(ordered)


_ORDERED_MODES = _order_modes(_discover_context_modes())
_MODE_BY_NAME = {context_mode.name: context_mode for context_mode in _ORDERED_MODES}
_MODE_BY_CLASS = {
    context_mode.cls: context_mode
    for context_mode in _ORDERED_MODES
}
_MODE_TO_CLASS = {
    context_mode.name: context_mode.cls
    for context_mode in _ORDERED_MODES
}

# Re-export commonly used classes for import stability.
FullTranscript = _MODE_TO_CLASS["full"]
CompactTranscript = _MODE_TO_CLASS["compact"]
ConciseTranscript = _MODE_TO_CLASS["concise"]
SlotTranscript = _MODE_TO_CLASS["slot"]
YujTranscript = _MODE_TO_CLASS["yuj"]
YconciseContext = _MODE_TO_CLASS["yconcise"]
YslotContext = _MODE_TO_CLASS["yslot"]
SolverStateContext = _MODE_TO_CLASS["stateful"]
CompoundContext = _MODE_TO_CLASS["compound"]
FocusedCompoundContext = _MODE_TO_CLASS["focused_compound"]
CompoundSelectiveContext = _MODE_TO_CLASS["compound_selective"]
SalienceContext = _MODE_TO_CLASS["salience"]
HalfLifeContext = _MODE_TO_CLASS["halflife"]


def list_context_modes() -> tuple[str, ...]:
    """Return CLI mode names in stable order."""
    return tuple(_MODE_TO_CLASS.keys())


def resolve_context_mode(mode: str) -> ContextMode:
    """Resolve a context mode name to its first-class registry record."""
    try:
        return _MODE_BY_NAME[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown context mode '{mode}'. Available: {list_context_modes()}"
        ) from exc


def resolve_context_mode_for_class(context_class: type) -> ContextMode | None:
    """Resolve a context manager class to its registry record, if registered."""
    return _MODE_BY_CLASS.get(context_class)


def resolve_context_class(mode: str):
    """Resolve a context mode name to its ContextManager class."""
    return resolve_context_mode(mode).cls


__all__ = [
    "ContextMode",
    "ContextModeMetadata",
    "list_context_modes",
    "resolve_context_mode",
    "resolve_context_mode_for_class",
    "resolve_context_class",
    "CompactTranscript",
    "ConciseTranscript",
    "CompoundContext",
    "CompoundSelectiveContext",
    "FocusedCompoundContext",
    "FullTranscript",
    "HalfLifeContext",
    "SalienceContext",
    "SolverStateContext",
    "SlotTranscript",
    "YconciseContext",
    "YslotContext",
    "YujTranscript",
]
