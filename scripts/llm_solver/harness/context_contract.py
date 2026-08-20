"""Serializable context contract metadata for run artifacts."""
from __future__ import annotations

from collections.abc import Mapping

from .context import ContextManager
from .context_strategies import (
    ContextMode,
    ContextModeMetadata,
    SolverStateContext,
    resolve_context_mode_for_class,
)
from .context_strategies._metadata import BASE_BUDGET_CONFIG_ATTRS


CONTEXT_CONTRACT_VERSION = 1


def _context_mode_record_for_class(
    context_class: type[ContextManager] | None,
) -> ContextMode | None:
    """Return the registry record for a context class, if one exists."""
    cls = context_class or SolverStateContext
    return resolve_context_mode_for_class(cls)


def context_mode_for_class(context_class: type[ContextManager] | None) -> str:
    """Return the registry mode for a context class, or a stable fallback."""
    cls = context_class or SolverStateContext
    context_mode = _context_mode_record_for_class(cls)
    if context_mode is not None:
        return context_mode.name
    return getattr(cls, "__name__", "unknown")


def _get(cfg, name: str):
    return getattr(cfg, name)


def _budget_snapshot(metadata: ContextModeMetadata | None, cfg) -> dict:
    attrs = (
        metadata.budget_config_attrs
        if metadata is not None
        else BASE_BUDGET_CONFIG_ATTRS
    )
    return {name: _get(cfg, name) for name in attrs}


def _suffix_present(metadata: ContextModeMetadata | None, cfg) -> bool:
    if metadata is None:
        return bool(_get(cfg, "state_context_suffix"))
    config_attr = metadata.constructor_config_attrs.get("suffix")
    return bool(config_attr and _get(cfg, config_attr))


def build_context_contract(
    context_class: type[ContextManager] | None,
    cfg,
) -> Mapping[str, object]:
    """Return the active model-facing context contract for artifacts."""
    cls = context_class or SolverStateContext
    context_mode = _context_mode_record_for_class(cls)
    metadata = context_mode.metadata if context_mode is not None else None
    mode = (
        context_mode.name
        if context_mode is not None
        else getattr(cls, "__name__", "unknown")
    )
    ignore_state = bool(_get(cfg, "context_ignore_state"))
    state_ignored_in_practice = (
        ignore_state
        and metadata is not None
        and metadata.state_ignored_when_context_ignore_state
    )
    return {
        "version": CONTEXT_CONTRACT_VERSION,
        "mode": mode,
        "class": f"{cls.__module__}.{cls.__name__}",
        "message_shape": (
            metadata.message_shape if metadata is not None else "strategy-defined"
        ),
        "state_source": (
            metadata.state_source if metadata is not None else "strategy-defined"
        ),
        "source_type": (
            metadata.source_type if metadata is not None else "strategy-defined"
        ),
        "normal_prompt_sources": (
            list(metadata.normal_prompt_sources)
            if metadata is not None
            else ["strategy-defined"]
        ),
        "section_order": (
            list(metadata.section_order)
            if metadata is not None
            else ["strategy_defined"]
        ),
        "section_labels": (
            dict(metadata.section_labels) if metadata is not None else {}
        ),
        "optional_sections_omitted_when_empty": True,
        "state_writer_enabled": bool(_get(cfg, "state_writer_enabled")),
        "context_ignore_state": ignore_state,
        "state_ignored_in_practice": state_ignored_in_practice,
        "file_freshness": (
            metadata.file_freshness if metadata is not None else "strategy-defined"
        ),
        "injection_support": (
            metadata.injection_support if metadata is not None else "strategy-defined"
        ),
        "suffix_present": _suffix_present(metadata, cfg),
        "budgets": _budget_snapshot(metadata, cfg),
    }
