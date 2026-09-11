"""Keep declared context permission separate from model capacity evidence."""
from dataclasses import replace


def capacity(value):
    return value if type(value) is int and value > 0 else None


def declared_context(cfg):
    record = getattr(cfg, "context_allocation", None) or {}
    return record.get("declared_context", cfg.context_size)


def allocate_context(cfg, *, observed=None, profile=None, declared=None,
                     source=None):
    """Apply one model's constraints without consuming the original allowance.

    Callers supply fresh capacity evidence on a model/endpoint change. Missing
    evidence stays unknown; neither a profile loader default nor spare backend
    capacity grants permission. Existing output policy remains a separate cap.
    """
    previous = getattr(cfg, "context_allocation", None) or {}
    declared = declared_context(cfg) if declared is None else declared
    if capacity(declared) is None:
        raise ValueError("declared context allowance must be a positive integer")
    observed = capacity(observed)
    profile_limit = capacity(getattr(profile, "context_capacity", None))
    effective = min(value for value in (declared, observed, profile_limit)
                    if value is not None)
    output_limit = previous.get("declared_output_tokens")
    if output_limit is None:
        output_limit = capacity(getattr(cfg, "max_tokens", None))
    if output_limit is None:
        output_limit = int(declared * cfg.max_tokens_fraction)
    prompt_limit = int(effective * cfg.context_fill_ratio)
    record = {
        "declared_context": declared,
        "declaration_source": source or previous.get("declaration_source", "caller"),
        "observed_capacity": observed,
        "profile_capacity": profile_limit,
        "profile_source": getattr(profile, "context_capacity_source", None) if profile_limit else None,
        "model": cfg.model, "base_url": cfg.base_url,
        "effective_context": effective,
        "prompt_limit": prompt_limit,
        "declared_output_tokens": output_limit,
        "output_character_limits": {
            "max_output_chars": getattr(cfg, "max_output_chars", None),
            "recent_tool_results_chars": getattr(cfg, "recent_tool_results_chars", None),
            "basis": "resolved_output_policy",
        },
    }
    updates = dict(
        context_size=effective, context_allocation=record,
        max_tokens=min(output_limit, int(effective * cfg.max_tokens_fraction)),
    )
    try:
        return replace(cfg, **updates)
    except TypeError:
        from copy import copy
        result = copy(cfg)
        for name, value in updates.items():
            setattr(result, name, value)
        return result
