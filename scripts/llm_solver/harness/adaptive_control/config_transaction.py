"""Resolved-baseline composition for live adaptive config changes."""
from __future__ import annotations

import dataclasses


def resolved_baseline_cfg(session, fallback_cfg):
    baseline = getattr(session, "adaptive_control_resolved_baseline_cfg", None)
    if baseline is None:
        baseline = fallback_cfg
        session.adaptive_control_resolved_baseline_cfg = baseline
    return baseline


def apply_candidate_delta(resolved_baseline, raw_baseline, raw_candidate):
    """Apply only candidate-owned changes to the resolved launch config."""
    baseline_values = dataclasses.asdict(raw_baseline)
    candidate_values = dataclasses.asdict(raw_candidate)
    delta = {
        field: value
        for field, value in candidate_values.items()
        if baseline_values.get(field) != value
    }
    return dataclasses.replace(resolved_baseline, **delta)


def commit_config(session, cfg) -> tuple[str, ...]:
    session.cfg = cfg
    client = getattr(session, "client", None)
    if client is not None and hasattr(client, "cfg"):
        client.cfg = cfg
        return ("client.cfg",)
    return ()
