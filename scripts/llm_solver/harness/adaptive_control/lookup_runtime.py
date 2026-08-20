"""Select an intervention from lookup TSV data.

Signal- and family-scoped selectors remain available. The ranked selector
skips interventions already tried in the current hurdle episode.
"""
from __future__ import annotations

import csv

from ...config import resolve_project_path

# Map lookup_status to a closed selection-block reason.
_STATUS_BLOCK = {
    "blocked_not_live_eligible": "medicine_not_live_eligible",
    "blocked_planned_executor": "executor_planned_not_implemented",
    "blocked_missing_executor": "missing_executor",
    "blocked_no_same_hurdle_clearance": "lookup_not_ready",
    "blocked_no_cleared_task": "lookup_not_ready",
    "blocked_preventive_start_only": "lookup_not_ready",
    "blocked_midrun": "lookup_not_ready",
    "blocked_executor": "lookup_not_ready",
    "blocked_live_disposition": "lookup_not_ready",
}

_BIG = 1 << 30


def load_lookup(path: str):
    """Read the lookup TSV, or None if no path / file is configured."""
    if not path:
        return None
    p = resolve_project_path(path)
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _rank(row: dict) -> int:
    try:
        ladder_rank = int(row.get("rank_within_ladder") or 0)
    except (TypeError, ValueError):
        ladder_rank = 0
    if ladder_rank > 0:
        return ladder_rank
    try:
        family_rank = int(row.get("rank_within_family") or 0)
    except (TypeError, ValueError):
        family_rank = 0
    if family_rank > 0:
        return family_rank
    try:
        r = int(row.get("rank_within_hurdle") or 0)
    except (TypeError, ValueError):
        return _BIG
    return r if r > 0 else _BIG


def _is_ready(row: dict) -> bool:
    return (
        row.get("lookup_status") == "ready_live_lookup"
        and row.get("reactive_lookup_allowed") == "true"
        and row.get("live_eligible") == "true"
        and row.get("ready_for_live", "true") != "false"
        and row.get("executor_status") == "implemented"
        and bool(row.get("runtime_executor_id") or "")
        and bool(row.get("candidate_config_path") or "")
    )


def _same_bundle_key(row: dict) -> str:
    return str(row.get("same_bundle_key", "") or "").strip()


def _excluded_bundle_keys(rows: list[dict], exclude: set[str]) -> set[str]:
    return {
        key
        for row in rows
        if row.get("intervention_id", "") in exclude
        for key in (_same_bundle_key(row),)
        if key
    }


def select(lookup_rows, online_signal_id: str):
    """Return (chosen_row | None, selection_status, blocked_reason).

    selection_status is 'selected' only when a ready_live_lookup row matched.
    """
    if lookup_rows is None:
        return None, "blocked", "missing_lookup_row"
    matches = [r for r in lookup_rows if r.get("online_signal_id") == online_signal_id and online_signal_id]
    if not matches:
        return None, "blocked", "missing_lookup_row"
    ready = [r for r in matches if _is_ready(r)]
    if ready:
        return min(ready, key=_rank), "selected", ""
    best = min(matches, key=_rank)
    return None, "blocked", _STATUS_BLOCK.get(best.get("lookup_status", ""), "lookup_not_ready")


def select_candidate(lookup_rows, online_signal_id: str, exclude_ids=()):
    """Ranked candidate selection for same-hurdle escalation.

    Like select(), but skips intervention_ids already applied in the current
    hurdle episode and reports the chosen candidate's rank plus the higher-ranked
    candidates it skipped (for the escalation ledger).

    Returns (chosen_row | None, selection_status, blocked_reason, rank, exclusion_refs).
    blocked_reason is `candidate_exhausted` when every ready candidate for the
    signal has already been applied in this episode.
    """
    exclude = set(exclude_ids or ())
    if lookup_rows is None:
        return None, "blocked", "missing_lookup_row", "", ""
    matches = [r for r in lookup_rows if r.get("online_signal_id") == online_signal_id and online_signal_id]
    if not matches:
        return None, "blocked", "missing_lookup_row", "", ""
    ready = sorted((r for r in matches if _is_ready(r)), key=_rank)
    if not ready:
        best = min(matches, key=_rank)
        return None, "blocked", _STATUS_BLOCK.get(best.get("lookup_status", ""), "lookup_not_ready"), "", ""
    excluded_bundles = _excluded_bundle_keys(ready, exclude)
    skipped: list[str] = []
    for r in ready:
        iid = r.get("intervention_id", "")
        if iid in exclude:
            skipped.append(f"{iid}@rank{r.get('rank_within_hurdle') or '?'}")
            continue
        bundle = _same_bundle_key(r)
        if bundle and bundle in excluded_bundles:
            skipped.append(f"{iid}@rank{r.get('rank_within_hurdle') or '?'}:same_bundle={bundle}")
            continue
        return r, "selected", "", str(r.get("rank_within_hurdle") or ""), ";".join(skipped)
    return None, "blocked", "candidate_exhausted", "", ";".join(skipped)


def select_by_family(lookup_rows, detector_family: str, exclude_ids=()):
    """Select a ready row from the detector-family lookup table.

    Return (chosen_row | None, selection_status, blocked_reason). Rows must
    carry `detector_family`.
    """
    if lookup_rows is None:
        return None, "blocked", "missing_lookup_row"
    family = (detector_family or "").strip()
    matches = [r for r in lookup_rows if r.get("detector_family") == family and family]
    if not matches:
        return None, "blocked", "missing_lookup_row"
    exclude = set(exclude_ids or ())
    ready = sorted((r for r in matches if _is_ready(r)), key=_rank)
    excluded_bundles = _excluded_bundle_keys(ready, exclude)
    for row in ready:
        if row.get("intervention_id", "") in exclude:
            continue
        bundle = _same_bundle_key(row)
        if bundle and bundle in excluded_bundles:
            continue
        return row, "selected", ""
    if ready:
        return None, "blocked", "candidate_exhausted"
    best = min(matches, key=_rank)
    return None, "blocked", _STATUS_BLOCK.get(
        best.get("lookup_status", ""),
        best.get("family_lookup_status", "") or "lookup_not_ready",
    )


def select_ranked_ladder(lookup_rows, exclude_ids=()):
    """Select the next ready intervention from one global ranked ladder.

    Detector family does not affect this choice. Skip interventions already
    applied in the current hurdle episode.
    """
    if lookup_rows is None:
        return None, "blocked", "missing_lookup_row"
    ready = sorted((row for row in lookup_rows if _is_ready(row)), key=_rank)
    if not ready:
        if not lookup_rows:
            return None, "blocked", "missing_lookup_row"
        best = min(lookup_rows, key=_rank)
        reason = _STATUS_BLOCK.get(
            best.get("lookup_status", ""),
            "lookup_not_ready",
        )
        return None, "blocked", reason

    exclude = set(exclude_ids or ())
    for row in ready:
        if row.get("intervention_id", "") not in exclude:
            return row, "selected", ""
    return None, "blocked", "candidate_exhausted"


def ranked_ladder_size(lookup_rows) -> int:
    """Count distinct ready interventions in the configured ladder."""
    if not lookup_rows:
        return 0
    return len({
        row.get("intervention_id", "")
        for row in lookup_rows
        if _is_ready(row) and row.get("intervention_id", "")
    })
