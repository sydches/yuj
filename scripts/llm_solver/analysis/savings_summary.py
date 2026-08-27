"""Aggregate legacy savings records and exact transformation records.

Ingests one or more savings JSONL files produced by
``harness/savings.py`` during a task run. Groups events by
(surface, bucket, layer, mechanism) for exact transformations. Older cost and
counterfactual savings records remain in a separate character-based table.

Sign convention on the input: delta_chars = output_chars - input_chars.
  negative → tokens saved (shrinking transform)
  positive → tokens paid (one-time cost)

The aggregator splits EXACT and ESTIMATE totals and never mixes them
in a headline figure. "net delta" reports the sum of exact records
only, with estimate records available for supporting context.

Usage:

    # One task.
    python3 -m scripts.llm_solver.analysis.savings_summary \\
        results/.../repos/<task>/.savings.jsonl

    # All tasks in a run dir (auto-discovers .savings.jsonl under repos/).
    python3 -m scripts.llm_solver.analysis.savings_summary \\
        results/.../

    # Multiple runs, JSONL tracker output for cross-campaign aggregation.
    python3 -m scripts.llm_solver.analysis.savings_summary \\
        results/.../ --tracker campaign_savings.jsonl

    # JSON to stdout for downstream tooling.
    python3 -m scripts.llm_solver.analysis.savings_summary --json <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _chain_breaks(records: list[dict]) -> list[dict]:
    """Return hash or step discontinuities within recorded text chains."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("event") != "transformation":
            continue
        chain_id = str(record.get("chain_id") or "")
        if not chain_id:
            continue
        key = (
            record.get("session", 0),
            record.get("turn", 0),
            str(record.get("tool_call_id") or ""),
            str(record.get("surface") or ""),
            chain_id,
        )
        grouped[key].append(record)

    breaks: list[dict] = []
    for key, chain in grouped.items():
        ordered = sorted(
            chain,
            key=lambda record: (
                int(record.get("chain_step", 0)),
                str(record.get("event_id") or ""),
            ),
        )
        steps = [int(record.get("chain_step", 0)) for record in ordered]
        expected = list(range(1, len(ordered) + 1))
        if steps != expected:
            breaks.append({"chain": key, "kind": "step", "steps": steps})
            continue
        for previous, current in zip(ordered, ordered[1:]):
            if previous.get("output_sha256") != current.get("input_sha256"):
                breaks.append({
                    "chain": key,
                    "kind": "hash",
                    "previous_event": previous.get("event_id"),
                    "current_event": current.get("event_id"),
                })
    return breaks


def _load_ledger(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _ledger_labels(path: Path, fallback_task: str) -> tuple[str, str]:
    records = _load_ledger(path)
    task_id = next(
        (str(row["task"]) for row in records if row.get("task")),
        fallback_task,
    )
    run_id = next(
        (str(row["run"]) for row in records if row.get("run")), "",
    )
    if not run_id and path.parent.name == "savings":
        run_id = path.parent.parent.name
    if (
        not run_id
        and path.name == ".savings.jsonl"
        and path.parent.parent.name == "repos"
    ):
        run_id = path.parents[2].name
    return run_id or path.parent.name, task_id


def _discover(paths: list[Path]) -> list[tuple[str, str, Path]]:
    """Resolve inputs to ``(run_id, task_id, ledger_path)`` rows."""
    pairs: list[tuple[str, str, Path]] = []

    def add(path: Path, fallback_task: str) -> None:
        run_id, task_id = _ledger_labels(path, fallback_task)
        pairs.append((run_id, task_id, path))

    for p in paths:
        if p.is_file() and p.name.endswith(".jsonl"):
            fallback = p.stem if p.stem not in {"savings", ".savings"} else p.parent.name
            add(p, fallback)
            continue
        if p.is_dir():
            # Task dir: has .savings.jsonl directly.
            direct = p / ".savings.jsonl"
            if direct.is_file():
                add(direct, p.name)
                continue
            # Current run layout: <run_dir>/savings/<task>.jsonl.
            savings = p / "savings"
            if savings.is_dir():
                for candidate in sorted(savings.glob("*.jsonl")):
                    add(candidate, candidate.stem)
                continue
            # Run dir: has repos/<task>/.savings.jsonl.
            repos = p / "repos"
            if repos.is_dir():
                for task_dir in sorted(repos.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    candidate = task_dir / ".savings.jsonl"
                    if candidate.is_file():
                        add(candidate, task_dir.name)
    return pairs


def _aggregate(records: list[dict]) -> dict:
    """Aggregate one task without mixing byte and legacy-char ledgers."""
    by_bucket: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "exact_delta": 0, "exact_count": 0,
        "estimate_delta": 0, "estimate_count": 0,
    })
    totals = {
        "exact_delta": 0, "exact_count": 0,
        "estimate_delta": 0, "estimate_count": 0,
        "costs_paid": 0, "savings": 0,
    }
    transform_by_bucket: dict[tuple[str, str, str, str], dict] = defaultdict(
        lambda: {
            "events": 0,
            "changes": 0,
            "input_bytes": 0,
            "output_bytes": 0,
            "delta_bytes": 0,
            "input_chars": 0,
            "output_chars": 0,
            "delta_chars": 0,
        }
    )
    transform_totals = {
        "events": 0,
        "changes": 0,
        "bytes_removed": 0,
        "bytes_added": 0,
        "delta_bytes": 0,
        "delta_chars": 0,
    }
    for r in records:
        if r.get("event") == "transformation":
            key = (
                str(r.get("surface", "?")),
                str(r.get("layer", "?")),
                str(r.get("bucket", "?")),
                str(r.get("mechanism", "?")),
            )
            row = transform_by_bucket[key]
            row["events"] += 1
            row["changes"] += int(r.get("change_count", 1))
            for field in (
                "input_bytes", "output_bytes", "delta_bytes",
                "input_chars", "output_chars", "delta_chars",
            ):
                row[field] += int(r.get(field, 0))
            delta_bytes = int(r.get("delta_bytes", 0))
            transform_totals["events"] += 1
            transform_totals["changes"] += int(r.get("change_count", 1))
            transform_totals["delta_bytes"] += delta_bytes
            transform_totals["delta_chars"] += int(r.get("delta_chars", 0))
            if delta_bytes < 0:
                transform_totals["bytes_removed"] += -delta_bytes
            else:
                transform_totals["bytes_added"] += delta_bytes
            continue
        key = (r.get("bucket", "?"), r.get("mechanism", "?"))
        measure = r.get("measure_type", "exact")
        delta = int(r.get("delta_chars", 0))
        if measure == "exact":
            by_bucket[key]["exact_delta"] += delta
            by_bucket[key]["exact_count"] += 1
            totals["exact_delta"] += delta
            totals["exact_count"] += 1
        else:
            by_bucket[key]["estimate_delta"] += delta
            by_bucket[key]["estimate_count"] += 1
            totals["estimate_delta"] += delta
            totals["estimate_count"] += 1
        if delta > 0:
            totals["costs_paid"] += delta
        else:
            totals["savings"] += -delta
    # Convert per-bucket dict to a sorted list of rows.
    rows = []
    for (bucket, mechanism), agg in sorted(by_bucket.items()):
        layer = next((r.get("layer", "") for r in records
                      if r.get("bucket") == bucket and r.get("mechanism") == mechanism), "")
        rows.append({
            "bucket": bucket,
            "mechanism": mechanism,
            "layer": layer,
            **agg,
        })
    transform_rows = [
        {
            "surface": surface,
            "layer": layer,
            "bucket": bucket,
            "mechanism": mechanism,
            **aggregate,
        }
        for (surface, layer, bucket, mechanism), aggregate
        in sorted(transform_by_bucket.items())
    ]
    return {
        "totals": totals,
        "rows": rows,
        "transformations": {
            "totals": transform_totals,
            "rows": transform_rows,
            "chain_breaks": _chain_breaks(records),
        },
    }


def _fmt_chars(n: int) -> str:
    sign = "+" if n > 0 else ""
    return f"{sign}{n:,}"


def _fmt_tokens(n: int) -> str:
    t = n // 4
    sign = "+" if t > 0 else ""
    return f"{sign}{t:,}"


def _fmt_bytes(n: int) -> str:
    sign = "+" if n > 0 else ""
    return f"{sign}{n:,}"


def format_markdown(per_task: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# Savings ledger summary\n")
    lines.append(f"Task ledgers analyzed: {len(per_task)}\n")

    transform_totals = {
        "events": sum(
            agg["transformations"]["totals"]["events"]
            for agg in per_task.values()
        ),
        "changes": sum(
            agg["transformations"]["totals"]["changes"]
            for agg in per_task.values()
        ),
        "bytes_removed": sum(
            agg["transformations"]["totals"]["bytes_removed"]
            for agg in per_task.values()
        ),
        "bytes_added": sum(
            agg["transformations"]["totals"]["bytes_added"]
            for agg in per_task.values()
        ),
        "delta_bytes": sum(
            agg["transformations"]["totals"]["delta_bytes"]
            for agg in per_task.values()
        ),
        "chain_breaks": sum(
            len(agg["transformations"]["chain_breaks"])
            for agg in per_task.values()
        ),
    }
    lines.append("## Exact transformations\n")
    lines.append(
        f"Events: {transform_totals['events']:,}; changed regions: "
        f"{transform_totals['changes']:,}; bytes removed: "
        f"{transform_totals['bytes_removed']:,}; bytes added: "
        f"{transform_totals['bytes_added']:,}; arithmetic net: "
        f"{_fmt_bytes(transform_totals['delta_bytes'])} bytes; chain breaks: "
        f"{transform_totals['chain_breaks']:,}."
    )
    lines.append(
        "The net is the sum of observed before/after deltas. It is not an "
        "estimate of later model behavior.\n"
    )

    transform_rows: dict[tuple[str, str, str, str], dict] = defaultdict(
        lambda: {"events": 0, "changes": 0, "delta_bytes": 0,
                 "delta_chars": 0}
    )
    for task_agg in per_task.values():
        for row in task_agg["transformations"]["rows"]:
            key = (
                row["surface"], row["layer"], row["bucket"], row["mechanism"]
            )
            for field in ("events", "changes", "delta_bytes", "delta_chars"):
                transform_rows[key][field] += row[field]
    lines.append(
        "| Surface | Layer | Bucket | Mechanism | Events | Changes | "
        "Delta bytes | Delta chars |"
    )
    lines.append(
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"
    )
    for (surface, layer, bucket, mechanism), row in sorted(transform_rows.items()):
        lines.append(
            f"| {surface} | {layer} | {bucket} | {mechanism} | "
            f"{row['events']} | {row['changes']} | "
            f"{_fmt_bytes(row['delta_bytes'])} | "
            f"{_fmt_chars(row['delta_chars'])} |"
        )
    lines.append("")

    # Legacy campaign totals stay character-based and separate.
    campaign_exact_delta = sum(agg["totals"]["exact_delta"] for agg in per_task.values())
    campaign_estimate_delta = sum(agg["totals"]["estimate_delta"] for agg in per_task.values())
    campaign_costs = sum(agg["totals"]["costs_paid"] for agg in per_task.values())
    campaign_savings = sum(agg["totals"]["savings"] for agg in per_task.values())

    lines.append("## Legacy cost and counterfactual records\n")
    lines.append(f"- Costs paid:      {_fmt_chars(campaign_costs)} chars = {_fmt_tokens(campaign_costs)} tokens")
    lines.append(f"- Savings:         {_fmt_chars(-campaign_savings)} chars = {_fmt_tokens(-campaign_savings)} tokens")
    lines.append(f"- **Net exact**:   {_fmt_chars(campaign_exact_delta)} chars = {_fmt_tokens(campaign_exact_delta)} tokens")
    if campaign_estimate_delta:
        lines.append(f"\nEstimated (not mixed into net): {_fmt_chars(campaign_estimate_delta)} chars = {_fmt_tokens(campaign_estimate_delta)} tokens")
    lines.append("")

    # Cross-task per-bucket aggregate.
    bucket_totals: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
        "exact_delta": 0, "exact_count": 0,
        "estimate_delta": 0, "estimate_count": 0,
    })
    for task_agg in per_task.values():
        for row in task_agg["rows"]:
            key = (row["layer"], row["bucket"], row["mechanism"])
            bucket_totals[key]["exact_delta"] += row["exact_delta"]
            bucket_totals[key]["exact_count"] += row["exact_count"]
            bucket_totals[key]["estimate_delta"] += row["estimate_delta"]
            bucket_totals[key]["estimate_count"] += row["estimate_count"]

    lines.append("## Per-bucket breakdown (across all tasks)\n")
    lines.append("| Layer | Bucket | Mechanism | Events | Δchars | Δtokens | Type |")
    lines.append("|-------|--------|-----------|--------|--------|---------|------|")
    for (layer, bucket, mechanism), agg in sorted(bucket_totals.items()):
        if agg["exact_count"]:
            lines.append(
                f"| {layer} | {bucket} | {mechanism} | {agg['exact_count']} | "
                f"{_fmt_chars(agg['exact_delta'])} | {_fmt_tokens(agg['exact_delta'])} | exact |"
            )
        if agg["estimate_count"]:
            lines.append(
                f"| {layer} | {bucket} | {mechanism} | {agg['estimate_count']} | "
                f"{_fmt_chars(agg['estimate_delta'])} | {_fmt_tokens(agg['estimate_delta'])} | estimate |"
            )
    lines.append("")

    # Per-run, per-task totals.
    lines.append("## Per-run, per-task totals\n")
    lines.append(
        "| Run | Task | Transform events | Transform delta bytes | Legacy events | "
        "Legacy net tokens |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for source_id, agg in sorted(per_task.items()):
        t = agg["totals"]
        tx = agg["transformations"]["totals"]
        source = agg.get("source", {})
        lines.append(
            f"| {source.get('run', '')} | {source.get('task', source_id)} | "
            f"{tx['events']} | {_fmt_bytes(tx['delta_bytes'])} | "
            f"{t['exact_count'] + t['estimate_count']} | "
            f"{_fmt_tokens(t['exact_delta'])} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate harness/.savings.jsonl records into a report.",
    )
    parser.add_argument("paths", nargs="+", type=Path,
                        help="Run dirs, task dirs, or .savings.jsonl paths.")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON payload instead of markdown.")
    parser.add_argument("--tracker", type=Path, default=None,
                        help="Append per-task aggregated records to this JSONL "
                             "for cross-campaign aggregation.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write markdown to this path (default: stdout).")
    args = parser.parse_args()

    pairs = _discover(args.paths)
    if not pairs:
        print("No .savings.jsonl files found under provided paths.", file=sys.stderr)
        sys.exit(1)

    per_task: dict[str, dict] = {}
    for run_id, task_id, ledger_path in pairs:
        records = _load_ledger(ledger_path)
        source_id = f"{run_id}/{task_id}"
        if source_id in per_task:
            source_id = f"{source_id}@{ledger_path}"
        per_task[source_id] = _aggregate(records)
        per_task[source_id]["source"] = {
            "run": run_id,
            "task": task_id,
            "ledger": str(ledger_path),
        }

    if args.tracker:
        args.tracker.parent.mkdir(parents=True, exist_ok=True)
        with open(args.tracker, "a") as f:
            for source_id, agg in per_task.items():
                f.write(json.dumps({
                    "event": "savings_summary",
                    "source_id": source_id,
                    **agg,
                }, default=str) + "\n")

    if args.json:
        print(json.dumps({"per_task": per_task}, indent=2, default=str))
        return

    md = format_markdown(per_task)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
    else:
        print(md)


if __name__ == "__main__":
    main()
