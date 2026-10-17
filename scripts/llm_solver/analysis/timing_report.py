"""Rank measured runtime costs from one or more harness traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def trace_paths(inputs):
    paths = set()
    for value in inputs:
        path = Path(value)
        if path.is_file():
            paths.add(path.resolve())
        elif path.is_dir():
            paths.update(p.resolve() for p in path.rglob("*.trace.jsonl"))
    return sorted(paths)


def collect(paths, *, limit=20):
    totals = {key: 0.0 for key in ("chat_call_ms", "tool_ms", "harness_ms")}
    operations = {}
    slow_turns, slow_operations = [], []
    turns = splits = detailed = invalid = 0
    for path in paths:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    invalid += 1
                    continue
                if not isinstance(row, dict) or row.get("event") != "turn_timing":
                    continue
                turns += 1
                identity = {"trace": str(path), "session": row.get("session_number"),
                            "turn": row.get("turn_number")}
                if all(isinstance(row.get(key), (int, float)) for key in totals):
                    splits += 1
                    for key in totals:
                        totals[key] += row[key]
                    slow_turns.append({**identity, **{key: row[key] for key in totals}})
                    slow_turns.sort(key=lambda r: r["harness_ms"], reverse=True)
                    del slow_turns[limit:]
                if row.get("operation_timing_version") != 1:
                    continue
                detailed += 1
                for item in row["operations"]:
                    key = tuple(item["operation"])
                    group = operations.setdefault(key, {
                        "operation": list(key), "calls": 0, "elapsed_ms": 0.0,
                        "self_ms": 0.0, "max_ms": 0.0, "raised": 0,
                    })
                    for field in ("calls", "elapsed_ms", "self_ms", "raised"):
                        group[field] += item[field]
                    group["max_ms"] = max(group["max_ms"], item["max_ms"])
                    slow_operations.append({**identity, **item})
                    slow_operations.sort(key=lambda r: r["self_ms"], reverse=True)
                    del slow_operations[limit:]
    return {
        "traces": len(paths), "timed_turns": turns, "split_turns": splits,
        "detailed_turns": detailed, "invalid_json_lines": invalid,
        "wall_ms": {key: round(value, 3) for key, value in totals.items()},
        "operations": sorted(operations.values(), key=lambda r: r["self_ms"], reverse=True),
        "slow_turns": slow_turns, "slow_operations": slow_operations,
    }


def render(report, *, limit=20):
    lines = [f"Traces: {report['traces']}; timing rows: {report['timed_turns']}; "
             f"operation breakdowns: {report['detailed_turns']}"]
    if report["invalid_json_lines"]:
        lines.append(f"Unreadable JSON rows: {report['invalid_json_lines']} (excluded)")
    if report["split_turns"]:
        totals = report["wall_ms"]
        lines.append(f"Measured wall time across {report['split_turns']} turns: "
                     f"model {totals['chat_call_ms'] / 1000:.3f}s; "
                     f"tool dispatch {totals['tool_ms'] / 1000:.3f}s; "
                     f"harness {totals['harness_ms'] / 1000:.3f}s")
    else:
        lines.append("Model/tool/harness split: unknown")
    if report["detailed_turns"] < report["timed_turns"]:
        lines.append("Turns without an operation breakdown remain unknown.")
    lines += ["", "Operations ranked by own time (seconds):",
              "     own  inclusive      max    calls  raised  operation"]
    for row in report["operations"][:limit]:
        lines.append(f"{row['self_ms'] / 1000:8.3f} {row['elapsed_ms'] / 1000:10.3f} "
                     f"{row['max_ms'] / 1000:8.3f} {row['calls']:8d} {row['raised']:7d}  "
                     + " > ".join(row["operation"]))
    if not report["operations"]:
        lines.append("No operation measurements recorded.")
    lines.append("Own time excludes measured children. Worker times can overlap; "
                 "do not add these to the wall-time split.")
    lines += ["", "Largest operation costs by turn and call:"]
    for row in report["slow_operations"][:limit]:
        lines.append(f"  {row['self_ms'] / 1000:.3f}s own, {row['calls']} calls: "
                     f"{row['trace']} session={row['session']} turn={row['turn']} "
                     f"call={row['tool_call_id'] or '(loop)'}: "
                     + " > ".join(row["operation"]))
    lines += ["", "Slowest turns by harness time:"]
    for row in report["slow_turns"][:limit]:
        lines.append(f"  harness={row['harness_ms'] / 1000:.3f}s "
                     f"tool={row['tool_ms'] / 1000:.3f}s model={row['chat_call_ms'] / 1000:.3f}s "
                     f"{row['trace']} session={row['session']} turn={row['turn']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="trace files or run directories")
    parser.add_argument("--limit", type=int, default=20, help="rows per ranking (default: 20)")
    parser.add_argument("--json", action="store_true", help="print structured results")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    paths = trace_paths(args.paths)
    if not paths:
        parser.error("no trace files found")
    report = collect(paths, limit=args.limit)
    print(json.dumps(report, indent=2) if args.json else render(report, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
