"""Append-only adaptive-control ledger writer.

One JSONL row per pause-point decision. The path is supplied by config; nothing
is written unless adaptive control is enabled.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .schema import ControlLedgerRow


def append_row(path: str, row: ControlLedgerRow) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dataclasses.asdict(row), sort_keys=True) + "\n")
