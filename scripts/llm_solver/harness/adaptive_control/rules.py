"""Detector rule-catalog loading and registry assembly.

The catalog says what rules exist and whether they are enabled for a detector
version. The implementation registry is deliberately small and explicit: config
can select known implementations, but cannot invent code.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DetectorFn = Callable[..., tuple[str, str, str]]


@dataclass(frozen=True)
class RuleSpec:
    catalog_version: str
    detector_version: str
    rule_id: str
    online_signal_id: str
    status: str
    enabled: str
    implementation: str
    input_contract_version: str
    config_key: str
    test_pool_version: str
    evidence_source: str
    notes: str

    @property
    def is_enabled(self) -> bool:
        return self.enabled.strip().lower() == "true"


def retired_noop(*_args, **_kwargs) -> tuple[str, str, str]:
    return "no_fire", "", ""


IMPLEMENTATIONS: dict[str, DetectorFn] = {
    "retired_noop": retired_noop,
}


def load_rule_catalog(path: str | Path) -> list[RuleSpec]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return [RuleSpec(**row) for row in csv.DictReader(fh, delimiter="\t")]


def build_registry(rows: list[RuleSpec], detector_version: str) -> dict[str, DetectorFn]:
    registry: dict[str, DetectorFn] = {}
    for row in rows:
        if row.detector_version != detector_version or not row.is_enabled:
            continue
        if row.status != "active":
            raise ValueError(f"enabled detector rule is not active: {row.rule_id}")
        fn = IMPLEMENTATIONS.get(row.implementation)
        if fn is None:
            raise ValueError(f"unknown detector implementation: {row.implementation}")
        if not row.online_signal_id:
            raise ValueError(f"enabled detector rule lacks online_signal_id: {row.rule_id}")
        registry[row.online_signal_id] = fn
    return registry
