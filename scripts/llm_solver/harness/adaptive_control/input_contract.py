"""Detector input-contract loading.

The contract is data, not detector logic. It records which artifact fields are
available live, which are controller outputs, and which are post-run/future
fields that a live detector must not read.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputField:
    contract_version: str
    artifact: str
    field: str
    availability: str
    created_by: str
    content_origin: str
    detector_input: str
    use_scope: str
    example: str
    notes: str

    @property
    def is_live_input(self) -> bool:
        return self.availability in {"live_prefix", "run_start"} and self.detector_input == "yes"


def load_input_contract(path: str | Path) -> list[InputField]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return [InputField(**row) for row in csv.DictReader(fh, delimiter="\t")]


def live_input_fields(rows: list[InputField]) -> set[tuple[str, str]]:
    return {(row.artifact, row.field) for row in rows if row.is_live_input}
