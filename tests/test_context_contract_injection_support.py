"""Per-strategy injection_support flag in the context_contract.

Injections are appended via ctx.add_user(<fragment>); only the `full` strategy delivers them as
a discrete user-role message. Every other strategy folds the
fragment into the next projected user message — the envelope wrap
survives but positional contract is lost. Surfaced in the contract
so operator overlays setting injections_enabled=true under a
projection-shaping strategy don't silently get a different shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pytest

from _config_helpers import make_config
from llm_solver.harness.context_contract import build_context_contract
from llm_solver.harness.context_strategies import (
    list_context_modes,
    resolve_context_class,
)


def test_full_supports_verbatim_injection():
    cls = resolve_context_class("full")
    c = build_context_contract(cls, make_config())
    assert c["injection_support"] == "verbatim"


def test_halflife_supports_verbatim_injection():
    cls = resolve_context_class("halflife")
    c = build_context_contract(cls, make_config())
    assert c["injection_support"] == "verbatim"


@pytest.mark.parametrize("mode", [
    m for m in (
        "compact", "concise", "slot", "yuj", "yconcise", "yslot",
        "stateful", "compound", "focused_compound", "compound_selective", "salience",
    )
])
def test_projection_modes_bury_injection(mode):
    cls = resolve_context_class(mode)
    c = build_context_contract(cls, make_config())
    assert c["injection_support"] == "buried_in_projection"


def test_every_registered_mode_has_injection_support():
    for mode in list_context_modes():
        cls = resolve_context_class(mode)
        c = build_context_contract(cls, make_config())
        assert c["injection_support"] != "strategy-defined", mode
