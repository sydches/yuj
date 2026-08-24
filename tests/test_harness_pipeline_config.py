"""Tests for harness loop, tools, solver, generate pipeline, config, and end-to-end integration."""
import json
import os
import subprocess as _subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import openai
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.server.types import TurnResult, Usage, ToolCall
from llm_solver.config import Config, load_config, MODEL_MAP, _deep_merge, get_sdk_config


# ──────────────────────────────────────────────
# Helper: build a Config without loading TOML
# ──────────────────────────────────────────────

from _config_helpers import make_config  # centralized defaults — see tests/_config_helpers.py


def make_turn_result(content=None, tool_calls=None, finish_reason="stop", prompt_tokens=10):
    return TurnResult(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5),
    )

# ──────────────────────────────────────────────
# 1. Config loading
# ──────────────────────────────────────────────

class TestConfig:

    def test_load_config_returns_config(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.base_url  # should have a value

    def test_model_map_has_aliases(self):
        assert "haiku" in MODEL_MAP
        assert "sonnet" in MODEL_MAP
        assert "qwen3-vl" in MODEL_MAP

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        overlay = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(base, overlay)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_load_config_with_overrides(self):
        cfg = load_config(overrides={"model": "test-override"})
        assert cfg.model == "test-override"

    def test_rumination_gate_grace_prefix_loads_from_toml(self, tmp_path):
        cfg_path = tmp_path / "gate_grace.toml"
        cfg_path.write_text(
            '[prompts]\nrumination_gate_grace_prefix = "CUSTOM GATE GRACE"\n',
            encoding="utf-8",
        )
        cfg = load_config(user_config=cfg_path)
        assert cfg.rumination_gate_grace_prefix == "CUSTOM GATE GRACE"

    def test_pre_mutation_gate_loads_from_toml(self, tmp_path):
        cfg_path = tmp_path / "pre_mutation_gate.toml"
        cfg_path.write_text(
            '[prompts]\npre_mutation_gate = "CUSTOM PRE MUTATION {turn_number}"\n',
            encoding="utf-8",
        )
        cfg = load_config(user_config=cfg_path)
        assert cfg.pre_mutation_gate == "CUSTOM PRE MUTATION {turn_number}"

    def test_conditional_path_rule_defaults_and_overlay(self, tmp_path):
        default = load_config()
        assert default.injections_path_rules_enabled is False
        assert default.injections_path_rule_repeat is False

        cfg_path = tmp_path / "path-rules.toml"
        cfg_path.write_text(
            "[injections]\n"
            "enabled = true\n"
            "path_rules_enabled = true\n"
            "path_rule_repeat = true\n",
            encoding="utf-8",
        )
        cfg = load_config(user_config=cfg_path)
        assert cfg.injections_enabled is True
        assert cfg.injections_path_rules_enabled is True
        assert cfg.injections_path_rule_repeat is True

    def test_path_rules_require_injection_master_switch(self, tmp_path):
        cfg_path = tmp_path / "invalid-path-rules.toml"
        cfg_path.write_text(
            "[injections]\npath_rules_enabled = true\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="requires injections.enabled"):
            load_config(user_config=cfg_path)

    @pytest.mark.parametrize(
        "field",
        ["path_rules_enabled", "path_rule_repeat"],
    )
    def test_path_rule_knobs_require_booleans(self, tmp_path, field):
        cfg_path = tmp_path / f"invalid-{field}.toml"
        cfg_path.write_text(
            "[injections]\n"
            "enabled = true\n"
            f'{field} = "yes"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=field):
            load_config(user_config=cfg_path)

    def test_get_sdk_config(self):
        sdk = get_sdk_config()
        assert "tools" in sdk or "default_model" in sdk

    def test_config_frozen(self):
        cfg = make_config()
        with pytest.raises(AttributeError):
            cfg.model = "changed"


# ──────────────────────────────────────────────
# 3. Harness tools
# ──────────────────────────────────────────────
