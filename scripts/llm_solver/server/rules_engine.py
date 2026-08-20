"""Rules engine — parse rules.toml and apply declarative transforms.

DSL ceiling: read-only cross-field guards, no state across turns,
no inter-rule dependencies, no computation beyond regex.
"""
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._shared.toml_compat import tomllib

log = logging.getLogger(__name__)

# Type alias for a single rule transform
RuleFn = Callable[[dict], dict]


def _check_guard(guard: dict[str, list[str]], response: dict) -> bool:
    """Check if all guard conditions are met (read-only)."""
    for field, allowed in guard.items():
        if response.get(field) not in allowed:
            return False
    return True


def _make_strip(rule: dict) -> RuleFn:
    """Build a strip rule: regex replace on a target field."""
    name = rule["name"]
    pattern = re.compile(rule["pattern"], re.DOTALL)
    target = rule["target"]
    replace = rule.get("replace", "")

    def apply(response: dict) -> dict:
        val = response.get(target)
        if val is None:
            return response
        new_val = pattern.sub(replace, val)
        if new_val != val:
            log.debug("Rule strip/%s fired on %s", name, target)
        response[target] = new_val
        return response

    return apply


def _make_map_finish_reason(rule: dict) -> RuleFn:
    """Build a finish_reason mapping rule."""
    from_val = rule["from"]
    to_val = rule["to"]

    def apply(response: dict) -> dict:
        if response.get("finish_reason") == from_val:
            log.debug("Rule map_finish_reason: %s -> %s", from_val, to_val)
            response["finish_reason"] = to_val
        return response

    return apply


def _make_extract_tool_calls(rule: dict) -> RuleFn:
    """Build an extract_tool_calls rule (placeholder — complex extraction is code modules)."""
    guard = rule.get("guard", {})
    source = rule.get("source", "content")

    def apply(response: dict) -> dict:
        if guard and not _check_guard(guard, response):
            return response
        # Only act if tool_calls is empty but source field has content
        if response.get("tool_calls"):
            return response
        val = response.get(source)
        if not val:
            return response
        # Attempt JSON extraction from content — basic pattern only
        # Complex extraction should be in code modules
        log.debug("Rule extract_tool_calls: checking %s field", source)
        return response

    return apply


def _make_fix_tool_call(rule: dict) -> RuleFn:
    """Build a fix_tool_call rule for missing IDs."""
    guard = rule.get("guard", {})
    when = rule.get("when", "")
    strategy = rule.get("strategy", "generate")

    def apply(response: dict) -> dict:
        if guard and not _check_guard(guard, response):
            return response
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            return response
        if when == "id_missing" and strategy == "generate":
            for i, tc in enumerate(tool_calls):
                if isinstance(tc, dict) and not tc.get("id"):
                    tc["id"] = f"call_0_{i}"
                    log.debug("Rule fix_tool_call: generated id for tool call %d", i)
        return response

    return apply


# Registry of rule type builders
_RULE_BUILDERS: dict[str, Callable[[dict], RuleFn]] = {
    "strip": _make_strip,
    "map_finish_reason": _make_map_finish_reason,
    "extract_tool_calls": _make_extract_tool_calls,
    "fix_tool_call": _make_fix_tool_call,
}


def parse_normalize_rules(path: Path) -> list[RuleFn]:
    """Parse a normalize rules.toml and return a list of rule functions."""
    if not path.is_file():
        return []
    with open(path, "rb") as f:
        data = tomllib.load(f)

    rules: list[RuleFn] = []
    for rule_type, builder in _RULE_BUILDERS.items():
        for rule in data.get(rule_type, []):
            rules.append(builder(rule))
    return rules


def parse_denormalize_rules(path: Path) -> dict[str, Any]:
    """Parse a denormalize rules.toml and return config dict.

    Denormalize rules are simpler — mostly config values like system_prompt strategy.
    """
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data


def apply_normalize_rules(rules: list[RuleFn], response: dict) -> dict:
    """Apply all normalize rules in order to a response dict."""
    for rule_fn in rules:
        response = rule_fn(response)
    return response
