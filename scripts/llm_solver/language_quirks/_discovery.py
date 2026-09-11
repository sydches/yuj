"""Bounded, declarative project-check detection; never execute project code."""
import configparser
import hashlib
import json
from pathlib import Path
import re
import tomllib

MAX_DECLARATION_BYTES = 65536


def read_declaration(cwd, name):
    # Runtime readers may supply a native path object. Keep its I/O methods
    # rather than coercing it into a path on the harness host.
    root = Path(cwd).resolve() if isinstance(cwd, str) else cwd.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("declaration escapes task")
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        raw = stream.read(MAX_DECLARATION_BYTES + 1)
    if len(raw) > MAX_DECLARATION_BYTES:
        raise ValueError("declaration exceeds observation bound")
    return raw


def _lookup(data, keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _matches(raw, rule):
    fmt = rule["format"]
    if fmt == "presence":
        return True, None
    text = raw.decode("utf-8")
    if fmt == "toml":
        data = tomllib.loads(text)
    elif fmt == "json":
        data = json.loads(text)
    elif fmt == "ini":
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(text)
        data = {key: dict(parser[key]) for key in parser.sections()}
    elif fmt == "text":
        data = text
    else:
        raise ValueError("unsupported declaration format")
    if rule.get("exclude_field") and _lookup(data, rule["exclude_field"]) is not None:
        return False, None
    if "section" in rule:
        return isinstance(_lookup(data, rule["section"]), dict), None
    value = _lookup(data, rule["field"]) if "field" in rule else data
    if "pattern" in rule:
        match = re.search(rule["pattern"], value) if isinstance(value, str) else None
        return match is not None, (match.group(0) if match and fmt == "text" else value)
    return value is not None, value


def inspect_runner_candidates(cwd, *, read_file=None):
    """Return candidates and the permitted observations that support them.

    Runtime callers supply their access-controlled reader. Each source is read
    once. Invalid, blocked or oversized sources cannot select a runner.
    """
    from . import _load_runner_quirk_dict, list_run_test_runner_descriptors

    reader = read_file or (lambda name: read_declaration(cwd, name))
    cache, candidates, observations = {}, [], []
    for descriptor in list_run_test_runner_descriptors():
        evidence = []
        rules = _load_runner_quirk_dict(descriptor.name).get("detection", [])
        for index, rule in enumerate(rules):
            name = rule["path"]
            observation = {"runner": descriptor.name, "rule": index, "path": name,
                           "rule_sha256": hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()}
            if name not in cache:
                try:
                    cache[name] = reader(name)
                except (OSError, ValueError):
                    cache[name] = False
            raw = cache[name]
            if raw is None:
                continue
            if raw is False:
                observation["status"] = "unavailable"
            else:
                observation["sha256"] = hashlib.sha256(raw).hexdigest()
                try:
                    matched, value = _matches(raw, rule)
                    observation["status"] = "matched" if matched else "not_matched"
                    if matched:
                        if value is not None:
                            observation["value"] = value
                        evidence.append(observation)
                except (ValueError, configparser.Error):
                    observation["status"] = "invalid"
            observations.append(observation)
        if evidence:
            candidates.append({"runner": descriptor.name, "evidence": evidence})
    return {"candidates": candidates, "observations": observations}


def selected_runner(report):
    candidates = report["candidates"]
    return candidates[0]["runner"] if len(candidates) == 1 else "generic"
