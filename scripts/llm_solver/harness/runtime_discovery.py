"""Bounded startup observations through the task's existing access boundary."""
from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
import tomllib

from .._shared.paths import package_data_path
from ._tools._common import _resolve
from ._tools._run_in_sandbox import _run_in_sandbox
from .prompt_imports import _UnreadableMatcher

MAX_PROBES = 12
MAX_SOURCE_BYTES = 65536
MAX_LAYOUT_ENTRIES = 512
MAX_LAYOUT_DEPTH = 2


def bind_command_environment(cfg, report, environment):
    """Bind command lookup to a uniquely observed runtime, within env policy.

    Runner descriptors identify which SDK observation supplies the executable.
    Explicit PATH policy takes precedence. This changes lookup only and does
    not run activation hooks or grant access to additional filesystem paths.
    """
    from ..language_quirks import _load_runner_quirk_dict

    environment = {**environment, **report.get("filesystem_view", {}).get("runtime_bindings", {})}
    selection = report.get("runner_selection", {})
    selected = selection.get("selected", {})
    if selection.get("status") != "selected" or selected.get("status") != "available":
        return dict(environment)
    spec = _load_runner_quirk_dict(selected.get("runner", "generic")).get("runtime", {})
    field = spec.get("command_path_from_runtime")
    if not field:
        return dict(environment)
    fact = {"source": "command_runtime_binding", "runner": selected["runner"],
            "status": "not_applied"}
    executable = selected.get("runtime", {}).get(field)
    result = dict(environment)
    if "PATH" in (getattr(cfg, "sandbox_env_set", {}) or {}):
        fact["reason"] = "explicit PATH policy takes precedence; use the observed absolute test command"
    elif not environment.get("PATH"):
        fact["reason"] = "PATH is absent under the declared environment policy"
    elif not isinstance(executable, str) or not executable.startswith("/") or ":" in executable or "\n" in executable:
        fact["reason"] = "the runtime did not report a usable absolute executable path"
    else:
        directory = str(Path(executable).parent)
        result["PATH"] = ":".join([directory, *[p for p in environment["PATH"].split(":") if p != directory]])
        fact.update(status="bound", executable=executable, path_prefix=directory,
                    meaning="Command PATH starts with this observed runtime directory. Discovery probes describe the initial environment. Activation hooks were not run.")
    # Keep this execution decision visible even when optional discovery facts
    # fill the model briefing. Full observations and probes remain in provenance.
    report["command_binding"] = fact
    report.setdefault("observations", []).append(fact)
    facts = report.setdefault("facts", [])
    facts.insert(0, fact)
    report["facts_chars"] = len(json.dumps(facts, ensure_ascii=True))
    return result


def _inspect_layout(cwd, spec, blocked, deadline):
    """Read a bounded task view without descending into dependency/VCS trees."""
    pending = [(cwd, 0)]
    folders, languages = [], {}
    scanned = 0
    limited = False
    root_inventory_complete = False
    while pending and time.monotonic() < deadline:
        directory, depth = pending.pop(0)
        if depth > 0 and scanned >= MAX_LAYOUT_ENTRIES:
            limited = True
            break
        try:
            entries = []
            with os.scandir(directory) as stream:
                for entry in stream:
                    if (depth > 0 and scanned >= MAX_LAYOUT_ENTRIES) or time.monotonic() >= deadline:
                        limited = True
                        break
                    scanned += 1
                    entries.append(entry)
                else:
                    if depth == 0:
                        root_inventory_complete = True
            for entry in sorted(entries, key=lambda item: item.name):
                path = Path(entry.path)
                try:
                    # Layout sampling never follows links. An excluded link
                    # does not make the inventory of root directories unknown.
                    if entry.is_symlink():
                        continue
                    target = _resolve(str(cwd), str(path))
                    if blocked.blocks(target):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth == 0:
                            folders.append(entry.name)
                        if entry.name not in spec.get("skip_descent", []):
                            if depth < MAX_LAYOUT_DEPTH:
                                pending.append((path, depth + 1))
                            else:
                                limited = True
                    elif entry.is_file(follow_symlinks=False):
                        language = spec.get("source_suffixes", {}).get(path.suffix)
                        if language:
                            observed = languages.setdefault(language, {"files": 0, "examples": []})
                            observed["files"] += 1
                            if len(observed["examples"]) < 2:
                                observed["examples"].append(str(path.relative_to(cwd)))
                except (OSError, ValueError):
                    if depth == 0:
                        root_inventory_complete = False
                    continue
        except OSError:
            limited = True
    return {"source": "task_directory_scan", "folders": folders,
            "source_languages": languages, "entries_examined": scanned,
            "limited": limited or bool(pending),
            "root_inventory_complete": root_inventory_complete,
            "meaning": "Source suffixes identify observed languages, not a single primary language. Dependency and VCS directories are not traversed."}


def _selected_fields(data, spec):
    selected = {}
    for kind in ("fields", "sections"):
        for keys in spec.get(kind, []):
            value = data
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    break
                value = value[key]
            else:
                selected[".".join(keys)] = True if kind == "sections" else value
    return selected


def _parse_fields(raw, spec):
    if spec["format"] == "json":
        data = json.loads(raw)
    elif spec["format"] == "toml":
        data = tomllib.loads(raw)
    elif spec["format"] == "ini":
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(raw)
        data = {section: dict(parser[section]) for section in parser.sections()}
    else:
        return raw.strip()
    if not isinstance(data, dict):
        raise ValueError("expected a mapping")
    return _selected_fields(data, spec)


def selection_inputs(cwd, names, unreadable_paths=()):
    """Fingerprint permitted declaration inputs, including absent sources."""
    from ..language_quirks._discovery import read_declaration
    from .time_budget import execution_deadline, remaining_before
    blocked = _UnreadableMatcher(cwd, unreadable_paths)
    result = {}
    for name in sorted(names):
        remaining_before(execution_deadline())
        try:
            if blocked.blocks(_resolve(str(cwd), name)):
                raise ValueError("blocked declaration")
            raw = read_declaration(cwd, name)
            result[name] = {"sha256": hashlib.sha256(raw).hexdigest()} if raw is not None else {"status": "missing"}
        except (OSError, ValueError):
            result[name] = {"status": "unavailable"}
    return result


def discover_runtime(cwd: Path, cfg, *, effective_env, unreadable_paths=(), selection_only=False):
    """Return candidate briefing facts and complete probe/cost provenance.

    Tool and project knowledge lives in descriptor data. No probe installs,
    activates, or executes project setup. Selection requires a declared check
    and an observed usable runtime; it does not establish passing tests.
    The caller runs this once before prompt assembly, inside the solve budget.
    """
    started = time.monotonic()
    from .time_budget import execution_deadline
    caller_deadline = execution_deadline()
    deadline = caller_deadline if caller_deadline is not None else float("inf")
    descriptor = package_data_path(
        __package__.rsplit(".", 1)[0] + ".language_quirks", "runtime.toml",
    )
    descriptor_bytes = descriptor.read_bytes()
    spec = tomllib.loads(descriptor_bytes.decode())["discovery"]
    commands = spec["commands"]
    if (not isinstance(commands, list) or len(commands) > 64
            or len(set(commands)) != len(commands)
            or any(not isinstance(n, str) or not re.fullmatch(r"[a-zA-Z0-9_+-]+", n)
                   for n in commands)):
        raise ValueError("invalid runtime discovery command registry")
    report = {
        "facts": [], "observations": [], "probes": [], "omitted_facts": 0,
        "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
    }
    from .sandbox import container_mode
    if cfg.sandbox_bash and cfg.sandbox_backend == "bwrap" and container_mode() is None:
        from .sandbox._filesystem import freeze_filesystem_view
        view = freeze_filesystem_view(
            cwd, effective_env, getattr(cfg, "skills_readable_dirs", ()) or (),
            unreadable_paths,
            deadline=deadline, bwrap_bin=cfg.bwrap_bin,
        )
        report["filesystem_view"] = view.record()
        for selection in view.native_selections:
            if selection.get("executed"):
                report["probes"].append({
                    "id": "native_manager:" + selection["manager"],
                    "command": shlex.join([selection["executable"], *selection["query"]]),
                    "status": "observed" if selection["status"] == "selected" else "failed",
                    "elapsed_seconds": selection["duration_seconds"],
                    "exit_code": selection.get("exit_code"),
                    "output_sha256": selection.get("output_sha256"),
                })

    def add_fact(fact):
        report["observations"].append(fact)
        # Admission happens against the assembled request, using its counter.
        report["facts"].append(fact)

    if getattr(cfg, "sandbox_env_inherit", "core") == "runtime":
        from .sandbox.env_policy import RUNTIME_ENVIRONMENT_NAMES

        # Report the mapping actually passed to probes and commands. A policy
        # override is not an inherited observation, and a selector is not proof
        # that the requested installation exists or is suitable for this task.
        fixed = getattr(cfg, "sandbox_env_set", {}) or {}
        for name in sorted(RUNTIME_ENVIRONMENT_NAMES):
            if name in effective_env:
                add_fact({"source": "command_runtime_setting", "name": name,
                          "value": effective_env[name],
                          "origin": "explicit_policy" if name in fixed else "inherited_environment",
                          "meaning": "Effective command setting; not proof of an installed or suitable runtime."})

    if "filesystem_view" in report:
        add_fact({"source": "sandbox_filesystem", "policy": view.record()["policy"],
                  "runtime_components": len(view.mounts),
                  "inventory_incomplete": any(
                      not reason.startswith("unverified_dependencies:") for reason in view.unresolved
                  ), "unresolved": list(view.unresolved)})
        for selection in view.native_selections:
            add_fact({"source": "native_toolchain_selection", **selection})

    blocked = _UnreadableMatcher(cwd, unreadable_paths)
    source_cache = {}

    def read_project(name):
        from ..language_quirks._discovery import read_declaration
        if time.monotonic() >= deadline:
            raise ValueError("observation budget exhausted")
        if name not in source_cache:
            target = _resolve(str(cwd), name)
            if blocked.blocks(target):
                raise ValueError("blocked declaration")
            source_cache[name] = read_declaration(cwd, name)
        return source_cache[name]

    layout = _inspect_layout(cwd, spec, blocked, deadline)
    add_fact(layout)
    observed_languages = set(layout["source_languages"])
    for file_spec in spec.get("files", []):
        name = file_spec["path"]
        fact = {"source": "project_file", "path": name}
        try:
            target = _resolve(str(cwd), name)
            if blocked.blocks(target):
                fact["status"] = "blocked"
            elif not target.is_file():
                continue
            elif file_spec["format"] == "presence":
                fact["status"] = "present"
            else:
                if target.stat().st_size > MAX_SOURCE_BYTES:
                    fact["status"] = "too_large"
                else:
                    raw = read_project(name)
                    if raw is None:
                        raise ValueError("declaration disappeared")
                    fact["sha256"] = hashlib.sha256(raw).hexdigest()
                    fact["values"] = _parse_fields(raw.decode("utf-8"), file_spec)
                    fact["status"] = "observed"
        except (OSError, ValueError, configparser.Error):
            # No exception text: it may reveal blocked targets or file contents.
            fact["status"] = "unavailable_or_invalid"
        if fact["status"] in {"observed", "present"} and file_spec.get("language"):
            fact["language_hint"] = file_spec["language"]
            observed_languages.add(file_spec["language"])
        add_fact(fact)

    from ..language_quirks._discovery import inspect_runner_candidates
    declarations = inspect_runner_candidates(cwd, read_file=read_project)
    report["runner_declarations"] = declarations
    if declarations["candidates"]:
        add_fact({"source": "project_checks", "candidates": declarations["candidates"]})

    def inspect(identifier, command, overrides=None):
        remaining = deadline - time.monotonic()
        record = {"id": identifier, "command": command}
        if overrides:
            record["environment_overrides"] = overrides
        if len(report["probes"]) >= MAX_PROBES + 1:
            record["status"] = "probe_limit"
            report["probes"].append(record)
            return None
        if remaining <= 0:
            record["status"] = "budget_exhausted"
            report["probes"].append(record)
            return None
        probe_started = time.monotonic()
        from .sandbox.policy import sandbox_execution_kwargs
        output, code, timed_out = _run_in_sandbox(
            command, cwd=str(cwd), timeout=remaining if caller_deadline is not None else None,
            bwrap_bin=cfg.bwrap_bin,
            **sandbox_execution_kwargs(cfg),
            unreadable_paths=tuple(unreadable_paths),
            readable_paths=tuple(getattr(cfg, "skills_readable_dirs", ()) or ()),
            effective_env={**effective_env, **overrides} if overrides else effective_env,
            allow_login_shell=cfg.sandbox_env_allow_login_shell,
            normalize_output=False, normalize_addresses=False,
        )
        record.update(exit_code=code, timed_out=timed_out,
                      elapsed_seconds=round(time.monotonic() - probe_started, 6))
        record["status"] = "observed" if code == 0 and not timed_out else "failed"
        if len(output.encode("utf-8")) > MAX_SOURCE_BYTES:
            record["status"] = "too_large"
        record["output_sha256"] = hashlib.sha256(output.encode("utf-8")).hexdigest()
        report["probes"].append(record)
        return output if record["status"] == "observed" else None

    inventory_command = (
        "for name in " + shlex.join(commands) + "; do "
        'location=$(command -v "$name" 2>/dev/null) || location=; '
        "printf '%s\\0%s\\0' \"$name\" \"$location\"; done"
    )
    inventory = inspect("executable_locations", inventory_command)
    locations = {}
    if inventory is not None:
        parts = inventory.split("\0")
        if (len(parts) != 2 * len(commands) + 1 or parts[-1] != ""
                or parts[:-1:2] != commands):
            report["probes"][-1]["status"] = "invalid_output"
        else:
            locations = dict(zip(commands, parts[1:-1:2]))
    if locations:
        add_fact({"source": "executable_locations", "status": "observed",
                  "available": {n: p for n, p in locations.items() if p},
                  "not_found": [n for n, p in locations.items() if not p]})
    else:
        add_fact({"source": "executable_locations",
                  "status": report["probes"][-1]["status"]})

    attempted = 0
    probes = sorted(spec.get("probes", []), key=lambda p: not observed_languages.intersection(p.get("languages", [])))
    # Environment locations are prerequisites for runner inspection. Optional
    # version calls must not consume the remaining budget before that work.
    probes = [p for p in probes if p.get("environment_candidates_field")] + [None] + [p for p in probes if not p.get("environment_candidates_field")]
    for probe in probes:
        if probe is None:
            from .runner_runtime import observe_runner_runtime
            selection = observe_runner_runtime(
                declarations, report["observations"], requested=cfg.analysis_task_format,
                inspect=inspect, folders=layout["folders"],
            )
            report["runner_selection"] = selection
            if time.monotonic() < deadline:
                selection["declaration_inputs"] = selection_inputs(
                    cwd, set(source_cache) | {f["path"] for f in spec.get("files", [])}, unreadable_paths)
            else:
                selection.update(status="unresolved", limited=True)
                selection.pop("selected", None)
            if selection["candidates"] or declarations["candidates"]:
                add_fact(selection)
            if selection_only:
                break
            continue
        executable = locations.get(probe["command"])
        if not executable:
            continue
        # command -v may report a function or alias instead of an executable.
        if "/" not in executable or "\n" in executable:
            add_fact({"source": probe["id"], "status": "not_an_executable_path"})
            continue
        if attempted >= MAX_PROBES:
            add_fact({"source": probe["id"], "status": "probe_limit"})
            continue
        attempted += 1
        command = (
            probe["script"].replace("{executable}", shlex.quote(executable))
            if "script" in probe else shlex.join([executable, *probe["args"]])
        )
        raw = inspect(probe["id"], command, probe.get("environment"))
        fact = {"source": probe["id"], "executable": executable,
                "status": report["probes"][-1]["status"]}
        if probe.get("environment"):
            fact["environment_overrides"] = probe["environment"]
        if raw is not None:
            try:
                values = _parse_fields(raw, probe)
                if probe["format"] == "json" and not values:
                    raise ValueError("expected observation fields")
                fact["values"] = values
                if probe.get("environment_candidates_field"):
                    fact["environment_candidates"] = values.get(probe["environment_candidates_field"], [])
                fact["meaning"] = probe["meaning"]
            except (ValueError, configparser.Error):
                fact["status"] = report["probes"][-1]["status"] = "invalid_output"
        add_fact(fact)
    report["elapsed_seconds"] = round(time.monotonic() - started, 6)
    report["facts_chars"] = len(json.dumps(report["facts"], ensure_ascii=True))
    return report
