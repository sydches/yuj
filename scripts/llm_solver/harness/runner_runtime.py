"""Observe executable candidates for declared checks within startup bounds."""
import json
import hashlib
import shlex

from ..language_quirks import _load_runner_quirk_dict

MAX_ENVIRONMENTS = 8


def observe_runner_runtime(declarations, facts, *, requested, inspect, folders):
    declared = [item["runner"] for item in declarations["candidates"]]
    explicit = requested not in ("", "auto")
    runners = [requested] if explicit else declared
    report = {"source": "runner_runtime", "request_source": "explicit_configuration" if explicit else "project_declarations",
              "declared_runners": declared, "candidates": [], "status": "unresolved",
              "limited": any(f.get("inventory_incomplete") is True for f in facts
                             if f["source"] == "sandbox_filesystem")}
    inventory = next((f.get("available", {}) for f in facts if f["source"] == "executable_locations"), {})
    project_files = {f["path"]: f.get("values", {}) for f in facts if f["source"] == "project_file"}
    unknown_files = {f["path"] for f in facts if f["source"] == "project_file"
                     and f.get("status") not in {"observed", "present"}}
    for runner in runners:
        if runner == "generic":
            continue
        spec = _load_runner_quirk_dict(runner).get("runtime", {})
        commands = spec.get("executables", [])
        manager_field = spec.get("manager_declaration")
        if manager_field:
            manager = project_files.get(spec["manager_file"], {}).get(manager_field)
            if manager:
                command = str(manager).split("@", 1)[0]
                commands = [command] if command in spec["manager_commands"] else []
        executables = [(inventory[name], "current_PATH") for name in commands if inventory.get(name)]
        env_executable = spec.get("environment_executable")
        local_inventory_known = True
        if env_executable:
            # Candidate names come from observed task folders, not a preferred
            # benchmark environment name. The shell checks their metadata in
            # the execution namespace, including valid environment symlinks.
            markers = spec["environment_markers"]
            roots = ["./" + name for name in folders]
            report["limited"] |= any(
                f.get("root_inventory_complete") is False for f in facts
                if f["source"] == "task_directory_scan")
            if roots:
                condition = " || ".join('[ -f "$directory/"' + shlex.quote(marker) + ' ]' for marker in markers)
                script = ("for directory in " + shlex.join(roots) + "; do if " + condition
                          + "; then printf '%s\\0' \"$directory/\"" + shlex.quote(env_executable) + "; fi; done")
                output = inspect(f"{runner}_project_environments", script)
                local_inventory_known = output == "" or (output is not None and output.endswith("\0"))
                if output is not None and output.endswith("\0"):
                    executables = [(path, "project_environment_marker") for path in output.split("\0")[:-1]] + executables
            for fact in facts:
                roots = fact.get("environment_candidates", [])
                if not isinstance(roots, list):
                    continue
                report["limited"] |= len(roots) > MAX_ENVIRONMENTS
                for root in roots[:MAX_ENVIRONMENTS]:
                    if isinstance(root, str) and root.startswith("/"):
                        executables.append((root.rstrip("/") + "/" + env_executable, fact["source"]))
        executables = list(dict.fromkeys(executables))
        report["limited"] |= len(executables) > MAX_ENVIRONMENTS
        seen, usable = set(), []
        for executable, source in executables[:MAX_ENVIRONMENTS]:
            if executable in seen or "\n" in executable or "/" not in executable:
                continue
            seen.add(executable)
            args = list(spec["probe_args"])
            constraints_known = spec.get("manager_file") not in unknown_files
            for key in spec.get("project_arguments", []):
                path, field = key.split(":", 1)
                constraints_known &= path not in unknown_files
                args.append(str(project_files.get(path, {}).get(field) or ""))
            output = inspect(f"{runner}_candidate_{len(report['candidates'])}",
                             shlex.join([executable, *args]), spec.get("environment"))
            candidate = {"runner": runner, "executable": executable, "source": source, "status": "not_established",
                         "runtime_spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}
            if spec.get("language"):
                candidate["language"] = spec["language"]
            if output is not None:
                available = True
                if spec.get("probe_format") == "json":
                    try:
                        details = json.loads(output)
                        if not constraints_known:
                            details["compatible"] = None
                            details["compatibility_reason"] = "project constraint source unavailable"
                        available = details.get("available") is True and details.get("compatible") is True
                        candidate["runtime"] = details
                        if details.get("available") is False:
                            candidate["status"] = "not_importable"
                        elif details.get("compatible") is False:
                            candidate["status"] = "incompatible"
                    except (ValueError, AttributeError):
                        available = False
                else:
                    candidate["version_output"] = output.strip()
                if available and constraints_known:
                    candidate["status"] = "available"
                    candidate["base_cmd"] = shlex.join([executable, *spec["command_args"]])
                    if spec.get("environment"):
                        candidate["environment"] = spec["environment"]
                    usable.append(candidate)
            report["candidates"].append(candidate)
        # Distinct aliases of the same interpreter/environment are one choice.
        unique = {}
        for candidate in usable:
            runtime = candidate.get("runtime", {})
            key = runtime.get("prefix") or candidate["executable"]
            unique.setdefault(key, candidate)
        usable = list(unique.values())
        local = [c for c in usable if c["source"] == "project_environment_marker"]
        current = [c for c in usable if c["source"] == "current_PATH"]
        preferred = local or current or usable
        def priority(candidate):
            return 0 if candidate["source"] == "project_environment_marker" else 1 if candidate["source"] == "current_PATH" else 2
        unknown_peer = bool(preferred) and any(
            candidate["runner"] == runner and candidate["status"] == "not_established"
            and priority(candidate) <= priority(preferred[0]) for candidate in report["candidates"]
        )
        if len(preferred) == 1 and len(runners) == 1 and local_inventory_known and not unknown_peer:
            report["status"] = "selected"
            report["selected"] = preferred[0]
        elif (len(preferred) > 1 or len(runners) > 1) and local_inventory_known:
            report["status"] = "ambiguous"
    if not runners or runners == ["generic"]:
        report["status"] = "no_declared_check"
    if report["limited"]:
        report["status"] = "unresolved"
        report.pop("selected", None)
    report["meaning"] = ("Declarations identify checks; probes establish executable availability and runtime compatibility. "
                         "Project-local environments take precedence, then the current PATH, then a sole available manager candidate. "
                         "Dependency completeness and test coverage require actual verification.")
    return report
