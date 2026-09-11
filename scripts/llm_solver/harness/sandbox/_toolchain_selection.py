"""Query an installed manager in a restricted, read-only startup namespace."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import tomllib

from ..._shared.paths import package_data_path
from ..prompt_imports import _UnreadableMatcher
from ._filesystem import MAX_CANDIDATES, Mount, build_filesystem_argv
from .env_policy import build_bwrap_env_argv, build_subprocess_env


def select_native_toolchain(view, environment, unreadable_paths, *, deadline, bwrap_bin):
    from ..runtime_discovery import MAX_SOURCE_BYTES
    from ._unreadable import _expand_unreadable_paths

    descriptor_bytes = package_data_path(
        __package__.rsplit(".harness", 1)[0] + ".language_quirks", "runtime.toml",
    ).read_bytes()
    if hashlib.sha256(descriptor_bytes).hexdigest() != view.descriptor_sha256:
        return replace(view, unresolved=(*view.unresolved, "native_selection:descriptor_changed"))
    descriptor = tomllib.loads(descriptor_bytes.decode())["filesystem"]
    spec = descriptor.get("native_manager")
    if not spec:
        return view
    task = Path(view.cwd)
    blocked = _UnreadableMatcher(task, unreadable_paths)
    search = ":".join(str(task / p) for p in environment.get("PATH", "").split(":"))
    found = shutil.which(spec["name"], path=search)
    if not found:
        return view
    manager = Path(found).absolute()
    if blocked.blocks(manager) or manager.is_relative_to(task) or manager.resolve().is_relative_to(task):
        return view
    proxies = []
    for name in spec["proxies"]:
        path = shutil.which(name, path=search)
        if path and not blocked.blocks(Path(path)) and Path(path).samefile(manager):
            proxies.append(name)
    if not proxies:
        return view
    started = time.monotonic()
    record = {"manager": spec["name"], "executable": str(manager), "status": "unresolved",
              "inputs": []}

    def unresolved(reason):
        record.update(reason=reason, duration_seconds=time.monotonic() - started)
        return replace(view, unresolved=tuple(sorted({*view.unresolved,
                       "unverified_dependencies:" + spec["name"] + ":" + reason})),
                       native_selections=(*view.native_selections, record))

    def read(path):
        if time.monotonic() >= deadline:
            raise ValueError("discovery_budget_exhausted")
        if blocked.blocks(path):
            raise ValueError("selection_metadata_masked")
        with path.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("selection_metadata_limit")
        record["inputs"].append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        return raw

    home_value = environment.get(spec["home_variable"])
    if home_value is None:
        if not view.home:
            return unresolved("manager_home_unknown")
        home_value = str(Path(view.home) / spec["default_home"])
    home = Path(home_value)
    if not home.is_absolute() or blocked.blocks(home):
        return unresolved("manager_home_unavailable")
    layout = next(item for item in descriptor["layouts"] if item["name"] == spec["layout"])
    candidates = {}

    def candidate(prefix):
        prefix = Path(prefix).absolute()
        root = prefix.resolve()
        if (len(root.parts) < 3 or root == Path(view.home).resolve() or root in task.parents
                or blocked.blocks(prefix)):
            return
        if not all((prefix / m).exists() and not blocked.blocks(prefix / m)
                   and (prefix / m).resolve().is_relative_to(root) for m in layout["markers"]):
            return
        mounts = []
        for part in layout["components"]:
            path = prefix / part
            if path.exists() and not blocked.blocks(path) and path.resolve().is_relative_to(root):
                if not path.is_relative_to(task):
                    kind = "task_alias" if path.resolve().is_relative_to(task) else "read_only"
                    mounts.append(Mount(str(path.resolve()), str(path),
                                        "native_manager:" + spec["name"] + "; layout=" + layout["name"], kind))
        candidates[str((prefix / spec["query_executable"]).absolute())] = (prefix, mounts)

    try:
        settings_path = home / spec["settings"]
        if not settings_path.resolve().is_relative_to(home.resolve()):
            return unresolved("selection_metadata_outside_manager")
        settings = tomllib.loads(read(settings_path).decode()) if settings_path.exists() else {}
        # Retain default selection and task-local overrides only. Do not make
        # unrelated project paths or ancestor overrides visible to the query.
        permitted = {k: v for k, v in settings.items() if k in spec["settings_fields"] and isinstance(v, str)}
        overrides = settings.get(spec["overrides_field"], {})
        permitted_overrides = {k: v for k, v in overrides.items()
                               if isinstance(k, str) and isinstance(v, str)
                               and Path(k).is_absolute() and Path(k).resolve().is_relative_to(task)}
        settings_text = "".join(f"{json.dumps(k, ensure_ascii=False)} = {json.dumps(v, ensure_ascii=False)}\n" for k, v in permitted.items())
        settings_text += "[" + spec["overrides_field"] + "]\n"
        settings_text += "".join(f"{json.dumps(k, ensure_ascii=False)} = {json.dumps(v, ensure_ascii=False)}\n" for k, v in permitted_overrides.items())
        inventory = home / spec["inventory"]
        if inventory.exists() and not blocked.blocks(inventory):
            with os.scandir(inventory) as entries:
                for i, entry in enumerate(entries):
                    if i >= MAX_CANDIDATES or time.monotonic() >= deadline:
                        return unresolved("selection_inventory_limit")
                    if entry.is_dir():
                        candidate(entry.path)
        selector = environment.get(spec["selection_variable"], "")
        if selector.startswith("/"):
            candidate(selector)
        for name in spec["project_files"]:
            path = task / name
            if path.exists():
                resolved_path = path.resolve()
                if not (resolved_path.is_relative_to(task) or any(
                        mount.kind == "read_only" and resolved_path.is_relative_to(mount.source)
                        for mount in view.mounts)):
                    record.setdefault("unavailable_inputs", []).append(
                        {"path": str(path), "reason": "outside_permitted_view"})
                    continue
                try:
                    value = tomllib.loads(read(path).decode())
                except tomllib.TOMLDecodeError:
                    continue  # Legacy channel syntax is interpreted by the manager.
                for key in spec["project_path_field"]:
                    value = value.get(key, {}) if isinstance(value, dict) else {}
                if isinstance(value, str) and value.startswith("/"):
                    candidate(value)
        if not candidates:
            return unresolved("no_installed_toolchain_candidates")
        candidate_mounts = {m.target: m for m in view.mounts}
        for _, mounts in candidates.values():
            candidate_mounts.update((m.target, m) for m in mounts)
        query_view = replace(view, mounts=tuple(candidate_mounts.values()))
        query_env = {**environment, spec["home_variable"]: str(home), **spec["probe_environment"]}
        masks, _, _ = _expand_unreadable_paths(tuple(unreadable_paths), sandbox_required=True)
        with tempfile.TemporaryDirectory(prefix="yuj-native-selection-") as temporary:
            settings_copy = Path(temporary) / "settings.toml"
            settings_copy.write_text(settings_text)
            task_root = view.task_root
            filesystem_args = build_filesystem_argv(query_view, task_writable=False, task_root=task_root)
            argv = [bwrap_bin, *filesystem_args,
                    "--ro-bind", str(settings_copy), str(settings_path), *masks,
                    "--unshare-all", "--cap-drop", "ALL", "--die-with-parent", "--chdir", view.cwd,
                    *build_bwrap_env_argv(query_env), "--", str(manager), *spec["query"]]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return unresolved("discovery_budget_exhausted")
            record.update(query=spec["query"], executed=True)
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=remaining,
                                        env=build_subprocess_env(query_env),
                                        pass_fds=filesystem_args.pass_fds)
                stdout.seek(0)
                stderr.seek(0)
                output = stdout.read(MAX_SOURCE_BYTES + 1)
                errors = stderr.read(MAX_SOURCE_BYTES + 1)
            if len(output) > MAX_SOURCE_BYTES or len(errors) > MAX_SOURCE_BYTES:
                return unresolved("native_query_output_limit")
        record.update(query=spec["query"], exit_code=result.returncode,
                      settings_sha256=hashlib.sha256(settings_text.encode()).hexdigest(),
                      output_sha256=hashlib.sha256(output + errors).hexdigest())
        if result.returncode:
            return unresolved("native_query_failed")
        selected = output.decode().strip()
        if selected not in candidates:
            return unresolved("native_selection_outside_candidates")
        prefix, mounts = candidates[selected]
        merged = {m.target: m for m in view.mounts}
        merged.update((m.target, m) for m in mounts)
        record.update(status="selected", toolchain=str(prefix), compiler=selected,
                      duration_seconds=time.monotonic() - started)
        resolved = {"unverified_dependencies:" + name for name in [spec["name"], *proxies]}
        return replace(view, mounts=tuple(merged.values()),
                       private_directories=(*view.private_directories, str(home)),
                       unresolved=tuple(r for r in view.unresolved if r not in resolved),
                       runtime_bindings=(*view.runtime_bindings, (spec["selection_variable"], str(prefix))),
                       native_selections=(*view.native_selections, record))
    except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired) as exc:
        return unresolved(type(exc).__name__)
