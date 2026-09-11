"""Install a descriptor-owned collection hook in the current execution view."""
from contextlib import contextmanager
import json
import os
import subprocess
import uuid

from ..language_quirks import FORMATS_DIR
from .task_path import TaskPath, resolve_task_path
from .time_budget import BudgetExhausted


def _create(path, data):
    if isinstance(path, TaskPath):
        path.files.create_bytes(str(path), data)
    else:
        with path.open("xb") as stream:
            stream.write(data)


@contextmanager
def component_selection(cwd, source, quirk, environment):
    record = {}
    if not source:
        yield environment, record
        return
    plugin = quirk.extra_fields.get("component_collection_plugin")
    if not plugin:
        raise ValueError("selected runner has no native component collection hook")
    root = resolve_task_path(cwd, ".")
    directory = root / ".tool_output"
    if directory.is_symlink():
        raise ValueError("component report directory is a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    module = "yuj_component_" + uuid.uuid4().hex
    code_path = directory / (module + ".py")
    report_path = directory / (module + ".json")
    created = []
    try:
        _create(code_path, (FORMATS_DIR / plugin).read_bytes())
        created.append(code_path)
        _create(report_path, b"")
        created.append(report_path)
        sources = source if isinstance(source, (list, tuple)) else [source]
        names = {}
        for value in sources:
            src = resolve_task_path(cwd, value)
            names[str(src.relative_to(root))] = [
                pattern.format(stem=src.stem, suffix=src.suffix)
                for pattern in quirk.component_test_names]
        env = {**environment,
               "PYTHONPATH": os.pathsep.join(filter(None, [str(directory), environment.get("PYTHONPATH")])),
               "PYTEST_PLUGINS": ",".join(filter(None, [environment.get("PYTEST_PLUGINS"), module])),
               "PYTHONDONTWRITEBYTECODE": "1",
               "YUJ_COMPONENT_NAMES": json.dumps(names),
               "YUJ_COMPONENT_ROOT": str(root),
               "YUJ_COMPONENT_REPORT": str(report_path)}
        yield env, record
        try:
            value = json.loads(report_path.read_text())
            if value.get("status") not in {"selected", "ambiguous", "no_candidate"}:
                raise ValueError("invalid component selection status")
            record.update(value)
        except (OSError, ValueError, BudgetExhausted, subprocess.TimeoutExpired) as error:
            record.update(status="unavailable", reason=str(error))
    finally:
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except (OSError, BudgetExhausted, subprocess.TimeoutExpired) as error:
                record['cleanup_error'] = str(error)
