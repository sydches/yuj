"""Select a component only after pytest applies its effective collection rules.

Loaded only for automatic component checks. Names suggest an association;
the actual collected items establish membership, not semantic coverage.
Sources and candidates use the task root; native item IDs retain pytest's root.
"""
import json
import os
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    yield
    paths = []
    matched_source = None
    # Preserve the selector's source order, choosing the first unique match.
    for raw_source, hints in json.loads(os.environ["YUJ_COMPONENT_NAMES"]).items():
        source = Path(raw_source)
        names = set(hints)
        # Use pytest's effective patterns, including -c/-o and project plugins.
        for pattern in config.getini("python_files"):
            if pattern.count("*") == 1 and not any(c in pattern for c in "?[]/"):
                names.add(pattern.replace("*", source.stem))
        matches = sorted({item.path for item in items if item.path.name in names})
        if matches:
            paths = matches
            matched_source = str(source)
        if len(matches) == 1:
            break
    selected = paths[0] if len(paths) == 1 else None
    kept = [item for item in items if selected is not None and item.path == selected]
    deselected = [item for item in items if selected is None or item.path != selected]
    items[:] = kept
    config.hook.pytest_deselected(items=deselected)
    root = os.environ["YUJ_COMPONENT_ROOT"]
    record = {
        "status": "selected" if selected else "ambiguous" if paths else "no_candidate",
        "source": matched_source,
        "association": "naming_hint",
        "path_root": root,
        "candidates": [os.path.relpath(path, root) for path in paths],
        "collection_root": str(config.rootpath),
        "config": str(config.inipath) if config.inipath else None,
        "python_files": config.getini("python_files"),
        "items": [item.nodeid for item in kept],
    }
    Path(os.environ["YUJ_COMPONENT_REPORT"]).write_text(json.dumps(record))
