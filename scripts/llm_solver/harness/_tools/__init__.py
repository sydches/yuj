"""Per-tool implementations for harness/tools.py.

Each module here owns one tool function plus its private helpers. The
public surface (`bash`, `read`, …, `dispatch`, `ToolRegistry`) is
re-exported from `harness/tools.py` so callers see no change.
"""
