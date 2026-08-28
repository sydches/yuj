"""Exact public runtime-resource contract for source and package builds.

Paths in :data:`ROOT_RUNTIME_FILES` are logical paths below the active Yuj
runtime root.  A source checkout reads them in place; a built distribution
places the same paths below ``scripts/llm_solver/_resources``.  Package-owned
TOML files remain next to the Python modules that consume them.

This module intentionally contains data only so the build backend can load it
without importing the runtime or resolving configuration.
"""
from __future__ import annotations


ROOT_RUNTIME_FILES = (
    "agents/prompts/research.md",
    "agents/research.toml",
    "config.toml",
    "configs/regimes/baselines/plain_long_solve.toml",
    "configs/regimes/treatment.toml",
    "configs/transformations.toml",
    "configs/treatment/hurdle_dictionary.trace_nets.v1.tsv",
    "configs/treatment/medicine_ladder.v1.tsv",
    "configs/treatment/overlays/duplicate_guard.toml",
    "configs/treatment/overlays/intent_gate.toml",
    "configs/treatment/overlays/loop_detect.toml",
    "configs/treatment/overlays/loop_detect_recovery.toml",
    "configs/treatment/overlays/unified_envelope.toml",
    "profiles/_base/denormalize/behavioral.py",
    "profiles/_base/denormalize/rules.toml",
    "profiles/_base/normalize/rules.toml",
    "profiles/_base/profile.toml",
    "profiles/_base/tool_descriptions/minimal/apply_patch.txt",
    "profiles/_base/tool_descriptions/minimal/apply_subagent.txt",
    "profiles/_base/tool_descriptions/minimal/ask_user.txt",
    "profiles/_base/tool_descriptions/minimal/bash.txt",
    "profiles/_base/tool_descriptions/minimal/bash_kill.txt",
    "profiles/_base/tool_descriptions/minimal/bash_poll.txt",
    "profiles/_base/tool_descriptions/minimal/checkpoint.txt",
    "profiles/_base/tool_descriptions/minimal/done.txt",
    "profiles/_base/tool_descriptions/minimal/edit.txt",
    "profiles/_base/tool_descriptions/minimal/exec_cell.txt",
    "profiles/_base/tool_descriptions/minimal/exit_plan_mode.txt",
    "profiles/_base/tool_descriptions/minimal/get_function_details.txt",
    "profiles/_base/tool_descriptions/minimal/glob.txt",
    "profiles/_base/tool_descriptions/minimal/grep.txt",
    "profiles/_base/tool_descriptions/minimal/list_definitions.txt",
    "profiles/_base/tool_descriptions/minimal/list_functions.txt",
    "profiles/_base/tool_descriptions/minimal/load_tools.txt",
    "profiles/_base/tool_descriptions/minimal/lsp.txt",
    "profiles/_base/tool_descriptions/minimal/notebook_edit.txt",
    "profiles/_base/tool_descriptions/minimal/read.txt",
    "profiles/_base/tool_descriptions/minimal/rewind.txt",
    "profiles/_base/tool_descriptions/minimal/run_tests.txt",
    "profiles/_base/tool_descriptions/minimal/structural_edit.txt",
    "profiles/_base/tool_descriptions/minimal/structural_search.txt",
    "profiles/_base/tool_descriptions/minimal/subagent_changes.txt",
    "profiles/_base/tool_descriptions/minimal/task.txt",
    "profiles/_base/tool_descriptions/minimal/terminal_io.txt",
    "profiles/_base/tool_descriptions/minimal/terminal_start.txt",
    "profiles/_base/tool_descriptions/minimal/think.txt",
    "profiles/_base/tool_descriptions/minimal/udiff.txt",
    "profiles/_base/tool_descriptions/minimal/write.txt",
    "profiles/_base/tool_descriptions/minimal/write_todos.txt",
    "profiles/_base/tool_schemas.toml",
    "profiles/devstral2-24b/chat_template_patched.jinja",
    "profiles/devstral2-24b/denormalize/behavioral.py",
    "profiles/devstral2-24b/profile.toml",
    "profiles/nemotron-cascade2-30b/chat_template_thinking_off.jinja",
    "profiles/nemotron-cascade2-30b/denormalize/behavioral.py",
    "profiles/nemotron-cascade2-30b/profile.toml",
    "profiles/qwen3.6-35b-a3b/chat_template_meanderix.jinja",
    "profiles/qwen3.6-35b-a3b/denormalize/behavioral.py",
    "profiles/qwen3.6-35b-a3b/denormalize/rules.toml",
    "profiles/qwen3.6-35b-a3b/normalize/rules.toml",
    "profiles/qwen3.6-35b-a3b/profile.toml",
    "profiles/qwen38-27b/chat_template_thinking_off.jinja",
    "profiles/qwen38-27b/denormalize/behavioral.py",
    "profiles/qwen38-27b/profile.toml",
    "security/patterns.toml",
)


PACKAGE_RUNTIME_FILES = (
    "bash_quirks/forbidden.toml",
    "bash_quirks/redactions.toml",
    "bash_quirks/rewrites.toml",
    "language_quirks/cargo.toml",
    "language_quirks/ctest.toml",
    "language_quirks/generic.toml",
    "language_quirks/go.toml",
    "language_quirks/jest.toml",
    "language_quirks/pytest.toml",
    "tool_quirks/glob.toml",
)


__all__ = ["PACKAGE_RUNTIME_FILES", "ROOT_RUNTIME_FILES"]
