"""FullTranscript context mode registration.

The class stays in ``harness.context`` for backward imports. This module makes
``full`` discoverable through the same strategy-module path as every other
context mode.
"""
from __future__ import annotations

from ..context import FullTranscript
from ._metadata import ContextModeMetadata


CONTEXT_MODE = "full"
CONTEXT_CLASS = FullTranscript
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=0,
    message_shape="append-only transcript",
    state_source="append_only_messages",
    source_type="append_log",
    normal_prompt_sources=("in_memory_append_log",),
    section_order=("append_only_chronological_messages",),
    section_labels={
        "append_only_chronological_messages": "<chronological transcript>",
    },
    file_freshness="append_only",
    injection_support="verbatim",
)
