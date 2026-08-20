"""Public LLM hurdle detector API.

Implementation is split into small modules to keep the detector package
easy to check while preserving this import path.
"""
from __future__ import annotations

from .llm_detector_core import (
    EVENT_TYPE,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    SIMPLE_PROMPT_VERSION,
    AtlasFamily,
    LLMDetectorPacket,
    LLMDetectorVerdict,
    append_detector_log,
    build_detector_log_row,
    build_detector_packet,
    load_atlas_families,
    parse_detector_verdict,
    render_detector_messages,
    run_detector_call,
)
from .llm_detector_runtime import maybe_run_llm_hurdle_detector

__all__ = [
    "EVENT_TYPE",
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "SIMPLE_PROMPT_VERSION",
    "AtlasFamily",
    "LLMDetectorPacket",
    "LLMDetectorVerdict",
    "append_detector_log",
    "build_detector_log_row",
    "build_detector_packet",
    "load_atlas_families",
    "maybe_run_llm_hurdle_detector",
    "parse_detector_verdict",
    "render_detector_messages",
    "run_detector_call",
]
