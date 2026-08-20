"""Tests for the XML-shape sink-pointer emitted by the harness when a
bash tool result exceeds the sink threshold.

The composition happens inside Session._filter_bash_output. Rather
than standing up a full Session, these tests exercise the emission
shape via direct format() on the config template plus a smaller
composition check using the config-supplied slice sizes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config


def _attrs(s: str) -> dict:
    m = re.search(r'<tool_result_meta (.*?)/>', s)
    assert m is not None, f"no tool_result_meta tag in: {s!r}"
    return dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))


class TestSinkPointerTemplate:

    def test_default_is_xml_self_closing_tag(self):
        cfg = make_config()
        rendered = cfg.sink_pointer.format(
            path=".tool_output/s1_0001_t4.log",
            chars=42314,
            lines=1200,
        )
        assert rendered.startswith("<tool_result_meta ")
        assert rendered.endswith("/>")
        attrs = _attrs(rendered)
        assert attrs["truncated"] == "true"
        assert attrs["original_bytes"] == "42314"
        assert attrs["original_lines"] == "1200"
        assert attrs["full_path"] == ".tool_output/s1_0001_t4.log"

    def test_no_prose_in_default(self):
        """The default must not contain the legacy bracketed prose —
        if a downstream user override re-introduces it, that is their
        call, but the shipped default is XML-shape."""
        cfg = make_config()
        assert "raw output:" not in cfg.sink_pointer
        assert "[" not in cfg.sink_pointer or cfg.sink_pointer.startswith("<")


class TestSinkBodyMarker:

    def test_body_marker_default_mentions_full_path_attribute(self):
        cfg = make_config()
        assert "full_path" in cfg.sink_body_marker


class TestSinkByteConfig:

    def test_defaults(self):
        cfg = make_config()
        assert cfg.sink_head_bytes == 1000
        assert cfg.sink_tail_bytes == 1000

    def test_override(self):
        cfg = make_config(sink_head_bytes=500, sink_tail_bytes=250)
        assert cfg.sink_head_bytes == 500
        assert cfg.sink_tail_bytes == 250


class TestCompositionShape:

    def test_composition_matches_loop_emission(self):
        """Mirror the exact composition done in loop.py::_filter_bash_output
        so the test breaks if the shape drifts."""
        cfg = make_config()
        raw = "A" * 5000
        head = raw[:cfg.sink_head_bytes]
        tail = raw[-cfg.sink_tail_bytes:]
        pointer = cfg.sink_pointer.format(
            path=".tool_output/test.log",
            chars=len(raw),
            lines=raw.count("\n") + 1,
        )
        result = f"{head}\n{cfg.sink_body_marker}\n{tail}\n{pointer}"
        assert result.endswith("/>")
        # The structured marker is parseable.
        attrs = _attrs(result)
        assert attrs["original_bytes"] == "5000"
        # Head and tail are verbatim in the composition.
        assert result.startswith(head)
        assert cfg.sink_body_marker in result
