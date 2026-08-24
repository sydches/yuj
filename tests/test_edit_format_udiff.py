"""Acceptance coverage for the unified-diff edit dialect."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver.harness._tools.edit import edit
from llm_solver.harness._tools.udiff import udiff_tool


def _cfg():
    return make_config(tools_edit_format="udiff")


def test_udiff_applies_exact_hunk(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("def value():\n    return 1\n")
    patch = """\
--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 2
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("OK: applied unified diff")
    assert target.read_text() == "def value():\n    return 2\n"
    assert result.applied_operations == (("update", "sample.py"),)


def test_udiff_fuzzy_context_matches_unique_whitespace_drift(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("def value():\n\treturn 1\n")
    patch = """\
--- a/sample.py
+++ b/sample.py
@@ -20,2 +20,2 @@
 def value():
-  return 1
+\treturn 2
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert "fuzzy=whitespace_normalized" in result
    assert target.read_text() == "def value():\n\treturn 2\n"


def test_udiff_ambiguous_fuzzy_match_surfaces_edit_candidates(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    original = "if ready:\n    run()\nif ready:\n    run()\n"
    target.write_text(original)
    patch = """\
--- a/sample.py
+++ b/sample.py
@@ -50,2 +50,2 @@
 if ready:
-  run()
+    stop()
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("ERROR: udiff hunk_ambiguous:")
    assert '<candidates total="2" cause_hint="whitespace_drift" path="sample.py">' in result
    assert '<candidate rank="1" strategy="whitespace_normalized"' in result
    assert target.read_text() == original


def test_udiff_and_exact_edit_share_candidate_protocol(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("if ready:\n    run()\nif ready:\n    run()\n")
    exact_result = edit(
        "sample.py",
        "if ready:\n  run()",
        "if ready:\n    stop()",
        cwd=str(tmp_path),
        cfg=make_config(edit_fuzzy_cascade_enabled=False),
    )
    diff_result = udiff_tool(
        "--- a/sample.py\n+++ b/sample.py\n@@ -50,2 +50,2 @@\n"
        " if ready:\n-  run()\n+    stop()\n",
        cwd=str(tmp_path),
        cfg=_cfg(),
    )

    exact_xml = ET.fromstring(exact_result[exact_result.index("<candidates"):])
    diff_xml = ET.fromstring(diff_result[diff_result.index("<candidates"):])
    assert exact_xml.tag == diff_xml.tag == "candidates"
    assert set(exact_xml.attrib) == set(diff_xml.attrib) == {
        "total", "cause_hint", "path",
    }
    assert set(exact_xml[0].attrib) == set(diff_xml[0].attrib) == {
        "rank", "strategy", "similarity", "line",
    }


def test_udiff_preverifies_all_files_before_any_write(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("old one\n")
    second.write_text("old two\n")
    patch = """\
--- a/first.txt
+++ b/first.txt
@@ -1 +1 @@
-old one
+new one
--- a/second.txt
+++ b/second.txt
@@ -1 +1 @@
-not present
+new two
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("ERROR: udiff hunk_not_found:")
    assert first.read_text() == "old one\n"
    assert second.read_text() == "old two\n"


def test_udiff_preverifies_add_parent_before_any_write(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    blocked_parent = tmp_path / "blocked"
    first.write_text("old\n")
    blocked_parent.write_text("not a directory\n")
    patch = """\
--- a/first.txt
+++ b/first.txt
@@ -1 +1 @@
-old
+new
--- /dev/null
+++ b/blocked/child.txt
@@ -0,0 +1 @@
+child
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("ERROR: udiff parent_not_directory:")
    assert first.read_text() == "old\n"
    assert blocked_parent.read_text() == "not a directory\n"


def test_udiff_adds_and_deletes_files(tmp_path: Path) -> None:
    old = tmp_path / "old.txt"
    old.write_text("remove me\n")
    patch = """\
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+line one
+line two
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-remove me
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("OK: applied unified diff")
    assert (tmp_path / "new.txt").read_text() == "line one\nline two\n"
    assert not old.exists()
    assert result.applied_operations == (
        ("add", "new.txt"),
        ("delete", "old.txt"),
    )


def test_udiff_inserts_without_removing_existing_lines(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("one\nthree\n")
    patch = """\
--- a/sample.txt
+++ b/sample.txt
@@ -1,0 +2 @@
+two
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("OK: applied unified diff")
    assert target.read_text() == "one\ntwo\nthree\n"


def test_udiff_rejects_partial_delete_file_patch(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("keep\nremove\n")
    patch = """\
--- a/sample.txt
+++ /dev/null
@@ -2 +0,0 @@
-remove
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("ERROR: udiff delete_not_empty:")
    assert target.read_text() == "keep\nremove\n"


def test_udiff_suffix_deletion_preserves_surviving_newline(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("keep\nremove\n")
    patch = """\
--- a/sample.txt
+++ b/sample.txt
@@ -2 +1,0 @@
-remove
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("OK: applied unified diff")
    assert target.read_bytes() == b"keep\n"


def test_udiff_rejects_path_escape_without_writing(tmp_path: Path) -> None:
    patch = """\
--- /dev/null
+++ ../outside.txt
@@ -0,0 +1 @@
+nope
"""

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("ERROR: udiff path_outside_cwd:")
    assert not (tmp_path.parent / "outside.txt").exists()


def test_udiff_preserves_explicit_no_newline_marker(tmp_path: Path) -> None:
    patch = (
        "--- /dev/null\n"
        "+++ b/no-newline.txt\n"
        "@@ -0,0 +1 @@\n"
        "+last line\n"
        "\\ No newline at end of file\n"
    )

    result = udiff_tool(patch, cwd=str(tmp_path), cfg=_cfg())

    assert result.startswith("OK: applied unified diff")
    assert (tmp_path / "no-newline.txt").read_bytes() == b"last line"
