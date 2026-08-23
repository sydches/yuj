from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.llm_solver.harness.sandbox.ignore_policy import (
    IgnorePolicyError,
    IgnoredPathError,
    load_ignore_policy,
    parse_ignore_lines,
)


def _policy(tmp_path: Path, text: str, *, names=(".yujignore",)):
    (tmp_path / names[0]).write_text(text)
    return load_ignore_policy(tmp_path, file_names=names)


def test_parser_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "\n# comment\n*.tmp\n\\#literal\n")

    assert policy.is_ignored("work.tmp")
    assert policy.is_ignored("#literal")
    assert not policy.is_ignored("comment")


def test_negation_uses_last_matching_rule_within_one_file(tmp_path: Path) -> None:
    policy = _policy(
        tmp_path,
        "*.log\n!important.log\nbuild/\n!build/keep.txt\n",
    )

    assert policy.is_ignored("nested/debug.log")
    assert not policy.is_ignored("nested/important.log")
    assert policy.is_ignored("build/drop.txt")
    assert not policy.is_ignored("build/keep.txt")


def test_trailing_spaces_are_trimmed_unless_escaped(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "trimmed   \nliteral\\  \n")

    assert policy.is_ignored("trimmed")
    assert not policy.is_ignored("trimmed   ")
    assert policy.is_ignored("literal ")


def test_leading_slash_anchors_only_at_the_task_root(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "/root.txt\nloose.txt\n")

    assert policy.is_ignored("root.txt")
    assert not policy.is_ignored("nested/root.txt")
    assert policy.is_ignored("loose.txt")
    assert policy.is_ignored("nested/loose.txt")


def test_trailing_slash_matches_directories_and_descendants_only(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "cache/\n")

    assert policy.is_ignored("cache", is_dir=True)
    assert policy.is_ignored("cache/data.json", is_dir=False)
    assert policy.is_ignored("nested/cache/data.json", is_dir=False)
    assert not policy.is_ignored("cache", is_dir=False)


def test_star_question_class_and_double_star_globs(tmp_path: Path) -> None:
    policy = _policy(
        tmp_path,
        "build/*.o\nfile?.txt\nimage[0-2].png\nsrc/**/generated.py\n",
    )

    assert policy.is_ignored("build/main.o")
    assert not policy.is_ignored("build/nested/main.o")
    assert policy.is_ignored("nested/file7.txt")
    assert not policy.is_ignored("file77.txt")
    assert policy.is_ignored("image1.png")
    assert not policy.is_ignored("image8.png")
    assert policy.is_ignored("src/generated.py")
    assert policy.is_ignored("src/a/b/generated.py")


def test_earlier_ignore_file_wins_when_both_files_match(tmp_path: Path) -> None:
    (tmp_path / ".firstignore").write_text("!keep.txt\n")
    (tmp_path / ".secondignore").write_text("keep.txt\n*.tmp\n")
    policy = load_ignore_policy(
        tmp_path, file_names=(".firstignore", ".secondignore"),
    )

    assert not policy.is_ignored("keep.txt")
    assert policy.is_ignored("scratch.tmp")


def test_single_source_hash_is_the_sha256_of_exact_file_bytes(tmp_path: Path) -> None:
    raw = b"# exact bytes\n*.secret\n"
    (tmp_path / ".yujignore").write_bytes(raw)

    policy = load_ignore_policy(tmp_path)

    assert policy.aggregate_hash == hashlib.sha256(raw).hexdigest()
    assert policy.trace_fields() == {
        "ignore_file_hash": hashlib.sha256(raw).hexdigest(),
        "ignore_file_names": [".yujignore"],
    }


def test_disabled_policy_does_not_read_or_apply_the_file(tmp_path: Path) -> None:
    (tmp_path / ".yujignore").write_bytes(b"\xff")

    policy = load_ignore_policy(tmp_path, enabled=False)

    assert not policy.is_ignored("anything")
    assert policy.trace_fields() == {
        "ignore_file_hash": None,
        "ignore_file_names": [],
    }


def test_require_visible_uses_file_not_found_semantics(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "private.txt\n")

    with pytest.raises(IgnoredPathError) as raised:
        policy.require_visible("private.txt", is_dir=False)

    assert isinstance(raised.value, FileNotFoundError)
    policy.require_visible("public.txt", is_dir=False)


def test_filter_paths_preserves_order_and_removes_hidden_entries(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "*.key\n")

    assert policy.filter_paths(("b.txt", "a.key", "a.txt")) == (
        "b.txt", "a.txt",
    )


def test_existing_ignored_paths_feed_sandbox_masks(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "data.bin").write_bytes(b"data")
    (tmp_path / "private.txt").write_text("private")
    (tmp_path / "public.txt").write_text("public")
    policy = _policy(tmp_path, "cache/\nprivate.txt\n")

    hidden = policy.existing_ignored_paths()

    assert str(tmp_path / "cache") in hidden
    assert str(tmp_path / "cache" / "data.bin") in hidden
    assert str(tmp_path / "private.txt") in hidden
    assert str(tmp_path / "public.txt") not in hidden


def test_paths_outside_root_and_parent_patterns_fail_closed(tmp_path: Path) -> None:
    policy = _policy(tmp_path, "*.tmp\n")

    with pytest.raises(IgnorePolicyError, match="escapes task root"):
        policy.is_ignored(tmp_path.parent / "outside.tmp")
    with pytest.raises(IgnorePolicyError, match="not allowed"):
        parse_ignore_lines(["../outside"], source_name="bad.ignore")


def test_invalid_utf8_and_external_symlink_sources_fail_closed(tmp_path: Path) -> None:
    (tmp_path / ".yujignore").write_bytes(b"\xff")
    with pytest.raises(IgnorePolicyError, match="valid UTF-8"):
        load_ignore_policy(tmp_path)

    outside = tmp_path.parent / "outside-ignore"
    outside.write_text("secret\n")
    (tmp_path / ".yujignore").unlink()
    (tmp_path / ".yujignore").symlink_to(outside)
    with pytest.raises(IgnorePolicyError, match="outside the task root"):
        load_ignore_policy(tmp_path)


@pytest.mark.parametrize(
    "names",
    [".yujignore", ("../outside",), ("/absolute",), ("dir/",)],
)
def test_invalid_ignore_file_name_shape_is_rejected(tmp_path: Path, names) -> None:
    with pytest.raises(IgnorePolicyError):
        load_ignore_policy(tmp_path, file_names=names)
