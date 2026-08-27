from pathlib import Path

import pytest

from tests.distribution_contract import (
    PACKAGE_RUNTIME_FILES,
    ROOT_RUNTIME_FILES,
    SDIST_GENERATED_FILES,
    WHEEL_METADATA_FILES,
    expected_sdist_payload,
    expected_wheel_payload,
    validate_sdist_members,
    validate_wheel_members,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_resource_manifest_is_exact_and_present():
    assert len(ROOT_RUNTIME_FILES) == 64
    assert len(PACKAGE_RUNTIME_FILES) == 10
    assert tuple(sorted(ROOT_RUNTIME_FILES)) == ROOT_RUNTIME_FILES
    assert tuple(sorted(PACKAGE_RUNTIME_FILES)) == PACKAGE_RUNTIME_FILES
    assert all((REPOSITORY / path).is_file() for path in ROOT_RUNTIME_FILES)
    assert all(
        (REPOSITORY / "scripts" / "llm_solver" / path).is_file()
        for path in PACKAGE_RUNTIME_FILES
    )
    description_root = REPOSITORY / "profiles" / "_base" / "tool_descriptions"
    description_files = {
        path.relative_to(REPOSITORY).as_posix()
        for path in description_root.glob("*/*.txt")
    }
    declared_descriptions = {
        path for path in ROOT_RUNTIME_FILES
        if path.startswith("profiles/_base/tool_descriptions/")
    }
    assert declared_descriptions == description_files


def test_wheel_member_allowlist_rejects_missing_and_private_paths():
    dist_info = "yuj-0.1.0.dist-info"
    members = expected_wheel_payload() | {
        f"{dist_info}/{path}" for path in WHEEL_METADATA_FILES
    } | {
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/licenses/LICENSES/Apache-2.0.txt",
        f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md",
    }
    counts = validate_wheel_members(members)
    assert counts["root_resources"] == 64

    missing = set(members)
    missing.remove("scripts/llm_solver/_resources/config.toml")
    with pytest.raises(AssertionError, match="missing=.*config.toml"):
        validate_wheel_members(missing)

    with pytest.raises(AssertionError, match="paper"):
        validate_wheel_members(members | {"paper/private-notes.md"})


def test_sdist_member_allowlist_rejects_tests_and_internal_records():
    top = "yuj-0.1.0"
    payload = expected_sdist_payload() | SDIST_GENERATED_FILES
    members = {f"{top}/{path}" for path in payload}
    assert validate_sdist_members(members)["root_resources"] == 64

    with pytest.raises(AssertionError, match="tests"):
        validate_sdist_members(members | {f"{top}/tests/test_private.py"})
    with pytest.raises(AssertionError, match="internal"):
        validate_sdist_members(members | {f"{top}/.internal/campaign.md"})
