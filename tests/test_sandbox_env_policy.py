from __future__ import annotations

import subprocess

import pytest

from scripts.llm_solver.harness.sandbox.env_policy import (
    CORE_ENVIRONMENT_NAMES,
    EnvironmentPolicy,
    EnvironmentPolicyError,
    build_bash_argv,
    build_bwrap_env_argv,
    build_subprocess_env,
)


HOST_ENV = {
    "PATH": "/host/bin",
    "HOME": "/host/home",
    "LANG": "en_GB.UTF-8",
    "TERM": "xterm-256color",
    "SAFE_NAME": "safe",
    "OPENAI_API_KEY": "host-key",
    "SERVICE_SECRET": "host-secret",
    "SESSION_TOKEN": "host-token",
}


def test_core_inherits_only_the_declared_core_names() -> None:
    effective = EnvironmentPolicy(inherit="core").resolve(HOST_ENV)

    assert tuple(effective) == tuple(sorted(CORE_ENVIRONMENT_NAMES))
    assert effective == {
        "HOME": "/host/home",
        "LANG": "en_GB.UTF-8",
        "PATH": "/host/bin",
        "TERM": "xterm-256color",
    }


def test_default_excludes_drop_secret_like_names_case_insensitively() -> None:
    host = dict(HOST_ENV, lowercase_token="also-secret", MONKEY="contains-key")

    effective = EnvironmentPolicy(inherit="all").resolve(host)

    assert effective == {
        "HOME": "/host/home",
        "LANG": "en_GB.UTF-8",
        "PATH": "/host/bin",
        "SAFE_NAME": "safe",
        "TERM": "xterm-256color",
    }


def test_resolution_order_is_excludes_then_set_then_include_allowlist() -> None:
    policy = EnvironmentPolicy(
        inherit="all",
        set={
            "PATH": "/fixed/bin",
            "OPENAI_API_KEY": "explicit-test-value",
            "ADDED": "fixed",
        },
        filters={
            "safe_*": "exclude",
            "path": "include",
            "openai_???_key": "include",
            "added": "include",
        },
    )

    effective = policy.resolve(HOST_ENV)

    # The ambient key is removed by the default exclusion, then the explicit
    # fixed value restores it.  The case-insensitive include list is last.
    assert effective == {
        "ADDED": "fixed",
        "OPENAI_API_KEY": "explicit-test-value",
        "PATH": "/fixed/bin",
    }


def test_none_starts_empty_and_fixed_values_are_still_available() -> None:
    policy = EnvironmentPolicy(inherit="none", set={"ONLY": "value"})
    assert policy.resolve(HOST_ENV) == {"ONLY": "value"}


def test_ignore_default_excludes_is_an_explicit_escape_hatch() -> None:
    policy = EnvironmentPolicy(inherit="all", ignore_default_excludes=True)
    effective = policy.resolve(HOST_ENV)
    assert effective["OPENAI_API_KEY"] == "host-key"
    assert effective["SERVICE_SECRET"] == "host-secret"
    assert effective["SESSION_TOKEN"] == "host-token"


def test_policy_copies_mutable_input_tables() -> None:
    fixed = {"A": "one"}
    filters = {"A": "include"}
    policy = EnvironmentPolicy(inherit="none", set=fixed, filters=filters)

    fixed["A"] = "changed"
    filters["B"] = "include"

    assert policy.resolve({}) == {"A": "one"}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"inherit": "host"}, "inherit"),
        ({"set": {"BAD=NAME": "x"}}, "contains"),
        ({"set": {"A": 1}}, "must be a string"),
        ({"filters": {"*": "keep"}}, "include.*exclude"),
        ({"ignore_default_excludes": "false"}, "must be a boolean"),
        ({"unknown": True}, "unknown field"),
    ],
)
def test_invalid_policy_is_rejected_loudly(raw: dict, message: str) -> None:
    with pytest.raises(EnvironmentPolicyError, match=message):
        EnvironmentPolicy.from_mapping(raw)


def test_bwrap_arguments_clear_ambient_env_and_are_name_sorted() -> None:
    argv = build_bwrap_env_argv({"Z": "last", "A": "first"})
    assert argv == [
        "--clearenv",
        "--setenv", "A", "first",
        "--setenv", "Z", "last",
    ]


def test_subprocess_mapping_runs_with_only_the_effective_names(tmp_path) -> None:
    effective = EnvironmentPolicy(
        inherit="none", set={"VISIBLE": "yes"},
    ).resolve(HOST_ENV)

    result = subprocess.run(
        ["/usr/bin/env"],
        cwd=tmp_path,
        env=build_subprocess_env(effective),
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == ["VISIBLE=yes"]


def test_login_shell_semantics_are_explicit() -> None:
    assert build_bash_argv("echo ok") == [
        "bash", "--noprofile", "--norc", "-o", "pipefail", "-c", "echo ok",
    ]
    assert build_bash_argv("echo ok", allow_login_shell=True) == [
        "bash", "--login", "-o", "pipefail", "-c", "echo ok",
    ]
    assert build_bash_argv(None) == [
        "bash", "--noprofile", "--norc", "-s",
    ]
