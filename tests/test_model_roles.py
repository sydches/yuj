"""Focused tests for auxiliary model-role resolution and accounting."""
from pathlib import Path

import pytest

from scripts.llm_solver.harness._loop.model_roles import (
    ModelRoleError,
    ModelRoleResolver,
    ModelRoleRouter,
    ModelTarget,
    RoleTokenLedger,
    parse_model_target,
    validate_role_specs,
)
from scripts.llm_solver.server.profile_loader import load_profile


FIXTURE_PROFILES = Path(__file__).parent / "fixtures" / "model_role_profiles"


def _profile_loader(name: str):
    return load_profile(name, FIXTURE_PROFILES)


def _main_target() -> ModelTarget:
    return ModelTarget(
        profile_name="_base",
        model="main-served-model",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        context_size=8192,
    )


def _resolver(role_specs=None) -> ModelRoleResolver:
    return ModelRoleResolver(
        main_target=_main_target(),
        role_specs=role_specs,
        profile_loader=_profile_loader,
    )


def test_roles_resolve_profile_endpoint_and_model_overrides_at_startup():
    resolver = _resolver(
        {
            "weak": "weak",
            "editor": {
                "profile": "editor",
                "endpoint": "http://127.0.0.1:8181/v1/",
                "model": "editor-served-model",
                "context_size": 12288,
            },
        }
    )

    weak = resolver.resolve("weak")
    assert weak.profile.name == "weak"
    assert weak.target.model == "weak"
    assert weak.target.base_url == "http://127.0.0.1:8080/v1"
    assert weak.target.api_key == "local"
    assert weak.target.context_size == 4096

    editor = resolver.resolve("editor")
    assert editor.profile.name == "editor"
    assert editor.target.model == "editor-served-model"
    assert editor.target.base_url == "http://127.0.0.1:8181/v1"
    assert editor.target.context_size == 12288
    assert resolver.configured_roles == ("weak", "editor")


def test_unset_side_role_falls_back_to_main_and_trace_records_actual_role():
    resolver = _resolver({"weak": "weak", "editor": ""})

    handoff = resolver.resolve("handoff")

    assert handoff.target == resolver.main.target
    assert handoff.profile is resolver.main.profile
    assert handoff.uses_main_fallback is True
    assert handoff.trace_fields() == {
        "role": "main",
        "requested_role": "handoff",
        "role_fallback": "main",
    }


def test_router_returns_resolved_client_and_reuses_main_for_unset_roles():
    built = []

    def factory(resolution):
        client = object()
        built.append((resolution.effective_role, resolution.target, client))
        return client

    router = ModelRoleRouter(_resolver({"weak": "weak"}), factory)

    main = router.client_for("main")
    fallback = router.client_for("classification")
    weak = router.client_for("weak")

    assert fallback.client is main.client
    assert fallback.resolution.uses_main_fallback is True
    assert weak.client is not main.client
    assert [role for role, _, _ in built] == ["main", "weak"]


def test_all_configured_role_profiles_are_validated_during_resolver_startup():
    with pytest.raises(ModelRoleError, match="role 'editor'.*failed validation"):
        _resolver({"weak": "weak", "editor": "broken"})


def test_role_token_ledger_splits_usage_by_effective_role():
    resolver = _resolver({"weak": "weak"})
    ledger = RoleTokenLedger()

    ledger.record(
        resolver.resolve("weak"),
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=60,
    )
    # An unset editor used the main target, so its tokens belong to main.
    ledger.record(
        resolver.resolve("editor"),
        prompt_tokens=80,
        completion_tokens=10,
        cached_tokens=0,
    )
    ledger.record("weak", prompt_tokens=50, completion_tokens=5, cached_tokens=25)

    assert ledger.metrics_fields() == {
        "tokens_by_role": {
            "main": {
                "requests": 1,
                "prompt_tokens": 80,
                "completion_tokens": 10,
                "cached_tokens": 0,
                "total_tokens": 90,
            },
            "weak": {
                "requests": 2,
                "prompt_tokens": 150,
                "completion_tokens": 25,
                "cached_tokens": 85,
                "total_tokens": 175,
            },
        }
    }


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"profile": "weak", "endpoint": "localhost:8080/v1"}, "absolute http"),
        (
            {"profile": "weak", "endpoint": "http://user:secret@localhost/v1"},
            "must not embed credentials",
        ),
        ({"profile": "weak", "context_size": 0}, "positive integer"),
        ({"profile": "weak", "unknown": True}, "unknown fields"),
        ({"profile": "weak", "profile_name": "other"}, "cannot set both"),
        (42, "profile string or table"),
    ],
)
def test_invalid_role_targets_fail_closed(value, match):
    with pytest.raises(ModelRoleError, match=match):
        parse_model_target(value, field="models.roles.weak")


def test_main_is_not_a_side_role_and_role_names_cannot_be_paths():
    with pytest.raises(ModelRoleError, match="reserved"):
        validate_role_specs({"main": "weak"})
    with pytest.raises(ModelRoleError, match="must contain only"):
        validate_role_specs({"../weak": "weak"})


def test_token_ledger_rejects_inconsistent_cached_count():
    with pytest.raises(ModelRoleError, match="cannot exceed"):
        RoleTokenLedger().record(
            "weak",
            prompt_tokens=10,
            completion_tokens=1,
            cached_tokens=11,
        )
