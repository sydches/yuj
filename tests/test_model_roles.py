"""Focused tests for auxiliary model-role resolution and accounting."""
from pathlib import Path

import pytest

from scripts.llm_solver.harness._loop.model_roles import (
    FALLBACK_REVERT_POLICIES,
    ModelFallbackController,
    ModelRoleError,
    ModelRoleResolver,
    ModelRoleRouter,
    ModelTarget,
    RoleTokenLedger,
    check_context_window,
    normalize_fallback_revert,
    parse_fallback_target,
    parse_model_target,
    validate_fallback_chains,
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


def test_role_keyed_fallback_chain_resolves_profiles_and_contexts_at_startup():
    resolver = _resolver({"weak": "weak"})
    controller = ModelFallbackController(
        resolver,
        fallback_chain={
            "main": ["weak@http://127.0.0.1:8182/v1"],
            "weak": [
                {
                    "profile": "editor",
                    "endpoint": "http://127.0.0.1:8282/v1",
                    "model": "editor-fallback-model",
                }
            ],
        },
        fallback_revert="never",
    )

    main_transition = controller.advance("main", reason="transient_exhausted")
    assert main_transition is not None
    assert main_transition.to_resolution.profile.name == "weak"
    assert main_transition.to_resolution.target.context_size == 4096
    assert main_transition.trace_fields() == {
        "role": "main",
        "from": "_base@http://127.0.0.1:8080/v1",
        "to": "weak@http://127.0.0.1:8182/v1",
        "reason": "transient_exhausted",
        "from_profile": "_base",
        "to_profile": "weak",
        "from_model": "main-served-model",
        "to_model": "weak",
        "from_context_size": 8192,
        "to_context_size": 4096,
    }

    weak_transition = controller.advance("weak", reason="server_oom")
    assert weak_transition is not None
    assert weak_transition.to_resolution.profile.name == "editor"
    assert weak_transition.to_resolution.target.model == "editor-fallback-model"
    assert weak_transition.to_resolution.target.context_size == 16384


def test_failing_fake_client_switches_only_after_retry_budget_is_exhausted():
    resolver = _resolver()
    controller = ModelFallbackController(
        resolver,
        fallback_chain=["weak@http://127.0.0.1:8182/v1"],
    )

    class FakeClient:
        def __init__(self, resolution):
            self.resolution = resolution
            self.calls = 0

        def chat(self):
            self.calls += 1
            if self.resolution.target.profile_name == "_base":
                raise ConnectionError("server unavailable")
            return "fallback response"

    router = ModelRoleRouter(resolver, FakeClient, controller)
    primary = router.client_for("main")
    max_transient_retries = 1

    for _ in range(max_transient_retries + 1):
        with pytest.raises(ConnectionError):
            primary.client.chat()

    switched = router.switch_after_retry_exhaustion(
        "main",
        reason="transient_exhausted",
    )

    assert switched is not None
    assert primary.client.calls == max_transient_retries + 1
    assert switched.routed_client.resolution.profile.name == "weak"
    assert switched.routed_client.resolution.target.context_size == 4096
    assert switched.routed_client.client.chat() == "fallback response"
    assert router.client_for("main").client is switched.routed_client.client


def test_fallback_context_window_check_uses_new_profile_limit():
    resolver = _resolver()
    controller = ModelFallbackController(
        resolver,
        fallback_chain=["weak@http://127.0.0.1:8182/v1"],
    )
    transition = controller.advance("main", reason="context_overflow")
    assert transition is not None

    fits = check_context_window(3000, transition.to_resolution, 0.8)
    over = check_context_window(3300, transition.to_resolution, 0.8)

    assert fits.context_size == 4096
    assert fits.prompt_token_limit == 3276
    assert fits.fits is True
    assert over.fits is False


def test_next_session_reverts_but_never_policy_keeps_fallback_active():
    resolver = _resolver()
    chain = ["weak@http://127.0.0.1:8182/v1"]
    next_session = ModelFallbackController(
        resolver,
        fallback_chain=chain,
        fallback_revert="next_session",
    )
    never = ModelFallbackController(
        resolver,
        fallback_chain=chain,
        fallback_revert="never",
    )
    assert next_session.advance("main", reason="transient_exhausted") is not None
    assert never.advance("main", reason="transient_exhausted") is not None

    assert next_session.begin_session() is True
    assert next_session.current("main").target.profile_name == "_base"
    assert never.begin_session() is False
    assert never.current("main").target.profile_name == "weak"


def test_fallback_metrics_flag_treatment_change_for_study_filters():
    controller = ModelFallbackController(
        _resolver(),
        fallback_chain=["weak@http://127.0.0.1:8182/v1"],
    )
    assert controller.metrics_fields() == {
        "model_fallback_used": False,
        "model_fallback_count": 0,
        "model_fallback_roles": [],
        "model_fallback_active_targets": {},
    }

    assert controller.advance("main", reason="transient_exhausted") is not None

    assert controller.metrics_fields() == {
        "model_fallback_used": True,
        "model_fallback_count": 1,
        "model_fallback_roles": ["main"],
        "model_fallback_active_targets": {
            "main": "weak@http://127.0.0.1:8182/v1"
        },
    }
    assert controller.advance("main", reason="transient_exhausted") is None


def test_fallback_profiles_are_validated_eagerly():
    with pytest.raises(ModelRoleError, match="profile 'broken'.*failed validation"):
        ModelFallbackController(
            _resolver(),
            fallback_chain=["broken@http://127.0.0.1:8182/v1"],
        )


def test_fallback_config_accepts_bare_main_list_and_role_table():
    main = validate_fallback_chains(["weak@http://127.0.0.1:8182/v1"])
    roles = validate_fallback_chains(
        {
            "weak": [
                {
                    "profile": "editor",
                    "endpoint": "http://127.0.0.1:8282/v1",
                }
            ]
        }
    )

    assert tuple(main) == ("main",)
    assert main["main"][0].profile_name == "weak"
    assert roles["weak"][0].base_url == "http://127.0.0.1:8282/v1"


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("weak", "<profile>@<endpoint>"),
        ("@http://localhost:8080/v1", "<profile>@<endpoint>"),
        ({"profile": "weak"}, "endpoint is required"),
        ("weak@localhost:8080/v1", "absolute http"),
    ],
)
def test_invalid_fallback_targets_fail_closed(value, match):
    with pytest.raises(ModelRoleError, match=match):
        parse_fallback_target(value, field="models.fallback_chain.main[0]")


def test_duplicate_fallback_targets_and_raw_error_text_are_rejected():
    duplicate = [
        "weak@http://127.0.0.1:8182/v1",
        "weak@http://127.0.0.1:8182/v1",
    ]
    with pytest.raises(ModelRoleError, match="repeats target"):
        ModelFallbackController(_resolver(), fallback_chain=duplicate)

    controller = ModelFallbackController(
        _resolver(),
        fallback_chain=["weak@http://127.0.0.1:8182/v1"],
    )
    with pytest.raises(ModelRoleError, match="reason must contain only"):
        controller.advance("main", reason="server failed with private detail")


def test_fallback_revert_policy_is_exact():
    assert FALLBACK_REVERT_POLICIES == ("never", "next_session")
    assert normalize_fallback_revert("NEXT_SESSION") == "next_session"
    with pytest.raises(ModelRoleError, match="must be one of"):
        normalize_fallback_revert("cooldown")
