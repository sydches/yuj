from __future__ import annotations

import base64
import hashlib
import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from _config_helpers import make_config
from scripts.llm_assist.__main__ import _persist_session_config_overlay, main
from scripts.llm_assist._anthropic import (
    _CCH_SEED,
    _CLAUDE_CODE_VERSION,
    _SUBSCRIPTION_BILLING_PREFIX,
    _SUBSCRIPTION_BETAS,
    _SUBSCRIPTION_MAX_OUTPUT_TOKENS,
    _SUBSCRIPTION_SYSTEM,
    _SUBSCRIPTION_USER_AGENT,
    _serialize_subscription_body,
    _subscription_billing_header,
    _to_anthropic_payload,
    _xxhash64,
)
from scripts.llm_assist._auth import (
    AccountIneligibleError,
    AuthBinding,
    AuthEndpointMismatchError,
    AuthProtocolError,
    CredentialChangedError,
    CredentialMalformedError,
    CredentialMissingError,
    CredentialRevokedError,
    CredentialSession,
    CredentialStore,
    UnsupportedAuthError,
    browser_sign_in,
    provider_spec,
)
from scripts.llm_assist._codex import _YUJ_CLIENT_VERSION
from scripts.llm_assist.runner import (
    _make_client,
    _protect_auth_environment,
    create_session,
    resolve_served_model,
    run_session,
)
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.harness._loop.trace_schema import emit_trace_event
from scripts.llm_solver.harness.sandbox.env_policy import EnvironmentPolicy


def _jwt(account_id: str = "acct_test") -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return (
        f"{encode({'alg': 'none'})}."
        f"{encode({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}.sig"
    )


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict | None = None,
        *,
        lines: list[str] | None = None,
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self._lines = lines or []
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode: bool = False):
        assert decode_unicode is True
        return iter(self._lines)


class FakeHTTP:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs):
        self.posts.append((url, kwargs))
        return self.responses.pop(0)

    def get(self, url: str, **kwargs):
        self.gets.append((url, kwargs))
        return self.responses.pop(0)


def _codex_sse(
    text: str = "ok",
    *,
    event_type: str = "response.completed",
    status: str | None = "completed",
) -> FakeResponse:
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2},
    }
    if status is not None:
        response["status"] = status
    completed = {"type": event_type, "response": response}
    return FakeResponse(
        lines=[
            f"event: {event_type}",
            f"data: {json.dumps(completed)}",
            "",
            "data: [DONE]",
            "",
        ]
    )


def _message_response(text: str = "ok") -> FakeResponse:
    return FakeResponse(
        payload={
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 2},
        }
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"", "ef46db3751d8e999"),
        (b"a", "d24ec4f1a98c6e5b"),
        (b"abc", "44bc2cf5ad770999"),
    ],
)
def test_xxhash64_matches_published_standard_vectors(
    value: bytes, expected: str,
):
    assert f"{_xxhash64(value):016x}" == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"cch=00000", "a47f7"),
        (b'{"messages":[],"cch=00000","x":1}', "3073d"),
        (
            b"x-anthropic-billing-header: cc_version=2.1.158; "
            b"cc_entrypoint=cli; cch=00000;",
            "f2b0b",
        ),
    ],
)
def test_claude_cch_matches_seeded_low_20_bit_vectors(
    value: bytes, expected: str,
):
    assert f"{_xxhash64(value, _CCH_SEED) & 0xFFFFF:05x}" == expected


def test_claude_billing_fingerprint_uses_first_user_message_and_version(
    monkeypatch: pytest.MonkeyPatch,
):
    version = "9.8.7"
    first_user_message = "0123456789abcdefghijk"
    monkeypatch.setattr(
        "scripts.llm_assist._anthropic._CLAUDE_CODE_VERSION", version
    )
    converted = _to_anthropic_payload({
        "model": "claude-test",
        "messages": [
            {"role": "assistant", "content": "before"},
            {"role": "user", "content": first_user_message},
            {"role": "user", "content": "later user message"},
        ],
        "max_tokens": 9,
    }, subscription=True)

    selected = "".join(first_user_message[index] for index in (4, 7, 20))
    suffix = hashlib.sha256(
        f"59cf53e54c78{selected}{version}".encode()
    ).hexdigest()[:3]
    expected = (
        f"{_SUBSCRIPTION_BILLING_PREFIX} "
        f"cc_version={version}.{suffix}; "
        "cc_entrypoint=claude-desktop; cch=00000;"
    )
    assert converted["system"][0] == {"type": "text", "text": expected}
    assert _subscription_billing_header(first_user_message) == expected


def test_claude_subscription_serialization_is_exact_and_deterministic():
    converted = _to_anthropic_payload({
        "model": "claude-test",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "0123456789abcdefghijk"},
            {"role": "user", "content": "later"},
        ],
        "max_tokens": 9,
    }, subscription=True)
    expected = (
        '{"model":"claude-test","messages":[{"role":"user","content":['
        '{"type":"text","text":"0123456789abcdefghijk"},'
        '{"type":"text","text":"later"}]}],"system":['
        '{"type":"text","text":"x-anthropic-billing-header: '
        'cc_version=2.1.220.8a7; cc_entrypoint=claude-desktop; cch=58e92;"},'
        '{"type":"text","text":"You are a Claude agent, built on '
        'Anthropic\'s Claude Agent SDK."},'
        '{"type":"text","text":"system"}],"max_tokens":9}'
    ).encode()

    assert _serialize_subscription_body(converted) == expected
    assert _serialize_subscription_body(converted) == expected


def test_claude_cch_replacement_is_anchored_to_the_billing_block():
    serialized = _serialize_subscription_body({
        "before": "cch=00000",
        "system": [{
            "type": "text",
            "text": f"{_SUBSCRIPTION_BILLING_PREFIX} cch=00000;",
        }],
        "after": "cch=00000",
    })
    decoded = json.loads(serialized)

    assert decoded["before"] == "cch=00000"
    assert decoded["after"] == "cch=00000"
    assert decoded["system"][0]["text"].startswith(
        f"{_SUBSCRIPTION_BILLING_PREFIX} cch="
    )
    assert decoded["system"][0]["text"] != (
        f"{_SUBSCRIPTION_BILLING_PREFIX} cch=00000;"
    )
    assert serialized.count(b"cch=00000") == 2


def test_provider_credentials_use_atomic_private_provider_files(tmp_path: Path):
    store = CredentialStore(tmp_path / "auth")

    first = store.save_api_key("claude", secret="first-secret")
    path = store.credential_path("claude")
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert store.load("claude", expected_binding=first).secret == "first-secret"

    second = store.save_api_key("claude", secret="second-secret")
    assert second.credential_id != first.credential_id
    assert store.load("claude", expected_binding=second).secret == "second-secret"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(store.root.glob(".claude.*")) == []


def test_logout_removes_only_the_selected_provider(tmp_path: Path):
    store = CredentialStore(tmp_path / "auth")
    claude = store.save_api_key("claude", secret="claude-secret")
    codex = store.save_api_key("codex", secret="codex-secret")
    store.select(claude)

    assert store.logout("claude") is True
    assert store.active_binding() is None
    with pytest.raises(CredentialMissingError, match="claude"):
        store.load("claude")
    assert store.load("codex", expected_binding=codex).secret == "codex-secret"


def test_oauth_protocol_constants_match_current_provider_contracts():
    claude = provider_spec("claude")
    codex = provider_spec("codex")

    assert claude.token_url == "https://api.anthropic.com/v1/oauth/token"
    assert claude.redirect_uri == "http://localhost:54545/callback"
    assert claude.scopes == (
        "org:create_api_key user:profile user:inference "
        "user:sessions:claude_code user:mcp_servers user:file_upload"
    )
    assert claude.token_url != "https://platform.claude.com/v1/oauth/token"
    assert claude.callback_port != 53692

    assert codex.authorize_url == "https://auth.openai.com/oauth/authorize"
    assert codex.token_url == "https://auth.openai.com/oauth/token"
    assert codex.redirect_uri == "http://localhost:1455/auth/callback"
    assert codex.scopes == (
        "openid profile email offline_access "
        "api.connectors.read api.connectors.invoke"
    )
    assert codex.subscription_base_url == "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize(
    ("provider", "token_payload", "expected_host", "expected_port"),
    [
        (
            "claude",
            {
                "access_token": "claude-access",
                "refresh_token": "claude-refresh",
                "expires_in": 3600,
            },
            "claude.ai",
            54545,
        ),
        (
            "codex",
            {
                "access_token": _jwt(),
                "refresh_token": "codex-refresh",
                "expires_in": 3600,
            },
            "auth.openai.com",
            1455,
        ),
    ],
)
def test_browser_sign_in_uses_pkce_and_no_client_secret(
    tmp_path: Path,
    provider: str,
    token_payload: dict,
    expected_host: str,
    expected_port: int,
):
    store = CredentialStore(tmp_path / "auth")
    http = FakeHTTP(FakeResponse(payload=token_payload))
    opened: list[str] = []

    def receive_callback(*, spec, state, authorization_url, open_browser):
        open_browser(authorization_url)
        return SimpleNamespace(code="auth-code", state=state)

    binding = browser_sign_in(
        provider,
        store=store,
        http=http,
        now=lambda: 1000.0,
        random_bytes=lambda size: b"x" * size,
        open_browser=lambda url: opened.append(url) or True,
        receive_callback=receive_callback,
    )

    query = parse_qs(urlparse(opened[0]).query)
    spec = provider_spec(provider)
    assert urlparse(opened[0]).hostname == expected_host
    assert query["code_challenge_method"] == ["S256"]
    assert query["response_type"] == ["code"]
    assert query["scope"] == [spec.scopes]
    assert "client_secret" not in query
    assert urlparse(query["redirect_uri"][0]).port == expected_port
    token_url, request = http.posts[0]
    token_body = request["json" if provider == "claude" else "data"]
    serialized_request = json.dumps(request, default=str)
    assert token_url == spec.token_url
    assert "client_secret" not in serialized_request
    assert token_body["code_verifier"]
    assert query["state"][0] != token_body["code_verifier"]
    assert request["headers"] == {
        "Content-Type": (
            "application/json"
            if provider == "claude"
            else "application/x-www-form-urlencoded"
        )
    }
    if provider == "claude":
        assert token_body["state"] == query["state"][0]
    record = store.load(provider, expected_binding=binding)
    assert record.auth_method == "subscription"
    assert record.access_token == token_payload["access_token"]
    assert record.expires_at == 4600.0
    assert store.active_binding() == binding
    assert stat.S_IMODE(store.selection_path.stat().st_mode) == 0o600


def test_claude_sign_in_retains_preferred_organization_identity(
    tmp_path: Path,
):
    store = CredentialStore(tmp_path / "auth")
    http = FakeHTTP(FakeResponse(payload={
        "access_token": "claude-access",
        "refresh_token": "claude-refresh",
        "expires_in": 3600,
        "account": {"uuid": "account-secondary"},
        "organization_id": "organization-primary",
    }))

    binding = browser_sign_in(
        "claude",
        store=store,
        http=http,
        now=lambda: 1000.0,
        random_bytes=lambda size: b"x" * size,
        open_browser=lambda _url: True,
        receive_callback=lambda **kwargs: SimpleNamespace(
            code="auth-code", state=kwargs["state"]
        ),
    )

    assert store.load("claude", expected_binding=binding).account_id == (
        "organization:organization-primary"
    )


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_expired_subscription_credentials_refresh_atomically(
    tmp_path: Path, provider: str,
):
    store = CredentialStore(tmp_path / "auth")
    access = "old-access" if provider == "claude" else _jwt("old-account")
    binding = store.save_subscription(
        provider,
        access_token=access,
        refresh_token="old-refresh",
        expires_at=10.0,
        account_id="old-account" if provider == "codex" else None,
    )
    refreshed_access = "new-access" if provider == "claude" else _jwt("old-account")
    http = FakeHTTP(FakeResponse(payload={
        "access_token": refreshed_access,
        "refresh_token": "new-refresh",
        "expires_in": 3600,
    }))

    credential = CredentialSession(
        binding,
        store=store,
        http=http,
        now=lambda: 1000.0,
    ).access()

    assert credential.token == refreshed_access
    saved = store.load(provider, expected_binding=binding)
    assert saved.refresh_token == "new-refresh"
    assert saved.expires_at == 4600.0
    assert saved.credential_id == binding.credential_id
    assert stat.S_IMODE(store.credential_path(provider).stat().st_mode) == 0o600
    refresh_url, refresh_request = http.posts[0]
    assert refresh_url == provider_spec(provider).token_url
    refresh_body = refresh_request[
        "json" if provider == "claude" else "data"
    ]
    assert refresh_body == {
        "grant_type": "refresh_token",
        "client_id": provider_spec(provider).client_id,
        "refresh_token": "old-refresh",
    }
    if provider == "claude":
        assert refresh_request["headers"] == {
            "Content-Type": "application/json",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": (
                "anthropic-sdk-typescript/0.94.0 userOAuthProvider"
            ),
        }
    else:
        assert refresh_request["headers"] == {
            "Content-Type": "application/x-www-form-urlencoded"
        }


@pytest.mark.parametrize(
    "identity_fields",
    [
        pytest.param(
            {
                "organization": {"uuid": "organization-primary"},
                "account": {"uuid": "account-changed"},
            },
            id="same-organization",
        ),
        pytest.param({}, id="identity-omitted"),
        pytest.param(
            {"account": {"uuid": "account-only"}},
            id="organization-pinned-account-only-refresh",
        ),
    ],
)
def test_claude_refresh_preserves_compatible_or_stronger_stored_identity(
    tmp_path: Path, identity_fields: dict,
):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_subscription(
        "claude",
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=10.0,
        account_id="organization:organization-primary",
    )
    token_payload = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 3600,
        **identity_fields,
    }

    credential = CredentialSession(
        binding,
        store=store,
        http=FakeHTTP(FakeResponse(payload=token_payload)),
        now=lambda: 1000.0,
    ).access()

    assert credential.account_id == "organization:organization-primary"
    assert store.load("claude", expected_binding=binding).account_id == (
        "organization:organization-primary"
    )


def test_claude_refresh_rejects_changed_organization_identity(tmp_path: Path):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_subscription(
        "claude",
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=10.0,
        account_id="organization:organization-primary",
    )
    session = CredentialSession(
        binding,
        store=store,
        http=FakeHTTP(FakeResponse(payload={
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
            "organization": {"uuid": "organization-changed"},
        })),
        now=lambda: 1000.0,
    )

    with pytest.raises(CredentialChangedError):
        session.access()

    saved = store.load("claude", expected_binding=binding)
    assert saved.account_id == "organization:organization-primary"
    assert saved.access_token == "old-access"
    assert saved.refresh_token == "old-refresh"


@pytest.mark.parametrize(
    ("status_code", "payload", "error_type"),
    [
        (401, {"error": {"code": "invalid_token"}}, CredentialRevokedError),
        (400, {"error": {"code": "refresh_token_reused"}}, CredentialRevokedError),
        (403, {"error": {"code": "account_not_eligible"}}, AccountIneligibleError),
        (404, {"error": {"code": "unsupported_grant_type"}}, UnsupportedAuthError),
    ],
)
def test_refresh_failures_have_specific_safe_errors(
    tmp_path: Path,
    status_code: int,
    payload: dict,
    error_type: type[Exception],
):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_subscription(
        "claude",
        access_token="expired-access",
        refresh_token="private-refresh",
        expires_at=1.0,
    )
    session = CredentialSession(
        binding,
        store=store,
        http=FakeHTTP(FakeResponse(status_code, payload)),
        now=lambda: 1000.0,
    )

    with pytest.raises(error_type) as raised:
        session.access()
    assert "private-refresh" not in str(raised.value)
    assert "expired-access" not in str(raised.value)


def test_missing_malformed_changed_and_unsupported_credentials_are_distinct(
    tmp_path: Path,
):
    store = CredentialStore(tmp_path / "auth")
    with pytest.raises(CredentialMissingError):
        store.load("claude")

    store.root.mkdir(parents=True, exist_ok=True)
    malformed = store.credential_path("claude")
    malformed.write_text("not-json")
    malformed.chmod(0o600)
    with pytest.raises(CredentialMalformedError):
        store.load("claude")

    first = store.save_api_key("claude", secret="one")
    store.save_api_key("claude", secret="two")
    with pytest.raises(CredentialRevokedError, match="changed"):
        store.load("claude", expected_binding=first)

    current = store.active_binding()
    assert current is None
    with pytest.raises(UnsupportedAuthError):
        CredentialSession(
            AuthBinding("claude", "unknown", "id"), store=store
        ).access()


@pytest.mark.parametrize(
    ("provider", "auth_method"),
    [
        ("claude", "api_key"),
        ("claude", "subscription"),
        ("codex", "api_key"),
        ("codex", "subscription"),
    ],
)
def test_one_assistant_request_for_each_provider_auth_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    auth_method: str,
):
    store = CredentialStore(tmp_path / "auth")
    token = f"{provider}-{auth_method}-secret"
    if auth_method == "subscription":
        access = token if provider == "claude" else _jwt()
        binding = store.save_subscription(
            provider,
            access_token=access,
            refresh_token=f"{token}-refresh",
            expires_at=10_000.0,
            account_id="acct_test" if provider == "codex" else None,
        )
    else:
        access = token
        binding = store.save_api_key(provider, secret=token)

    sdk_requests: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.api_key = kwargs["api_key"]
            self.chat = SimpleNamespace(completions=SimpleNamespace(
                create=lambda **payload: sdk_requests.append(payload) or SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )],
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
                    model_dump_json=lambda: "{}",
                )
            ))
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    monkeypatch.setattr("scripts.llm_solver.server.client.openai.OpenAI", FakeOpenAI)
    http = FakeHTTP(
        _message_response() if provider == "claude" else _codex_sse()
    )
    base_url = {
        ("claude", "api_key"): "https://api.anthropic.com/v1",
        ("claude", "subscription"): "https://api.anthropic.com/v1",
        ("codex", "api_key"): "https://api.openai.com/v1",
        ("codex", "subscription"): "https://chatgpt.com/backend-api/codex",
    }[(provider, auth_method)]
    cfg = make_config(
        provider="anthropic" if provider == "claude" else "openai-compatible",
        base_url=base_url,
        api_key="yuj-host-credential",
    )
    client = _make_client(
        cfg,
        profile=None,
        auth_binding=binding,
        auth_store=store,
        http=http,
        now=lambda: 1000.0,
    )
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 10,
    }
    response = client._call_api(payload)

    assert response.choices[0].message.content == "ok"
    if provider == "codex" and auth_method == "api_key":
        assert len(sdk_requests) == 1
        assert http.posts == []
    else:
        assert len(http.posts) == 1
        wire = http.posts[0][1]
        assert token not in json.dumps(wire.get("json", {}))
        if auth_method == "subscription":
            assert wire["headers"].get("Authorization") == f"Bearer {access}"


def test_claude_subscription_request_uses_current_minimal_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request_id = "00000000-0000-4000-8000-000000000043"
    monkeypatch.setattr(
        "scripts.llm_assist._anthropic.uuid",
        SimpleNamespace(uuid4=lambda: request_id),
    )
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_subscription(
        "claude",
        access_token="claude-access",
        refresh_token="claude-refresh",
        expires_at=10_000,
    )
    http = FakeHTTP(FakeResponse(payload={
        "content": [{
            "type": "tool_use",
            "id": "tool-2",
            "name": "_read_file",
            "input": {"path": "README.md"},
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 7, "output_tokens": 2},
    }))
    client = _make_client(
        make_config(
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key="yuj-host-credential",
        ),
        profile=None,
        auth_binding=binding,
        auth_store=store,
        http=http,
        now=lambda: 1000.0,
    )
    client.set_session_id("session-safe-id")

    input_payload = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "use the tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "tool-1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": "contents",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read one file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "max_tokens": 100_000,
    }
    expected_payload = _to_anthropic_payload(
        input_payload, subscription=True
    )
    expected_body = _serialize_subscription_body(expected_payload)
    response = client._call_anthropic_api(input_payload)

    _, request = http.posts[0]
    headers = request["headers"]
    assert _CLAUDE_CODE_VERSION == "2.1.220"
    assert headers == {
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer claude-access",
        "anthropic-beta": _SUBSCRIPTION_BETAS,
        "anthropic-dangerous-direct-browser-access": "true",
        "User-Agent": _SUBSCRIPTION_USER_AGENT,
        "x-app": "cli",
        "x-client-request-id": request_id,
        "X-Claude-Code-Session-Id": "session-safe-id",
    }
    assert _SUBSCRIPTION_BETAS == "claude-code-20250219"
    assert set(request) == {"headers", "data", "timeout"}
    assert request["data"] == expected_body
    assert b"cch=00000" not in request["data"]
    payload = json.loads(request["data"])
    assert len(payload["system"]) == 3
    assert payload["system"][0]["type"] == "text"
    assert payload["system"][1:] == [
        {"type": "text", "text": _SUBSCRIPTION_SYSTEM},
        {"type": "text", "text": "system"},
    ]
    assert payload["system"][0]["text"].startswith(
        f"{_SUBSCRIPTION_BILLING_PREFIX} "
        f"cc_version={_CLAUDE_CODE_VERSION}."
    )
    assert _SUBSCRIPTION_SYSTEM == (
        "You are a Claude agent, built on Anthropic's Claude Agent SDK."
    )
    assert payload["max_tokens"] == _SUBSCRIPTION_MAX_OUTPUT_TOKENS == 64_000
    assert [tool["name"] for tool in payload["tools"]] == [
        "_read_file",
        "web_search",
    ]
    assert payload["messages"][1]["content"][0]["name"] == "_read_file"
    assert response.choices[0].message.tool_calls[0].function.name == "read_file"


def test_claude_api_key_request_remains_plain_json(tmp_path: Path):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_api_key("claude", secret="claude-api-key")
    http = FakeHTTP(_message_response())
    client = _make_client(
        make_config(
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key="yuj-host-credential",
        ),
        profile=None,
        auth_binding=binding,
        auth_store=store,
        http=http,
    )

    client._call_anthropic_api({
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 10,
    })

    _, request = http.posts[0]
    assert set(request) == {"headers", "json", "timeout"}
    assert request["headers"] == {
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "X-Api-Key": "claude-api-key",
    }
    assert request["json"]["system"] == "system"
    assert _SUBSCRIPTION_BILLING_PREFIX not in json.dumps(request["json"])
    for header in (
        "Accept",
        "Authorization",
        "anthropic-beta",
        "anthropic-dangerous-direct-browser-access",
        "User-Agent",
        "x-app",
        "x-client-request-id",
        "X-Claude-Code-Session-Id",
    ):
        assert header not in request["headers"]


def test_codex_subscription_request_pins_yuj_session_envelope(tmp_path: Path):
    store = CredentialStore(tmp_path / "auth")
    access_token = _jwt()
    binding = store.save_subscription(
        "codex",
        access_token=access_token,
        refresh_token="codex-refresh",
        expires_at=10_000,
        account_id="acct_test",
    )
    http = FakeHTTP(_codex_sse())
    client = _make_client(
        make_config(
            provider="openai-compatible",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="yuj-host-credential",
        ),
        profile=None,
        auth_binding=binding,
        auth_store=store,
        http=http,
        now=lambda: 1000.0,
    )
    client.set_session_id("session-safe-id")

    client._call_responses_api({
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 10,
    })

    url, request = http.posts[0]
    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert _YUJ_CLIENT_VERSION == "0.1.0"
    assert request["headers"] == {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": "acct_test",
        "originator": "yuj",
        "version": _YUJ_CLIENT_VERSION,
        "User-Agent": "yuj",
        "OpenAI-Beta": "responses=experimental",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "conversation_id": "session-safe-id",
        "session_id": "session-safe-id",
        "x-client-request-id": "session-safe-id",
    }
    assert request["json"]["prompt_cache_key"] == "session-safe-id"
    assert client._headers()["version"] == _YUJ_CLIENT_VERSION


@pytest.mark.parametrize(
    ("event_type", "status", "expected_status", "expected_finish_reason"),
    [
        pytest.param(
            "response.completed", None, "completed", "stop", id="completed"
        ),
        pytest.param("response.done", None, "completed", "stop", id="done"),
        pytest.param(
            "response.incomplete",
            None,
            "incomplete",
            "length",
            id="incomplete",
        ),
        pytest.param(
            "response.incomplete",
            "completed",
            "completed",
            "stop",
            id="incomplete-event-explicit-completed-status",
        ),
        pytest.param(
            "response.done",
            "incomplete",
            "incomplete",
            "length",
            id="done-event-explicit-incomplete-status",
        ),
    ],
)
def test_codex_terminal_events_preserve_or_default_response_status(
    tmp_path: Path,
    event_type: str,
    status: str | None,
    expected_status: str,
    expected_finish_reason: str,
):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_subscription(
        "codex",
        access_token=_jwt(),
        refresh_token="codex-refresh",
        expires_at=10_000,
        account_id="acct_test",
    )
    client = _make_client(
        make_config(
            provider="openai-compatible",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="yuj-host-credential",
        ),
        profile=None,
        auth_binding=binding,
        auth_store=store,
        http=FakeHTTP(_codex_sse(event_type=event_type, status=status)),
        now=lambda: 1000.0,
    )

    response = client._call_responses_api({
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    })

    assert json.loads(response.model_dump_json())["status"] == expected_status
    assert response.choices[0].finish_reason == expected_finish_reason


@pytest.mark.parametrize(
    ("lines", "safe_message"),
    [
        pytest.param(
            ["data: {provider-private-detail", ""],
            "subscription response was malformed",
            id="malformed-json",
        ),
        pytest.param(
            [
                'data: {"type":"response.output_text.delta",'
                '"delta":"provider-private-detail"}',
                "",
                "data: [DONE]",
                "",
            ],
            "subscription response did not complete",
            id="missing-terminal-event",
        ),
        pytest.param(
            [
                'data: {"type":"response.failed",'
                '"response":{"error":"provider-private-detail"}}',
                "",
            ],
            "subscription response failed",
            id="response-failed",
        ),
        pytest.param(
            [
                'data: {"type":"error",'
                '"error":{"message":"provider-private-detail"}}',
                "",
            ],
            "subscription response failed",
            id="error",
        ),
    ],
)
def test_codex_terminal_protocol_errors_are_secret_safe(
    tmp_path: Path, lines: list[str], safe_message: str,
):
    store = CredentialStore(tmp_path / "auth")
    access_token = _jwt()
    binding = store.save_subscription(
        "codex",
        access_token=access_token,
        refresh_token="codex-refresh",
        expires_at=10_000,
        account_id="acct_test",
    )
    client = _make_client(
        make_config(
            provider="openai-compatible",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="yuj-host-credential",
        ),
        profile=None,
        auth_binding=binding,
        auth_store=store,
        http=FakeHTTP(FakeResponse(lines=lines)),
        now=lambda: 1000.0,
    )

    with pytest.raises(AuthProtocolError, match=safe_message) as raised:
        client._call_responses_api({
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        })

    rendered = str(raised.value)
    assert "provider-private-detail" not in rendered
    assert access_token not in rendered


def test_subscription_transport_rejects_endpoint_change_before_contact(
    tmp_path: Path,
):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_subscription(
        "codex",
        access_token=_jwt(),
        refresh_token="refresh",
        expires_at=10_000,
        account_id="acct_test",
    )
    http = FakeHTTP(_codex_sse())
    cfg = make_config(
        provider="openai-compatible",
        base_url="https://api.openai.com/v1",
        api_key="yuj-host-credential",
    )

    with pytest.raises(AuthEndpointMismatchError):
        _make_client(
            cfg,
            profile=None,
            auth_binding=binding,
            auth_store=store,
            http=http,
        )
    assert http.posts == []


def test_managed_model_resolution_never_substitutes_first_listed_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_api_key("claude", secret="managed-key")
    cfg = make_config(
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key="yuj-host-credential",
        model="configured-model",
        profile_name="configured-model",
    )

    class FakeClient:
        def health_check(self):
            return ["different-first-model"]

    monkeypatch.setattr(
        "scripts.llm_assist.runner.load_config", lambda **_kwargs: cfg
    )
    monkeypatch.setattr(
        "scripts.llm_assist.runner._make_client",
        lambda *_args, **_kwargs: FakeClient(),
    )

    model, served = resolve_served_model(
        [],
        config_overrides={
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
        },
        auth_binding=binding,
        auth_store=store,
    )

    assert model == "configured-model"
    assert served == ["different-first-model"]


@pytest.mark.parametrize(
    ("provider", "auth_method"),
    [
        ("claude", "api_key"),
        ("claude", "subscription"),
        ("codex", "api_key"),
        ("codex", "subscription"),
    ],
)
def test_managed_auth_environment_is_never_available_to_tools(
    tmp_path: Path, provider: str, auth_method: str,
):
    store = CredentialStore(tmp_path / "auth")
    if auth_method == "api_key":
        binding = store.save_api_key(
            provider, environment="MANAGED_PROVIDER_KEY"
        )
    else:
        binding = store.save_subscription(
            provider,
            access_token=(
                "subscription-access" if provider == "claude" else _jwt()
            ),
            refresh_token="subscription-refresh",
            expires_at=10_000,
            account_id="acct_test" if provider == "codex" else None,
        )
    sandbox_set = {
        "YUJ_AUTH_HOME": "/target/.credentials",
        "SAFE": "ok",
    }
    sandbox_filters = {"YUJ_*": "include", "SAFE": "include"}
    host_environment = {
        "YUJ_AUTH_HOME": "/host/.credentials",
        "SAFE": "host-ok",
    }
    if auth_method == "api_key":
        sandbox_set["MANAGED_PROVIDER_KEY"] = "explicit-secret"
        sandbox_filters["MANAGED_*"] = "include"
        host_environment["MANAGED_PROVIDER_KEY"] = "host-secret"
    cfg = make_config(
        model_fallback_chain={"main": [{"profile": "other"}]},
        sandbox_env_inherit="all",
        sandbox_env_set=sandbox_set,
        sandbox_env_filters=sandbox_filters,
        sandbox_env_ignore_default_excludes=True,
    )

    protected = _protect_auth_environment(cfg, binding, store=store)
    effective = EnvironmentPolicy(
        inherit=protected.sandbox_env_inherit,
        set=protected.sandbox_env_set,
        filters=protected.sandbox_env_filters,
        ignore_default_excludes=protected.sandbox_env_ignore_default_excludes,
    ).resolve(host_environment)

    assert effective == {"SAFE": "ok"}
    assert protected.model_fallback_chain == {}


def test_managed_api_key_rejection_is_specific_and_transcript_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_api_key("codex", secret="never-in-transcript")

    class Rejected(Exception):
        status_code = 401

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(
                create=lambda **_payload: (_ for _ in ()).throw(Rejected())
            ))
            self.models = SimpleNamespace(data=[])

    monkeypatch.setattr("scripts.llm_solver.server.client.openai.OpenAI", FakeOpenAI)
    client = _make_client(
        make_config(
            provider="openai-compatible",
            base_url="https://api.openai.com/v1",
            api_key="yuj-host-credential",
        ),
        profile=None,
        auth_binding=binding,
        auth_store=store,
    )
    transcript = tmp_path / "transcript.log"
    client.set_transcript(transcript)

    with pytest.raises(CredentialRevokedError):
        client._call_api({
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        })
    client.close_transcript()
    assert "never-in-transcript" not in transcript.read_text()


def test_credential_store_rejects_target_repository_location(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    store = CredentialStore(target / ".credentials")

    with pytest.raises(CredentialMalformedError, match="target repository"):
        store.require_outside(target)


def test_setup_accepts_normal_xdg_config_and_auth_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    xdg_home = tmp_path / "xdg"
    config_path = xdg_home / "yuj" / "config.local.toml"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(config_path))
    monkeypatch.delenv("YUJ_AUTH_HOME", raising=False)

    assert main([
        "setup",
        "--provider", "claude",
        "--auth", "api-key",
        "--model", "claude-sonnet-4-5",
        "--sandbox", "none",
        "--api-key", "xdg-layout-key",
        "--force",
    ]) == 0

    auth_root = xdg_home / "yuj" / "auth"
    assert config_path.is_file()
    assert (auth_root / "claude.json").is_file()
    assert stat.S_IMODE(auth_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((auth_root / "claude.json").stat().st_mode) == 0o600
    assert "xdg-layout-key" not in config_path.read_text()


def test_login_rejects_credentials_inside_current_target_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository = tmp_path / "target-repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    nested = repository / "nested"
    nested.mkdir()
    auth_root = repository / ".credentials"
    monkeypatch.chdir(nested)
    monkeypatch.setenv("YUJ_AUTH_HOME", str(auth_root))

    with pytest.raises(SystemExit, match="target repository"):
        main([
            "login",
            "--provider", "codex",
            "--auth", "api-key",
            "--api-key", "must-not-be-written",
        ])

    assert not auth_root.exists()


def test_runner_rechecks_repository_boundary_before_resume_credential_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository = tmp_path / "target-repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    target = repository / "nested"
    target.mkdir()
    auth_root = repository / ".credentials"
    auth_store = CredentialStore(auth_root)
    binding = auth_store.save_api_key("claude", secret="inside-target")
    assist_store = SessionStore(tmp_path / "assist")
    record = create_session(
        assist_store,
        cwd=target,
        model="claude-sonnet-4-5",
        prompt_text="work",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
        auth_binding=binding,
    )
    monkeypatch.setenv("YUJ_AUTH_HOME", str(auth_root))
    monkeypatch.setattr(
        "scripts.llm_assist.runner.load_config",
        lambda **_kwargs: make_config(runtime_mode="assistant", max_sessions=1),
    )
    monkeypatch.setattr(
        "scripts.llm_assist.runner._resolve_session_worktree",
        lambda _store, saved, **_kwargs: (None, saved),
    )

    with pytest.raises(CredentialMalformedError, match="target repository"):
        run_session(assist_store, record, resume=True)


def test_setup_stores_managed_api_key_outside_config_and_reports_safe_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    config_path = tmp_path / "checkout" / "config.local.toml"
    config_path.parent.mkdir()
    auth_home = tmp_path / "host-auth"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(config_path))
    monkeypatch.setenv("YUJ_AUTH_HOME", str(auth_home))

    assert main([
        "setup",
        "--provider", "claude",
        "--auth", "api-key",
        "--model", "claude-sonnet-4-5",
        "--sandbox", "none",
        "--api-key", "setup-private-key",
        "--force",
    ]) == 0

    output = capsys.readouterr().out
    config_text = config_path.read_text()
    assert "setup-private-key" not in config_text
    assert 'api_key = "yuj-host-credential"' in config_text
    assert "setup-private-key" not in output
    assert "provider: claude" in output
    assert "authentication: api_key" in output

    assert main(["auth-status"]) == 0
    status_output = capsys.readouterr().out
    assert "active_provider: claude" in status_output
    assert "authentication: api_key" in status_output
    assert "setup-private-key" not in status_output
    assert stat.S_IMODE((auth_home / "claude.json").stat().st_mode) == 0o600


def test_session_status_identifies_pinned_provider_and_auth_method(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    store = SessionStore(tmp_path / "assist")
    record = store.create_session(
        cwd=tmp_path / "work",
        model="gpt-5.4",
        prompt_text="work",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
        provider="codex",
        auth_method="subscription",
        credential_id="credential-id-not-rendered",
    )
    monkeypatch.setattr("scripts.llm_assist.__main__.SessionStore", lambda: store)

    assert main(["status", record.session_id]) == 0
    output = capsys.readouterr().out
    assert "provider: codex" in output
    assert "authentication: subscription" in output
    assert "credential-id-not-rendered" not in output


def test_credentials_never_enter_session_artifacts_or_repository_files(
    tmp_path: Path,
):
    auth_store = CredentialStore(tmp_path / "host-auth")
    subscription = auth_store.save_subscription(
        "claude",
        access_token="artifact-forbidden-access",
        refresh_token="artifact-forbidden-refresh",
        expires_at=10_000,
    )
    api_key = auth_store.save_api_key(
        "codex", secret="artifact-forbidden-api-key"
    )
    assist_store = SessionStore(tmp_path / "assist")
    work = tmp_path / "target-repository"
    work.mkdir()
    (work / "README.md").write_text("target file\n")

    records = []
    for binding, model, overrides in (
        (
            subscription,
            "claude-sonnet-4-5",
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com/v1",
                "api_key": "yuj-host-credential",
            },
        ),
        (
            api_key,
            "gpt-5.4",
            {
                "provider": "openai-compatible",
                "base_url": "https://api.openai.com/v1",
                "api_key": "yuj-host-credential",
            },
        ),
    ):
        record = create_session(
            assist_store,
            cwd=work,
            model=model,
            prompt_text="work",
            prompt_source="inline",
            context_mode="full",
            system_prompt_path=None,
            config_paths=[],
            auth_binding=binding,
        )
        record = _persist_session_config_overlay(
            assist_store,
            record,
            base_config_paths=[],
            transport_overrides=overrides,
        )
        trace_path = record.artifact_path / ".trace.jsonl"
        with trace_path.open("a") as trace_file:
            emit_trace_event(
                trace_file,
                "session_start",
                session_number=1,
                thinking_level="off",
                sandbox_backend="bwrap",
                container_runtime="",
                container_image_digest="",
                ignore_file_hash="",
                sandbox_env_names=["SAFE"],
                edit_format="patch",
                repo_map_tokens=0,
            )
        assert (record.artifact_path / "prompt.txt").is_file()
        assert (record.artifact_path / "session.json").is_file()
        assert (record.artifact_path / "provider.toml").is_file()
        assert trace_path.is_file()
        assert assist_store.get_session(record.session_id).credential_id == (
            binding.credential_id
        )
        records.append(record)

    public_state = "\n".join(
        path.read_text(errors="ignore")
        for root in (work, *(record.artifact_path for record in records))
        for path in root.rglob("*")
        if path.is_file()
    )
    for forbidden in (
        "artifact-forbidden-access",
        "artifact-forbidden-refresh",
        "artifact-forbidden-api-key",
        subscription.credential_id,
        api_key.credential_id,
    ):
        assert forbidden not in public_state


def test_provider_runtime_rejection_is_specific_and_does_not_retry_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = CredentialStore(tmp_path / "auth")
    binding = store.save_subscription(
        "claude",
        access_token="revoked-access",
        refresh_token="refresh",
        expires_at=10_000,
    )
    http = FakeHTTP(FakeResponse(401, {"error": {"type": "authentication_error"}}))

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setattr("scripts.llm_solver.server.client.openai.OpenAI", FakeOpenAI)
    client = _make_client(
        make_config(
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key="yuj-host-credential",
        ),
        profile=None,
        auth_binding=binding,
        auth_store=store,
        http=http,
        now=lambda: 1000.0,
    )

    with pytest.raises(CredentialRevokedError) as raised:
        client._call_anthropic_api({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        })
    assert "revoked-access" not in str(raised.value)
    assert len(http.posts) == 1
