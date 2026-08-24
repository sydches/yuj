"""Provider-scoped credentials for the Yuj assistant shell.

Credential values stay in this module and in host-side credential files.  The
measurement configuration sees only its existing provider, endpoint, and API
key fields.
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import http.server
import json
import os
import secrets
import stat
import subprocess
import tempfile
import time
import uuid
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import parse_qs, urlencode, urlparse

import requests


_CREDENTIAL_VERSION = 1
_SUPPORTED_PROVIDERS = frozenset({"claude", "codex"})
_SUPPORTED_METHODS = frozenset({"api_key", "subscription"})
_REFRESH_SKEW_SECONDS = 300.0
_CLAUDE_REFRESH_BETA = "oauth-2025-04-20"
_CLAUDE_REFRESH_USER_AGENT = (
    "anthropic-sdk-typescript/0.94.0 userOAuthProvider"
)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    core_provider: str
    api_key_base_url: str
    subscription_base_url: str
    client_id: str
    authorize_url: str
    token_url: str
    callback_port: int
    callback_path: str
    scopes: str
    token_encoding: str

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self.callback_port}{self.callback_path}"


_PROVIDER_SPECS = {
    "claude": ProviderSpec(
        name="claude",
        core_provider="anthropic",
        api_key_base_url="https://api.anthropic.com/v1",
        subscription_base_url="https://api.anthropic.com/v1",
        client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        authorize_url="https://claude.ai/oauth/authorize",
        token_url="https://api.anthropic.com/v1/oauth/token",
        callback_port=54545,
        callback_path="/callback",
        scopes=(
            "org:create_api_key user:profile user:inference "
            "user:sessions:claude_code user:mcp_servers user:file_upload"
        ),
        token_encoding="json",
    ),
    "codex": ProviderSpec(
        name="codex",
        core_provider="openai-compatible",
        api_key_base_url="https://api.openai.com/v1",
        subscription_base_url="https://chatgpt.com/backend-api/codex",
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        authorize_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        callback_port=1455,
        callback_path="/auth/callback",
        scopes=(
            "openid profile email offline_access "
            "api.connectors.read api.connectors.invoke"
        ),
        token_encoding="form",
    ),
}


class ProviderAuthError(RuntimeError):
    """Base class for safe, provider-specific authentication failures."""

    code = "provider_auth_error"

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider} authentication: {message}")


class CredentialMissingError(ProviderAuthError):
    code = "credential_missing"


class CredentialMalformedError(ProviderAuthError):
    code = "credential_malformed"


class CredentialExpiredError(ProviderAuthError):
    code = "credential_expired"


class CredentialRevokedError(ProviderAuthError):
    code = "credential_revoked"


class CredentialChangedError(CredentialRevokedError):
    code = "credential_changed"


class AccountIneligibleError(ProviderAuthError):
    code = "account_ineligible"


class UnsupportedAuthError(ProviderAuthError):
    code = "auth_unsupported"


class AuthProtocolError(ProviderAuthError):
    code = "auth_protocol_error"


class AuthEndpointMismatchError(ProviderAuthError):
    code = "auth_endpoint_mismatch"


class ProviderRequestError(ProviderAuthError):
    code = "provider_request_error"


@dataclass(frozen=True)
class AuthBinding:
    """Non-secret identity pinned to one assistant session."""

    provider: str
    auth_method: str
    credential_id: str


@dataclass(frozen=True)
class CredentialRecord:
    provider: str
    auth_method: str
    credential_id: str
    secret: str | None = None
    environment: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: float | None = None
    account_id: str | None = None

    @property
    def binding(self) -> AuthBinding:
        return AuthBinding(self.provider, self.auth_method, self.credential_id)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "version": _CREDENTIAL_VERSION,
            "provider": self.provider,
            "auth_method": self.auth_method,
            "credential_id": self.credential_id,
        }
        for name in (
            "secret",
            "environment",
            "access_token",
            "refresh_token",
            "expires_at",
            "account_id",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


@dataclass(frozen=True)
class AccessCredential:
    token: str
    account_id: str | None = None


@dataclass(frozen=True)
class BrowserCallback:
    code: str
    state: str


def auth_home() -> Path:
    """Return the host-only credential directory."""
    configured = os.environ.get("YUJ_AUTH_HOME")
    if configured:
        return _absolute_path(Path(configured).expanduser())
    configured_home = os.environ.get("XDG_CONFIG_HOME")
    base = (
        Path(configured_home).expanduser()
        if configured_home
        else Path.home() / ".config"
    )
    return _absolute_path(base / "yuj" / "auth")


def provider_spec(provider: str) -> ProviderSpec:
    try:
        return _PROVIDER_SPECS[provider]
    except KeyError as exc:
        raise UnsupportedAuthError(provider, "unsupported provider") from exc


class CredentialStore:
    """Atomic provider-scoped credential and active-selection store."""

    def __init__(self, root: Path | None = None):
        self.root = (
            _absolute_path(Path(root).expanduser())
            if root is not None
            else auth_home()
        )

    def credential_path(self, provider: str) -> Path:
        provider_spec(provider)
        return self.root / f"{provider}.json"

    @property
    def selection_path(self) -> Path:
        return self.root / "active.json"

    def require_outside(self, *targets: Path) -> None:
        """Reject credential storage inside any target repository."""
        protected = [Path(target).expanduser().resolve() for target in targets]
        candidates = [
            self.selection_path,
            *(self.credential_path(provider) for provider in _SUPPORTED_PROVIDERS),
        ]
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            for target in protected:
                if resolved == target or resolved.is_relative_to(target):
                    raise CredentialMalformedError(
                        "host",
                        "credential files cannot be stored in a target repository",
                    )

    def require_outside_target(self, target: Path) -> None:
        """Guard a known run target, using its Git root when available."""
        path = Path(target).expanduser().resolve()
        self.require_outside(_known_repository_root(path) or path)

    def require_outside_current_repository(self) -> None:
        """Guard setup/login when the current directory belongs to a repo."""
        repository = _known_repository_root(Path.cwd())
        if repository is not None:
            self.require_outside(repository)

    def save_api_key(
        self,
        provider: str,
        *,
        secret: str | None = None,
        environment: str | None = None,
    ) -> AuthBinding:
        provider_spec(provider)
        if bool(secret) == bool(environment):
            raise CredentialMalformedError(
                provider, "provide exactly one API key value or environment name"
            )
        record = CredentialRecord(
            provider=provider,
            auth_method="api_key",
            credential_id=uuid.uuid4().hex,
            secret=secret,
            environment=environment,
        )
        with self.locked(provider):
            self._write_record(record)
        return record.binding

    def save_subscription(
        self,
        provider: str,
        *,
        access_token: str,
        refresh_token: str,
        expires_at: float,
        account_id: str | None = None,
        credential_id: str | None = None,
    ) -> AuthBinding:
        provider_spec(provider)
        record = CredentialRecord(
            provider=provider,
            auth_method="subscription",
            credential_id=credential_id or uuid.uuid4().hex,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=float(expires_at),
            account_id=account_id,
        )
        self._validate_record(record)
        with self.locked(provider):
            self._write_record(record)
        return record.binding

    def replace_subscription(
        self,
        binding: AuthBinding,
        *,
        access_token: str,
        refresh_token: str,
        expires_at: float,
        account_id: str | None,
    ) -> CredentialRecord:
        record = CredentialRecord(
            provider=binding.provider,
            auth_method="subscription",
            credential_id=binding.credential_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=float(expires_at),
            account_id=account_id,
        )
        self._validate_record(record)
        self._write_record(record)
        return record

    def load(
        self,
        provider: str,
        *,
        expected_binding: AuthBinding | None = None,
    ) -> CredentialRecord:
        path = self.credential_path(provider)
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise CredentialMissingError(
                provider,
                f"credential is missing; run `yuj login --provider {provider}`",
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
        ):
            raise CredentialMalformedError(
                provider, "credential file must be a regular user-only file"
            )
        try:
            raw = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CredentialMalformedError(
                provider, "credential file is not valid JSON"
            ) from exc
        record = self._record_from_dict(provider, raw)
        if expected_binding is not None and record.binding != expected_binding:
            raise CredentialChangedError(
                provider,
                "credential changed after the session was created; start a new session",
            )
        return record

    def select(self, binding: AuthBinding) -> None:
        if binding.provider not in _SUPPORTED_PROVIDERS:
            raise UnsupportedAuthError(binding.provider, "unsupported provider")
        self._write_atomic(
            self.selection_path,
            {
                "version": _CREDENTIAL_VERSION,
                "provider": binding.provider,
                "auth_method": binding.auth_method,
                "credential_id": binding.credential_id,
            },
        )

    def active_binding(self) -> AuthBinding | None:
        path = self.selection_path
        if not path.exists():
            return None
        try:
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & 0o077
            ):
                raise ValueError("unsafe permissions")
            raw = json.loads(path.read_text())
            if raw.get("version") != _CREDENTIAL_VERSION:
                raise ValueError("unsupported version")
            binding = AuthBinding(
                provider=str(raw["provider"]),
                auth_method=str(raw["auth_method"]),
                credential_id=str(raw["credential_id"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CredentialMalformedError(
                "active", "active authentication selection is malformed"
            ) from exc
        provider_spec(binding.provider)
        if binding.auth_method not in _SUPPORTED_METHODS or not binding.credential_id:
            raise CredentialMalformedError(
                binding.provider, "active authentication selection is malformed"
            )
        return binding

    def clear_selection(self) -> None:
        try:
            self.selection_path.unlink()
        except FileNotFoundError:
            return

    def logout(self, provider: str) -> bool:
        provider_spec(provider)
        removed = False
        with self.locked(provider):
            try:
                self.credential_path(provider).unlink()
                removed = True
            except FileNotFoundError:
                pass
            active = self.active_binding()
            if active is not None and active.provider == provider:
                self.clear_selection()
        return removed

    @contextlib.contextmanager
    def locked(self, provider: str) -> Iterator[None]:
        provider_spec(provider)
        self._ensure_root()
        lock_path = self.root / f"{provider}.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise CredentialMalformedError(
                provider, "credential lock file is unsafe"
            ) from exc
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _record_from_dict(self, provider: str, raw: object) -> CredentialRecord:
        if not isinstance(raw, dict):
            raise CredentialMalformedError(provider, "credential must be a JSON object")
        if raw.get("version") != _CREDENTIAL_VERSION:
            raise UnsupportedAuthError(provider, "credential version is unsupported")
        try:
            record = CredentialRecord(
                provider=str(raw["provider"]),
                auth_method=str(raw["auth_method"]),
                credential_id=str(raw["credential_id"]),
                secret=_optional_string(raw.get("secret")),
                environment=_optional_string(raw.get("environment")),
                access_token=_optional_string(raw.get("access_token")),
                refresh_token=_optional_string(raw.get("refresh_token")),
                expires_at=_optional_number(raw.get("expires_at")),
                account_id=_optional_string(raw.get("account_id")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CredentialMalformedError(
                provider, "credential fields are malformed"
            ) from exc
        if record.provider != provider:
            raise CredentialMalformedError(
                provider, "credential provider does not match file"
            )
        self._validate_record(record)
        return record

    def _validate_record(self, record: CredentialRecord) -> None:
        provider_spec(record.provider)
        if record.auth_method not in _SUPPORTED_METHODS:
            raise UnsupportedAuthError(
                record.provider, "authentication method is unsupported"
            )
        if not record.credential_id:
            raise CredentialMalformedError(record.provider, "credential ID is missing")
        if record.auth_method == "api_key":
            if bool(record.secret) == bool(record.environment):
                raise CredentialMalformedError(
                    record.provider,
                    "API key credential must contain exactly one key source",
                )
            return
        if not record.access_token or not record.refresh_token:
            raise CredentialMalformedError(
                record.provider, "subscription token fields are missing"
            )
        if record.expires_at is None or record.expires_at <= 0:
            raise CredentialMalformedError(
                record.provider, "subscription expiration is malformed"
            )
        if record.provider == "codex":
            if not record.account_id:
                raise AccountIneligibleError(
                    record.provider,
                    "credential has no eligible ChatGPT account",
                )
            token_account = _codex_account_id(record.access_token or "")
            if token_account != record.account_id:
                raise CredentialMalformedError(
                    record.provider,
                    "credential account identity is inconsistent",
                )

    def _write_record(self, record: CredentialRecord) -> None:
        self._write_atomic(self.credential_path(record.provider), record.to_dict())

    def _ensure_root(self) -> None:
        if self.root.is_symlink():
            raise CredentialMalformedError(
                "host", "credential path must not be a symbolic link"
            )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise CredentialMalformedError("host", "credential path is not a directory")
        self.root.chmod(0o700)

    def _write_atomic(self, path: Path, payload: dict[str, object]) -> None:
        self._ensure_root()
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=self.root
        )
        temp_path = Path(temp_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                descriptor = -1
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            path.chmod(0o600)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


class CredentialSession:
    """Resolve one pinned credential and refresh it without fallback."""

    def __init__(
        self,
        binding: AuthBinding,
        *,
        store: CredentialStore | None = None,
        http=None,
        now: Callable[[], float] | None = None,
    ):
        self.binding = binding
        self.store = store or CredentialStore()
        self.http = http or requests.Session()
        self.now = now or time.time

    def access(self) -> AccessCredential:
        if self.binding.auth_method not in _SUPPORTED_METHODS:
            raise UnsupportedAuthError(
                self.binding.provider, "authentication method is unsupported"
            )
        with self.store.locked(self.binding.provider):
            record = self.store.load(
                self.binding.provider, expected_binding=self.binding
            )
            if record.auth_method == "api_key":
                token = record.secret
                if record.environment:
                    token = os.environ.get(record.environment)
                    if not token:
                        raise CredentialMissingError(
                            record.provider,
                            f"environment variable {record.environment} is missing",
                        )
                if not token:
                    raise CredentialMissingError(record.provider, "API key is missing")
                return AccessCredential(token)

            assert record.expires_at is not None
            if record.expires_at <= self.now() + _REFRESH_SKEW_SECONDS:
                record = self._refresh(record)
            return AccessCredential(
                record.access_token or "", account_id=record.account_id
            )

    def environment_name(self) -> str | None:
        record = self.store.load(
            self.binding.provider, expected_binding=self.binding
        )
        return record.environment

    def _refresh(self, record: CredentialRecord) -> CredentialRecord:
        if not record.refresh_token:
            raise CredentialExpiredError(
                record.provider, "credential expired and has no refresh token"
            )
        spec = provider_spec(record.provider)
        body = {
            "grant_type": "refresh_token",
            "client_id": spec.client_id,
            "refresh_token": record.refresh_token,
        }
        request: dict[str, object] = {
            "headers": _token_headers(spec, refresh=True),
            "timeout": (10, 30),
        }
        request["json" if spec.token_encoding == "json" else "data"] = body
        try:
            response = self.http.post(spec.token_url, **request)
        except Exception as exc:
            raise AuthProtocolError(
                record.provider, "token refresh transport failed"
            ) from exc
        if not _response_ok(response):
            _raise_auth_response(record.provider, response, operation="refresh")
        token = _parse_token_response(
            record.provider,
            response,
            now=self.now(),
            previous_refresh_token=record.refresh_token,
        )
        account_id = record.account_id
        if record.provider == "codex":
            account_id = _codex_account_id(token["access_token"])
            if account_id != record.account_id:
                raise CredentialChangedError(
                    record.provider,
                    "refresh resolved a different account; start a new session",
                )
        elif token["account_id"] is not None and record.account_id is None:
            account_id = str(token["account_id"])
        elif token["account_id"] is not None and record.account_id is not None:
            refreshed_identity = str(token["account_id"])
            stored_kind = record.account_id.partition(":")[0]
            refreshed_kind = refreshed_identity.partition(":")[0]
            if (
                stored_kind == refreshed_kind
                and refreshed_identity != record.account_id
            ):
                raise CredentialChangedError(
                    record.provider,
                    "refresh resolved a different account; start a new session",
                )
        return self.store.replace_subscription(
            record.binding,
            access_token=token["access_token"],
            refresh_token=token["refresh_token"],
            expires_at=token["expires_at"],
            account_id=account_id,
        )


def browser_sign_in(
    provider: str,
    *,
    store: CredentialStore | None = None,
    http=None,
    now: Callable[[], float] | None = None,
    random_bytes: Callable[[int], bytes] | None = None,
    open_browser: Callable[[str], object] | None = None,
    receive_callback=None,
) -> AuthBinding:
    """Complete one browser OAuth/PKCE flow and select its credential."""
    spec = provider_spec(provider)
    store = store or CredentialStore()
    http = http or requests.Session()
    now = now or time.time
    random_bytes = random_bytes or secrets.token_bytes
    open_browser = open_browser or webbrowser.open
    receive_callback = receive_callback or _receive_browser_callback

    verifier = _base64url(random_bytes(64))
    challenge = _base64url(hashlib.sha256(verifier.encode()).digest())
    state = _base64url(random_bytes(32))
    params = {
        "response_type": "code",
        "client_id": spec.client_id,
        "redirect_uri": spec.redirect_uri,
        "scope": spec.scopes,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if provider == "claude":
        params["code"] = "true"
    else:
        params.update({
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "yuj",
        })
    authorization_url = f"{spec.authorize_url}?{urlencode(params)}"
    callback = receive_callback(
        spec=spec,
        state=state,
        authorization_url=authorization_url,
        open_browser=open_browser,
    )
    if not callback.code or callback.state != state:
        raise AuthProtocolError(provider, "browser callback state did not match")

    body = {
        "grant_type": "authorization_code",
        "client_id": spec.client_id,
        "code": callback.code,
        "redirect_uri": spec.redirect_uri,
        "code_verifier": verifier,
    }
    if provider == "claude":
        body["state"] = state
    request: dict[str, object] = {
        "headers": _token_headers(spec),
        "timeout": (10, 30),
    }
    request["json" if spec.token_encoding == "json" else "data"] = body
    try:
        response = http.post(spec.token_url, **request)
    except Exception as exc:
        raise AuthProtocolError(provider, "token exchange transport failed") from exc
    if not _response_ok(response):
        _raise_auth_response(provider, response, operation="exchange")
    token = _parse_token_response(provider, response, now=now())
    account_id = (
        _codex_account_id(token["access_token"])
        if provider == "codex"
        else token["account_id"]
    )
    binding = store.save_subscription(
        provider,
        access_token=token["access_token"],
        refresh_token=token["refresh_token"],
        expires_at=token["expires_at"],
        account_id=account_id,
    )
    store.select(binding)
    return binding


def validate_auth_endpoint(cfg, binding: AuthBinding) -> ProviderSpec:
    """Reject provider or endpoint drift before a managed credential is read."""
    spec = provider_spec(binding.provider)
    expected_url = (
        spec.subscription_base_url
        if binding.auth_method == "subscription"
        else spec.api_key_base_url
    )
    actual_url = str(getattr(cfg, "base_url", "")).rstrip("/")
    if getattr(cfg, "provider", "") != spec.core_provider:
        raise AuthEndpointMismatchError(
            binding.provider, "configured provider changed after authentication"
        )
    if actual_url != expected_url.rstrip("/"):
        raise AuthEndpointMismatchError(
            binding.provider, "configured endpoint changed after authentication"
        )
    return spec


def classify_provider_response(provider: str, response) -> None:
    """Raise a safe specific error for a non-success model response."""
    if _response_ok(response):
        return
    status = int(getattr(response, "status_code", 0) or 0)
    if status == 401:
        raise CredentialRevokedError(provider, "credential was rejected or revoked")
    if status == 403:
        raise AccountIneligibleError(
            provider, "account is not eligible for this request"
        )
    if status in {404, 405, 410, 422}:
        raise UnsupportedAuthError(
            provider, "subscription protocol is unsupported or changed"
        )
    if status == 400:
        raise AuthProtocolError(provider, "provider rejected the subscription request")
    raise ProviderRequestError(provider, f"provider request failed with HTTP {status}")


def _receive_browser_callback(
    *,
    spec: ProviderSpec,
    state: str,
    authorization_url: str,
    open_browser: Callable[[str], object],
) -> BrowserCallback:
    result: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib callback name
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            code = query.get("code", [""])[0]
            callback_state = query.get("state", [""])[0]
            valid = (
                parsed.path == spec.callback_path
                and bool(code)
                and callback_state == state
            )
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if valid:
                result.update(code=code, state=callback_state)
                body = "Authentication completed. You can close this window."
            else:
                body = "Authentication did not complete. Return to the terminal."
            self.wfile.write(body.encode())

        def log_message(self, _format, *_args):
            return

    class CallbackServer(http.server.HTTPServer):
        allow_reuse_address = True

    try:
        server = CallbackServer(("127.0.0.1", spec.callback_port), Handler)
    except OSError as exc:
        raise AuthProtocolError(
            spec.name, f"callback port {spec.callback_port} is unavailable"
        ) from exc
    try:
        server.timeout = 300
        open_browser(authorization_url)
        server.handle_request()
    finally:
        server.server_close()
    if not result:
        raise AuthProtocolError(spec.name, "browser callback was missing or invalid")
    return BrowserCallback(result["code"], result["state"])


def _parse_token_response(
    provider: str,
    response,
    *,
    now: float,
    previous_refresh_token: str | None = None,
) -> dict[str, object]:
    try:
        raw = response.json()
    except Exception as exc:
        raise CredentialMalformedError(
            provider, "token response was not valid JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise CredentialMalformedError(provider, "token response was malformed")
    access_token = raw.get("access_token")
    refresh_token = raw.get("refresh_token") or previous_refresh_token
    expires_in = raw.get("expires_in")
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or isinstance(expires_in, bool)
        or not isinstance(expires_in, (int, float))
        or expires_in <= 0
    ):
        raise CredentialMalformedError(provider, "token response fields were malformed")
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": now + float(expires_in),
        "account_id": (
            _claude_billing_identity(raw)
            if provider == "claude"
            else None
        ),
    }


def _claude_billing_identity(raw: dict) -> str | None:
    for kind, field, flat_keys in (
        (
            "organization",
            "organization",
            ("organization_uuid", "organization_id"),
        ),
        ("account", "account", ("account_uuid", "account_id")),
    ):
        container = raw.get(field)
        if isinstance(container, dict):
            for key in ("uuid", "id"):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return f"{kind}:{value}"
        for key in flat_keys:
            value = raw.get(key)
            if isinstance(value, str) and value:
                return f"{kind}:{value}"
    return None


def _raise_auth_response(provider: str, response, *, operation: str) -> None:
    status = int(getattr(response, "status_code", 0) or 0)
    code = _safe_error_code(response).lower()
    if status == 403 or "ineligible" in code or "not_eligible" in code:
        raise AccountIneligibleError(
            provider, f"account rejected during token {operation}"
        )
    if status == 404 or "unsupported" in code:
        raise UnsupportedAuthError(
            provider, f"token {operation} protocol is unsupported"
        )
    if (
        status == 401
        or "reused" in code
        or "invalidated" in code
        or "revoked" in code
        or code in {"invalid_grant", "invalid_token", "invalid_refresh_token"}
    ):
        raise CredentialRevokedError(
            provider, f"credential was revoked during token {operation}"
        )
    if "expired" in code:
        raise CredentialExpiredError(
            provider, f"credential expired during token {operation}"
        )
    raise AuthProtocolError(
        provider, f"token {operation} failed with HTTP {status}"
    )


def _safe_error_code(response) -> str:
    try:
        raw = response.json()
    except Exception:
        return ""
    if not isinstance(raw, dict):
        return ""
    error = raw.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        for key in ("code", "type", "error"):
            if isinstance(error.get(key), str):
                return error[key]
    if isinstance(raw.get("code"), str):
        return raw["code"]
    return ""


def _codex_account_id(token: str) -> str:
    parts = token.split(".")
    if len(parts) != 3:
        raise CredentialMalformedError("codex", "access token is not a valid JWT")
    try:
        payload = json.loads(_decode_base64url(parts[1]))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CredentialMalformedError(
            "codex", "access token is not a valid JWT"
        ) from exc
    claims = (
        payload.get("https://api.openai.com/auth")
        if isinstance(payload, dict)
        else None
    )
    account_id = claims.get("chatgpt_account_id") if isinstance(claims, dict) else None
    if not isinstance(account_id, str) or not account_id:
        raise AccountIneligibleError(
            "codex", "credential has no eligible ChatGPT account"
        )
    return account_id


def _token_headers(
    spec: ProviderSpec, *, refresh: bool = False
) -> dict[str, str]:
    content_type = (
        "application/json"
        if spec.token_encoding == "json"
        else "application/x-www-form-urlencoded"
    )
    headers = {"Content-Type": content_type}
    if refresh and spec.name == "claude":
        headers.update({
            "anthropic-beta": _CLAUDE_REFRESH_BETA,
            "User-Agent": _CLAUDE_REFRESH_USER_AGENT,
        })
    return headers


def _response_ok(response) -> bool:
    if hasattr(response, "ok"):
        return bool(response.ok)
    status = int(getattr(response, "status_code", 0) or 0)
    return 200 <= status < 300


def _base64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_base64url(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode()).decode()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("expected a non-empty string")
    return value


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number")
    return float(value)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _known_repository_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


__all__ = [
    "AccessCredential",
    "AccountIneligibleError",
    "AuthBinding",
    "AuthEndpointMismatchError",
    "AuthProtocolError",
    "CredentialChangedError",
    "CredentialExpiredError",
    "CredentialMalformedError",
    "CredentialMissingError",
    "CredentialRecord",
    "CredentialRevokedError",
    "CredentialSession",
    "CredentialStore",
    "ProviderAuthError",
    "ProviderRequestError",
    "UnsupportedAuthError",
    "auth_home",
    "browser_sign_in",
    "classify_provider_response",
    "provider_spec",
    "validate_auth_endpoint",
]
