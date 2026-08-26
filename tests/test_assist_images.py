from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from _config_helpers import make_config
from scripts.llm_assist.__main__ import main
from scripts.llm_assist._anthropic import _to_anthropic_payload
from scripts.llm_assist._auth import CredentialStore
from scripts.llm_assist._codex import _to_responses_payload
from scripts.llm_assist._images import (
    ImageInputError,
    image_evidence,
    load_session_images,
    read_image_inputs,
    save_image_segment,
)
from scripts.llm_assist.runner import _make_client
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.models import model_supports_image_inputs
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.replay_client import ReplayClient
from scripts.llm_solver.server.types import ImageInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
PNG_1X1_ALT = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg"
    "YPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)


def _write_png(path: Path, data: bytes = PNG_1X1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _image_input(data: bytes = PNG_1X1) -> ImageInput:
    return ImageInput(media_type="image/png", data=data)


class _SDKResponse:
    def __init__(self, text: str = "ok"):
        self.choices = [SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=[]),
            finish_reason="stop",
        )]
        self.usage = SimpleNamespace(prompt_tokens=7, completion_tokens=2)

    def model_dump_json(self) -> str:
        return json.dumps({
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        })


class _FakeOpenAI:
    requests: list[dict] = []

    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda **payload: self._record(payload)
        ))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    @classmethod
    def _record(cls, payload: dict):
        cls.requests.append(payload)
        return _SDKResponse()


class _HTTPResponse:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        lines: list[str] | None = None,
    ):
        self.status_code = 200
        self.ok = True
        self._payload = payload or {}
        self._lines = lines or []
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        assert decode_unicode is True
        return iter(self._lines)


class _FakeHTTP:
    def __init__(self, response: _HTTPResponse):
        self.response = response
        self.posts: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs):
        self.posts.append((url, kwargs))
        return self.response


def _anthropic_response() -> _HTTPResponse:
    return _HTTPResponse({
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 2},
    })


def _jwt() -> str:
    encode = lambda value: base64.urlsafe_b64encode(  # noqa: E731
        json.dumps(value, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return (
        f"{encode({'alg': 'none'})}."
        f"{encode({'https://api.openai.com/auth': {'chatgpt_account_id': 'acct'}})}.sig"
    )


def _codex_response() -> _HTTPResponse:
    completed = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }],
            "usage": {"input_tokens": 7, "output_tokens": 2},
        },
    }
    return _HTTPResponse(lines=[
        "event: response.completed",
        f"data: {json.dumps(completed)}",
        "",
        "data: [DONE]",
        "",
    ])


def test_reads_actual_png_bytes_and_snapshots_exact_session_evidence(tmp_path: Path):
    source = _write_png(tmp_path / "looks-like-text.txt")

    pending = read_image_inputs([source])
    assert len(pending) == 1
    assert pending[0].media_type == "image/png"
    assert pending[0].size_bytes == len(PNG_1X1)
    assert pending[0].sha256 == hashlib.sha256(PNG_1X1).hexdigest()
    assert (pending[0].width, pending[0].height) == (1, 1)

    artifact_dir = tmp_path / "session"
    saved = save_image_segment(
        artifact_dir,
        segment_number=1,
        prompt_text="Inspect the supplied image.",
        images=pending,
    )
    assert saved[0].data == PNG_1X1
    assert saved[0].relative_path == "attachments/segment-0001/image-0001.png"

    manifest_text = (artifact_dir / "attachments.json").read_text()
    manifest = json.loads(manifest_text)
    assert manifest["schema"] == "yuj.assistant-attachments"
    assert manifest["schema_version"] == 1
    assert manifest["segments"][0]["user_text"] == {
        "sha256": hashlib.sha256(b"Inspect the supplied image.").hexdigest(),
        "utf8_bytes": len(b"Inspect the supplied image."),
        "chars": len("Inspect the supplied image."),
    }
    assert manifest["segments"][0]["images"][0] == {
        "image_number": 1,
        "display_name": "looks-like-text.txt",
        "media_type": "image/png",
        "size_bytes": len(PNG_1X1),
        "sha256": hashlib.sha256(PNG_1X1).hexdigest(),
        "width": 1,
        "height": 1,
        "relative_path": "attachments/segment-0001/image-0001.png",
    }
    assert str(tmp_path) not in manifest_text
    assert str(source) not in manifest_text
    assert base64.b64encode(PNG_1X1).decode() not in manifest_text
    assert (artifact_dir / saved[0].relative_path).read_bytes() == PNG_1X1


def test_changed_or_missing_source_cannot_replace_saved_bytes(tmp_path: Path):
    source = _write_png(tmp_path / "screen.png")
    artifact_dir = tmp_path / "session"
    save_image_segment(
        artifact_dir,
        segment_number=1,
        prompt_text="Use this screen.",
        images=read_image_inputs([source]),
    )

    source.write_bytes(PNG_1X1_ALT)
    loaded_after_change = load_session_images(artifact_dir)
    assert loaded_after_change[0].data == PNG_1X1
    assert loaded_after_change[0].sha256 == hashlib.sha256(PNG_1X1).hexdigest()

    source.unlink()
    loaded_after_delete = load_session_images(artifact_dir)
    assert loaded_after_delete[0].data == PNG_1X1


def test_display_metadata_is_printable_and_bounded(tmp_path: Path):
    source = _write_png(tmp_path / ("\x7f" + "x" * 120 + ".png"))

    image = read_image_inputs([source])[0]

    assert image.display_name.startswith("_")
    assert len(image.display_name) == 96
    assert image.display_name.isprintable()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "not readable"),
        ("directory", "regular file"),
        ("unsupported", "unsupported image media type"),
        ("symlink", "symbolic link"),
    ],
)
def test_rejects_untrusted_or_unsupported_image_paths(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    path = tmp_path / "input.png"
    if mutation == "directory":
        path.mkdir()
    elif mutation == "unsupported":
        path.write_bytes(b"not an image despite the suffix")
    elif mutation == "symlink":
        target = _write_png(tmp_path / "target.png")
        path.symlink_to(target)

    with pytest.raises(ImageInputError, match=message):
        read_image_inputs([path])


def test_rejects_count_per_file_and_aggregate_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.llm_assist._images as images_module

    first = _write_png(tmp_path / "first.png")
    second = _write_png(tmp_path / "second.png", PNG_1X1_ALT)

    monkeypatch.setattr(images_module, "MAX_IMAGE_COUNT", 1)
    with pytest.raises(ImageInputError, match="at most 1 image"):
        read_image_inputs([first, second])

    monkeypatch.setattr(images_module, "MAX_IMAGE_COUNT", 20)
    monkeypatch.setattr(images_module, "MAX_IMAGE_BYTES", len(PNG_1X1) - 1)
    with pytest.raises(ImageInputError, match="per-file limit"):
        read_image_inputs([first])

    monkeypatch.setattr(images_module, "MAX_IMAGE_BYTES", len(PNG_1X1) + 10)
    monkeypatch.setattr(
        images_module,
        "MAX_IMAGE_TOTAL_BYTES",
        len(PNG_1X1) + len(PNG_1X1_ALT) - 1,
    )
    with pytest.raises(ImageInputError, match="aggregate limit"):
        read_image_inputs([first, second])


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_saved_attachment_integrity_failure_is_loud(tmp_path: Path, damage: str):
    source = _write_png(tmp_path / "input.png")
    artifact_dir = tmp_path / "session"
    saved = save_image_segment(
        artifact_dir,
        segment_number=1,
        prompt_text="Inspect it.",
        images=read_image_inputs([source]),
    )[0]
    stored_path = artifact_dir / saved.relative_path
    if damage == "missing":
        stored_path.unlink()
    else:
        stored_path.write_bytes(PNG_1X1_ALT)

    with pytest.raises(ImageInputError, match="saved image evidence"):
        load_session_images(artifact_dir)


def test_saved_attachment_parent_symlink_cannot_read_outside_session(
    tmp_path: Path,
):
    source = _write_png(tmp_path / "input.png")
    artifact_dir = tmp_path / "session"
    save_image_segment(
        artifact_dir,
        segment_number=1,
        prompt_text="Inspect it.",
        images=read_image_inputs([source]),
    )
    outside = tmp_path / "outside"
    (artifact_dir / "attachments").rename(outside)
    (artifact_dir / "attachments").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(ImageInputError, match="saved image evidence.*symbolic link"):
        load_session_images(artifact_dir)


def test_snapshot_rejects_attachment_parent_symlink(tmp_path: Path):
    source = _write_png(tmp_path / "input.png")
    artifact_dir = tmp_path / "session"
    outside = tmp_path / "outside"
    artifact_dir.mkdir()
    outside.mkdir()
    (artifact_dir / "attachments").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(ImageInputError, match="attachment directory.*symbolic link"):
        save_image_segment(
            artifact_dir,
            segment_number=1,
            prompt_text="Inspect it.",
            images=read_image_inputs([source]),
        )
    assert list(outside.iterdir()) == []


def test_new_task_image_flag_saves_segment_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = SessionStore(tmp_path / "assist")
    target = tmp_path / "target"
    target.mkdir()
    source = _write_png(tmp_path / "screen.png")
    second_source = _write_png(tmp_path / "expected.png", PNG_1X1_ALT)
    seen: list[str] = []

    def fake_run_session(store_obj, record, *, resume):
        assert resume is False
        assert [
            image.data for image in load_session_images(record.artifact_path)
        ] == [PNG_1X1, PNG_1X1_ALT]
        seen.append(record.session_id)
        store_obj.update_session(
            record.session_id, status="completed", last_finish_reason="stop"
        )
        return True, "stop"

    monkeypatch.chdir(target)
    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.preflight_assistant_startup"), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("vision-test", ["vision-test"]),
            ), \
            patch("scripts.llm_assist.__main__.validate_image_capability"), \
            patch(
                "scripts.llm_assist.__main__.run_session",
                side_effect=fake_run_session,
            ):
        result = main([
            "--image", str(source),
            "--image", str(second_source),
            "Inspect", "the", "failure", "shown.",
        ])

    assert result == 0
    assert len(seen) == 1
    record = store.get_session(seen[0])
    assert record is not None
    assert record.prompt_text == "Inspect the failure shown."
    evidence = image_evidence(record.artifact_path)
    assert len(evidence) == 2
    assert evidence[0].segment_number == 1
    assert evidence[0].display_name == "screen.png"
    assert evidence[1].display_name == "expected.png"


def test_resume_image_flag_requires_and_delivers_segment_text(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    target = tmp_path / "target"
    target.mkdir()
    source = _write_png(tmp_path / "new-screen.png")
    record = store.create_session(
        cwd=target,
        model="vision-test",
        prompt_text="Fix the original failure.",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    record.artifact_path.mkdir(parents=True)
    (record.artifact_path / ".trace.jsonl").write_text(json.dumps({
        "event": "session_end",
        "session_number": 1,
        "finish_reason": "max_turns",
        "turns": 2,
    }) + "\n")
    store.update_session(
        record.session_id, status="paused", last_finish_reason="max_turns"
    )
    seen_text: list[str] = []

    def fake_run_session(
        store_obj,
        selected,
        *,
        resume,
        resume_prompt_text=None,
    ):
        assert resume is True
        seen_text.append(resume_prompt_text)
        saved = load_session_images(selected.artifact_path)
        assert saved[0].segment_number == 2
        assert saved[0].data == PNG_1X1
        store_obj.update_session(
            selected.session_id, status="completed", last_finish_reason="stop"
        )
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.validate_image_capability"), \
            patch(
                "scripts.llm_assist.__main__.run_session",
                side_effect=fake_run_session,
            ):
        result = main([
            "resume", record.session_id,
            "--prompt-text", "The new screenshot shows the remaining error.",
            "--image", str(source),
        ])

    assert result == 0
    assert seen_text == ["The new screenshot shows the remaining error."]


def test_resume_rejects_images_without_follow_up_text():
    with pytest.raises(SystemExit, match="require follow-up text"):
        main(["resume", "SESSION", "--image", "screen.png"])


def test_resume_accepts_multiline_stdin_without_an_image_and_records_source(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    target = tmp_path / "target"
    target.mkdir()
    record = store.create_session(
        cwd=target,
        model="text-test",
        prompt_text="Fix the original failure.",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    record.artifact_path.mkdir(parents=True)
    (record.artifact_path / ".trace.jsonl").write_text(json.dumps({
        "event": "session_end",
        "session_number": 1,
        "finish_reason": "max_turns",
        "turns": 2,
    }) + "\n")
    store.update_session(
        record.session_id, status="paused", last_finish_reason="max_turns"
    )
    prompt = "Check both cases.\n\n  Keep this indentation.\n"
    seen_text: list[str] = []

    def fake_run_session(
        store_obj,
        selected,
        *,
        resume,
        resume_prompt_text=None,
    ):
        assert resume is True
        seen_text.append(resume_prompt_text)
        assert load_session_images(selected.artifact_path) == ()
        store_obj.update_session(
            selected.session_id, status="completed", last_finish_reason="stop"
        )
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.run_session",
                side_effect=fake_run_session,
            ), patch("sys.stdin", io.StringIO(prompt)):
        result = main([
            "resume", record.session_id,
            "--prompt-file", "-",
        ])

    assert result == 0
    assert seen_text == [prompt]
    events = [
        json.loads(line)
        for line in (record.artifact_path / ".trace.jsonl").read_text().splitlines()
    ]
    evidence = next(event for event in events if event["event"] == "operator_followup")
    assert evidence == {
        "event": "operator_followup",
        "trace_schema_version": 2,
        "session_number": 2,
        "prompt_source": "stdin",
        "text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "text_chars": len(prompt),
    }
    assert prompt not in (record.artifact_path / ".trace.jsonl").read_text()


def test_unsupported_local_model_is_rejected_before_session_or_model_run(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    target = tmp_path / "target"
    target.mkdir()
    source = _write_png(tmp_path / "screen.png")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.preflight_assistant_startup"
            ) as preflight_mock, \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("local-text-model", ["local-text-model"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session") as run_mock:
        with pytest.raises(SystemExit, match="does not declare image input support"):
            main([
                "--cwd", str(target),
                "--image", str(source),
                "Inspect it.",
            ])

    run_mock.assert_not_called()
    preflight_mock.assert_not_called()
    assert store.list_sessions() == []


def test_status_and_show_render_bounded_evidence_without_bytes_or_source_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    store = SessionStore(tmp_path / "assist")
    target = tmp_path / "target"
    target.mkdir()
    source = _write_png(tmp_path / "private-parent" / "screen.png")
    record = store.create_session(
        cwd=target,
        model="vision-test",
        prompt_text="Inspect it.",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_image_segment(
        record.artifact_path,
        segment_number=1,
        prompt_text=record.prompt_text,
        images=read_image_inputs([source]),
    )
    digest = hashlib.sha256(PNG_1X1).hexdigest()

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["status", record.session_id]) == 0
        status_output = capsys.readouterr().out
        assert main([
            "show", record.session_id, "--turns", "0", "--trace-lines", "0"
        ]) == 0
        show_output = capsys.readouterr().out

    for output in (status_output, show_output):
        assert "attachments: 1 image, " in output
        assert "media_type=image/png" in output
        assert f"sha256={digest}" in output
        assert "dimensions=1x1" in output
        assert "screen.png" in output
        assert str(source.parent) not in output
        assert base64.b64encode(PNG_1X1).decode() not in output


def test_openai_compatible_request_uses_image_parts_and_preserves_text(
    monkeypatch: pytest.MonkeyPatch,
):
    _FakeOpenAI.requests = []
    monkeypatch.setattr(
        "scripts.llm_solver.server.client.openai.OpenAI", _FakeOpenAI
    )
    client = LlamaClient(make_config(), profile=None)
    client.set_image_inputs([_image_input()])
    client.chat([{"role": "user", "content": "Inspect exactly."}], [], turn=0)

    content = _FakeOpenAI.requests[0]["messages"][0]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(PNG_1X1).decode()
        },
    }
    assert content[1] == {"type": "text", "text": "Inspect exactly."}


def test_anthropic_and_codex_adapters_emit_provider_native_image_blocks():
    common = {
        "model": "vision-test",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(PNG_1X1).decode()
                    },
                },
                {"type": "text", "text": "Inspect exactly."},
            ],
        }],
        "max_tokens": 10,
    }

    expected_anthropic_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(PNG_1X1).decode(),
            },
        },
        {"type": "text", "text": "Inspect exactly."},
    ]
    assert _to_anthropic_payload(
        common, subscription=False
    )["messages"][0]["content"] == expected_anthropic_content
    assert _to_anthropic_payload(
        common, subscription=True
    )["messages"][0]["content"] == expected_anthropic_content

    codex_content = _to_responses_payload(
        common, session_id="session"
    )["input"][0]["content"]
    assert codex_content == [
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,"
            + base64.b64encode(PNG_1X1).decode(),
        },
        {"type": "input_text", "text": "Inspect exactly."},
    ]


@pytest.mark.parametrize(
    ("provider", "auth_method"),
    [
        ("claude", "api_key"),
        ("claude", "subscription"),
        ("codex", "api_key"),
        ("codex", "subscription"),
    ],
)
def test_image_request_reaches_every_managed_provider_auth_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    auth_method: str,
):
    _FakeOpenAI.requests = []
    monkeypatch.setattr(
        "scripts.llm_solver.server.client.openai.OpenAI", _FakeOpenAI
    )
    credentials = CredentialStore(tmp_path / "auth")
    if auth_method == "api_key":
        binding = credentials.save_api_key(provider, secret="key")
    else:
        binding = credentials.save_subscription(
            provider,
            access_token="access" if provider == "claude" else _jwt(),
            refresh_token="refresh",
            expires_at=10_000,
            account_id="acct" if provider == "codex" else None,
        )
    http = _FakeHTTP(
        _anthropic_response() if provider == "claude" else _codex_response()
    )
    base_url = {
        ("claude", "api_key"): "https://api.anthropic.com/v1",
        ("claude", "subscription"): "https://api.anthropic.com/v1",
        ("codex", "api_key"): "https://api.openai.com/v1",
        ("codex", "subscription"): "https://chatgpt.com/backend-api/codex",
    }[(provider, auth_method)]
    client = _make_client(
        make_config(
            provider="anthropic" if provider == "claude" else "openai-compatible",
            base_url=base_url,
            api_key="yuj-host-credential",
        ),
        profile=load_profile("_base", PROJECT_ROOT / "profiles"),
        auth_binding=binding,
        auth_store=credentials,
        http=http,
        now=lambda: 1000.0,
    )
    client.set_image_inputs([_image_input()])
    client.chat([{"role": "user", "content": "Inspect exactly."}], [], turn=0)

    data_url = (
        "data:image/png;base64," + base64.b64encode(PNG_1X1).decode()
    )
    if provider == "codex" and auth_method == "api_key":
        content = _FakeOpenAI.requests[0]["messages"][0]["content"]
        assert content == [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Inspect exactly."},
        ]
    elif provider == "codex":
        content = http.posts[0][1]["json"]["input"][0]["content"]
        assert content == [
            {"type": "input_image", "image_url": data_url},
            {"type": "input_text", "text": "Inspect exactly."},
        ]
    else:
        request = http.posts[0][1]
        body = request.get("json") or json.loads(request["data"])
        content = body["messages"][0]["content"]
        assert content == [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(PNG_1X1).decode(),
                },
            },
            {"type": "text", "text": "Inspect exactly."},
        ]


def test_model_capability_is_fail_closed_but_profiles_can_declare_local_support():
    assert model_supports_image_inputs(
        "gpt-5.4", provider="openai", profile_supports_image_inputs=False
    )
    assert model_supports_image_inputs(
        "gpt-4o-mini-2024-07-18",
        provider="openai",
        profile_supports_image_inputs=False,
    )
    assert model_supports_image_inputs(
        "o3-2025-04-16",
        provider="openai",
        profile_supports_image_inputs=False,
    )
    assert model_supports_image_inputs(
        "claude-sonnet-4-5",
        provider="anthropic",
        profile_supports_image_inputs=False,
    )
    assert not model_supports_image_inputs(
        "local-text-model",
        provider="openai-compatible",
        profile_supports_image_inputs=False,
    )
    assert not model_supports_image_inputs(
        "unknown-remote-model",
        provider="openai",
        profile_supports_image_inputs=False,
    )
    assert model_supports_image_inputs(
        "local-vision-model",
        provider="openai-compatible",
        profile_supports_image_inputs=True,
    )


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o-audio-preview",
        "gpt-4o-realtime-preview",
        "gpt-4o-search-preview",
        "gpt-4o-transcribe",
        "o1-mini",
        "o1-preview",
        "o3-mini",
    ],
)
def test_openai_text_audio_and_specialized_variants_fail_closed(model: str):
    assert not model_supports_image_inputs(
        model,
        provider="openai",
        profile_supports_image_inputs=False,
    )


def test_profile_image_capability_defaults_false_and_validates_boolean(tmp_path: Path):
    base = load_profile("_base", PROJECT_ROOT / "profiles")
    assert base.supports_image_inputs is False

    profile_dir = tmp_path / "vision"
    profile_dir.mkdir()
    (profile_dir / "profile.toml").write_text(
        "[profile]\n"
        "format_version = 1\n"
        'name = "vision"\n'
        'family = "vision"\n'
        'inherits = ""\n'
        "\n[model]\n"
        "supports_image_inputs = true\n"
        "\n[reasoning_levels.off]\n"
        "enabled = false\n"
    )
    assert load_profile("vision", tmp_path).supports_image_inputs is True

    (profile_dir / "profile.toml").write_text(
        "[profile]\n"
        "format_version = 1\n"
        'name = "vision"\n'
        'family = "vision"\n'
        'inherits = ""\n'
        "\n[model]\n"
        'supports_image_inputs = "yes"\n'
        "\n[reasoning_levels.off]\n"
        "enabled = false\n"
    )
    with pytest.raises(ValueError, match="supports_image_inputs must be a boolean"):
        load_profile("vision", tmp_path)


def test_text_only_transport_request_is_byte_for_byte_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    _FakeOpenAI.requests = []
    monkeypatch.setattr(
        "scripts.llm_solver.server.client.openai.OpenAI", _FakeOpenAI
    )
    messages = [{"role": "user", "content": "Keep  two spaces.\n"}]
    client = LlamaClient(make_config(runtime_mode="assistant"), profile=None)
    client.chat(messages, [], turn=0)

    assert _FakeOpenAI.requests[0]["messages"] == messages
    assert _FakeOpenAI.requests[0]["messages"][0]["content"] == (
        "Keep  two spaces.\n"
    )


def test_measurement_client_does_not_discover_assistant_attachments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _FakeOpenAI.requests = []
    monkeypatch.setattr(
        "scripts.llm_solver.server.client.openai.OpenAI", _FakeOpenAI
    )
    assist_home = tmp_path / "assist"
    assist_home.mkdir()
    (assist_home / "attachments.json").write_text("not measurement input")
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(assist_home))

    client = LlamaClient(make_config(runtime_mode="measurement"), profile=None)
    client.chat([{"role": "user", "content": "measurement task"}], [], turn=0)

    assert _FakeOpenAI.requests[0]["messages"] == [
        {"role": "user", "content": "measurement task"}
    ]


def test_replay_uses_recorded_image_request_after_source_is_missing(tmp_path: Path):
    source = _write_png(tmp_path / "source.png")
    image_url = "data:image/png;base64," + base64.b64encode(
        source.read_bytes()
    ).decode()
    transcript = tmp_path / "transcript.log"
    transcript.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Inspect it."},
                ],
            }]
        })
        + "\n=== turn 001 output ===\n"
        + json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": "seen"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        })
        + "\n"
    )
    source.unlink()

    result = ReplayClient(transcript).chat(
        [{"role": "user", "content": "Inspect it."}], [], turn=0
    )

    assert result.content == "seen"
    assert image_url in transcript.read_text()
