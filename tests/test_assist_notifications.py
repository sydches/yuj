"""Opt-in terminal notification behavior for assistant sessions."""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import _notify_session_result, main
from scripts.llm_assist.notifications import send_session_notification
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver import config as config_module
from scripts.llm_solver.config import ConfigLayerSpec, load_config, resolve_config
from scripts.llm_solver.config_inspection import (
    build_inspection_document,
    render_inspection_human,
)


@pytest.mark.parametrize(
    ("success", "finish_reason", "state"),
    [
        (True, "stop", "completed"),
        (False, "error", "failed"),
        (False, "approval_required", "approval required"),
        (False, "input_required", "input required"),
    ],
)
def test_bell_notification_distinguishes_attention_states(
    success: bool,
    finish_reason: str,
    state: str,
) -> None:
    stream = io.StringIO()

    delivered = send_session_notification(
        mode="bell",
        session_ref="1234abcd",
        success=success,
        finish_reason=finish_reason,
        interactive=True,
        stream=stream,
    )

    assert delivered is True
    assert stream.getvalue() == f"\aYuj session 1234abcd: {state}\n"


@pytest.mark.parametrize(
    ("mode", "interactive"),
    [("off", True), ("bell", False)],
)
def test_notification_is_silent_when_disabled_or_noninteractive(
    mode: str,
    interactive: bool,
) -> None:
    stream = io.StringIO()

    delivered = send_session_notification(
        mode=mode,
        session_ref="1234abcd",
        success=True,
        finish_reason="stop",
        interactive=interactive,
        stream=stream,
    )

    assert delivered is False
    assert stream.getvalue() == ""


def test_notification_failure_does_not_escape() -> None:
    class BrokenStream:
        def write(self, _text: str) -> None:
            raise OSError("terminal unavailable")

        def flush(self) -> None:
            raise AssertionError("flush must not run after write fails")

    assert send_session_notification(
        mode="bell",
        session_ref="1234abcd",
        success=False,
        finish_reason="error",
        interactive=True,
        stream=BrokenStream(),
    ) is False


def test_cli_notification_wrapper_is_best_effort() -> None:
    record = SimpleNamespace(short_id="1234abcd")

    with patch(
        "scripts.llm_assist.__main__._is_interactive",
        side_effect=OSError("terminal unavailable"),
    ):
        _notify_session_result(
            record,
            success=True,
            finish_reason="stop",
            mode="bell",
        )


def test_notification_setting_loads_validates_and_is_inspectable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "_LOCAL_CONFIG",
        tmp_path / "absent-config.local.toml",
    )
    overlay = tmp_path / "notifications.toml"
    overlay.write_text('[assistant]\nnotifications = "bell"\n')

    cfg = load_config(overlay)
    resolved = resolve_config(
        user_config=[overlay],
        layer_specs=[
            ConfigLayerSpec(
                overlay,
                "user-overlay",
                "overlay",
                "--config[1]",
            )
        ],
    )
    document = build_inspection_document(resolved, success=True)
    entries = {
        tuple(entry["path_components"]): entry
        for entry in document["settings"]
    }

    assert cfg.assistant_notifications == "bell"
    assert entries[("assistant", "notifications")] == {
        "path": "assistant.notifications",
        "path_components": ["assistant", "notifications"],
        "value": "bell",
        "source_layer": "user-overlay",
        "redacted": False,
        "redaction_reasons": [],
    }
    assert (
        'assistant.notifications = "bell" [user-overlay]'
        in render_inspection_human(document)
    )

    overlay.write_text('[assistant]\nnotifications = "desktop"\n')
    with pytest.raises(
        ValueError,
        match=(
            "config error: assistant.notifications must be 'off' or 'bell', "
            "got 'desktop'"
        ),
    ):
        load_config(overlay)


def test_completed_cli_run_delivers_one_safe_notification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "_LOCAL_CONFIG",
        tmp_path / "absent-config.local.toml",
    )
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    overlay = tmp_path / "notifications.toml"
    overlay.write_text('[assistant]\nnotifications = "bell"\n')

    def fake_run_session(store_obj, record, *, resume):
        trace_path = Path(record.artifact_dir) / ".trace.jsonl"
        trace_path.write_text(json.dumps({
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        }) + "\n")
        store_obj.update_session(
            record.session_id,
            status="completed",
            last_finish_reason="stop",
        )
        return True, "stop"

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__.preflight_assistant_startup"),
        patch(
            "scripts.llm_assist.__main__.resolve_served_model",
            return_value=("served-model", ["served-model"]),
        ),
        patch(
            "scripts.llm_assist.__main__.run_session",
            side_effect=fake_run_session,
        ),
        patch(
            "scripts.llm_assist.__main__.send_session_notification"
        ) as notifier,
        patch("scripts.llm_assist.__main__._is_interactive", return_value=True),
    ):
        rc = main([
            "--cwd",
            str(work_dir),
            "--config",
            str(overlay),
            "--prompt-text",
            "private task text",
        ])

    record = store.list_sessions(limit=1)[0]
    assert rc == 0
    notifier.assert_called_once_with(
        mode="bell",
        session_ref=record.short_id,
        success=True,
        finish_reason="stop",
        interactive=True,
    )
