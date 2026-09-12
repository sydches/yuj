"""Avoid discarded transcript work without delaying a configured diary."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.server.client import LlamaClient


@pytest.mark.parametrize('enabled_sink,record', [(False, True), (True, False)])
def test_disabled_transcript_does_not_serialize_response(tmp_path, monkeypatch, enabled_sink, record):
    monkeypatch.setenv('YUJ_STREAMING', '0')
    client = LlamaClient(make_config(context_size=0, max_tokens=20))
    client.get_request_token_counter = lambda: None
    if enabled_sink:
        client.set_transcript(tmp_path / 'diary.log')
    response = SimpleNamespace(
        choices=[], model_dump_json=Mock(side_effect=AssertionError('unused serialization')),
    )
    client.client.chat.completions.create = Mock(return_value=response)
    try:
        assert client._call_api({'messages': []}, record_transcript=record) is response
        response.model_dump_json.assert_not_called()
        assert client._transcript_call_n == 0
    finally:
        client.close_transcript()
        client.client.close()


def test_transcript_input_is_visible_before_request_and_output_before_return(tmp_path, monkeypatch):
    monkeypatch.setenv('YUJ_STREAMING', '0')
    client = LlamaClient(make_config(context_size=0, max_tokens=20))
    client.get_request_token_counter = lambda: None
    diary = tmp_path / 'diary.log'
    client.set_transcript(diary)
    response = SimpleNamespace(choices=[], model_dump_json=lambda: '{"result":"café"}')

    def create(**payload):
        text = diary.read_text()
        assert 'turn 001 input' in text
        assert 'turn 001 output' not in text
        return response

    client.client.chat.completions.create = create
    try:
        assert client._call_api({'messages': []}) is response
        assert diary.read_text().endswith('=== turn 001 output ===\n{"result":"café"}\n')
    finally:
        client.close_transcript()
        client.client.close()
