"""Received usage survives context refusal, including a successful retry."""
import io
import json
from unittest.mock import patch

import httpx
import pytest

from _config_helpers import make_config
from test_backend_token_counting import attach_transport
from test_length_continuation_integration import _client
from scripts.llm_solver.harness.loop import Session


@pytest.mark.parametrize('outcome', ['no_recovery', 'recovered', 'refused_again', 'no_prefill'])
def test_received_usage_is_charged_once_across_continuation_refusal(tmp_path, outcome):
    cfg = make_config(max_turns=1, context_size=1000, max_tokens=399,
        tokenizer_id='auto', length_continue_max=1, sandbox_bash=False, reply_mode='conversation')
    client = _client(cfg, supports_prefill=outcome != 'no_prefill', normalize=lambda value: value)
    generations, forced, requests = [], [], []

    def handler(request):
        body = json.loads(request.content)
        requests.append(request.url.path)
        continued = body.get('continue_final_message') is True
        if request.url.path.endswith('/input_tokens'):
            return httpx.Response(200, json={'input_tokens': 1000 if continued else 600})
        assert not continued, 'refused continuation must not reach generation'
        retry_succeeded = bool(generations) and outcome == 'recovered'
        usage = (600, 20) if retry_succeeded else (600, 399)
        generations.append(usage)
        return httpx.Response(200, json={
            'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'served',
            'choices': [{'index': 0, 'finish_reason': 'stop' if retry_succeeded else 'length',
                         'message': {'role': 'assistant', 'content': 'partial ' * 399, 'tool_calls': []}}],
            'usage': {'prompt_tokens': usage[0], 'completion_tokens': usage[1],
                      'total_tokens': sum(usage)}})

    def compact(session, messages, **kwargs):
        if kwargs.get('force'):
            forced.append(True)
            if outcome in {'recovered', 'refused_again'}:
                session._compaction_count = getattr(session, '_compaction_count', 0) + 1
        return messages

    attach_transport(client, handler)
    trace = io.StringIO()
    try:
        with patch.object(Session, '_get_server_ctx', return_value=1000), patch(
            'scripts.llm_solver.harness._loop.chat_io.maybe_compact_messages', side_effect=compact):
            session = Session(cfg, client, 'system', 'task', str(tmp_path), trace_file=trace)
            result = session.run()
    finally:
        client.client.close()
    expected = [(600, 399)] + ([(600, 20)] if outcome == 'recovered' else
                              [(600, 399)] if outcome == 'refused_again' else [])
    assert generations == expected
    assert result.total_prompt_tokens == sum(p for p, _ in expected)
    assert result.total_completion_tokens == sum(c for _, c in expected)
    assert session._abandoned_chat_usage is None
    assert len(forced) == (0 if outcome == 'no_prefill' else 1)
    if outcome in {'no_recovery', 'refused_again'}:
        assert result.finish_reason == 'context_full'
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    errors = [e for e in events if e['event'] == 'api_error']
    assert len(errors) == (0 if outcome == 'no_prefill' else 2 if outcome == 'refused_again' else 1)
    assert all(e['generation_sent'] is False for e in errors)
