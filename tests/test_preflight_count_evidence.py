"""Preflight must distinguish backend estimates from matching reported counts."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.llm_solver.harness._loop.run_step import _preflight_estimate
from scripts.llm_solver.harness.local_tokenizer import LocalTokenizer
from scripts.llm_solver.harness.operation_timing import operation_profile
from scripts.llm_solver.server.token_counting import (
    BackendTokenCounter, _request_count_reuse, clear_request_count_reuse, request_count_reuse,
)


def _session(value, record):
    counter = SimpleNamespace(count=Mock(return_value=value), last=record)
    return SimpleNamespace(
        _tokenizer=counter,
        context=SimpleNamespace(get_messages=lambda: [{"role": "user", "content": "task"}],
                                estimate_tokens=lambda: 777),
    )


@pytest.mark.parametrize("record,authoritative", [
    ({"count_basis": "backend_input_tokens", "count_precision": "backend_reported", "prompt_tokens": 100}, True),
    ({"count_basis": "backend_input_tokens", "count_precision": "estimate", "prompt_tokens": 100}, False),
    ({"count_basis": "backend_input_tokens", "count_precision": "unverified", "prompt_tokens": 100}, False),
    ({"count_basis": "backend_input_tokens", "count_precision": "backend_reported", "prompt_tokens": 99}, False),
    ({"count_basis": "backend_input_tokens"}, False),
    ({"count_basis": "backend_input_tokens_unverified", "count_precision": "unverified", "prompt_tokens": 100}, False),
])
def test_record_basis_precision_and_value_must_agree(monkeypatch, record, authoritative):
    monkeypatch.setattr("scripts.llm_solver.harness._loop.run_step.effective_model_tool_schemas", lambda _: [])
    session = _session(100, record)
    assert _preflight_estimate(session) == 100
    assert session._preflight_count_authoritative is authoritative


@pytest.mark.parametrize("value", [True, -1, 3.5, "100"])
def test_invalid_counter_value_does_not_become_an_integer_fact(monkeypatch, value):
    monkeypatch.setattr("scripts.llm_solver.harness._loop.run_step.effective_model_tool_schemas", lambda _: [])
    session = _session(value, {"count_basis": "backend_input_tokens",
                              "count_precision": "backend_reported", "prompt_tokens": value})
    assert _preflight_estimate(session) == 777
    assert session._preflight_count_authoritative is False


def test_counting_recovery_reassesses_precision_each_time(monkeypatch):
    monkeypatch.setattr("scripts.llm_solver.harness._loop.run_step.effective_model_tool_schemas", lambda _: [])
    record = {"count_basis": "backend_input_tokens", "count_precision": "backend_reported", "prompt_tokens": 100}
    session = _session(100, record)
    for precision, expected in (("backend_reported", True), ("estimate", False), ("backend_reported", True)):
        session._tokenizer.last = {**record, "count_precision": precision}
        assert _preflight_estimate(session) == 100
        assert session._preflight_count_authoritative is expected
    session._tokenizer.count.side_effect = OSError("count unavailable")
    assert _preflight_estimate(session) == 777
    assert session._preflight_count_authoritative is False


@pytest.mark.parametrize('failure', [None, 'lookup', 'messages', 'schemas', 'count', 'fallback'])
def test_preflight_spans_preserve_evaluation_order_and_fallback(monkeypatch, failure):
    from scripts.llm_solver.harness import operation_timing
    from scripts.llm_solver.harness.context import chars_div_4

    events, clock = [], [0.0]
    messages = [{'role': 'user', 'content': 'private fixture contents'}]

    def observe(stage):
        events.append(stage)
        clock[0] += 0.001
        if failure == stage or (failure == 'fallback' and stage in ('count', 'estimate')):
            raise OSError(stage)

    class Counter:
        @property
        def count(self):
            observe('lookup')
            def count(messages, tools=None):
                observe('count')
                return 100
            return count

    def get_messages():
        observe('messages')
        return messages

    def schemas(session):
        observe('schemas')
        return []

    def estimate():
        observe('estimate')
        return 777

    monkeypatch.setattr(operation_timing.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr('scripts.llm_solver.harness._loop.run_step.effective_model_tool_schemas', schemas)
    session = SimpleNamespace(_tokenizer=Counter(),
        context=SimpleNamespace(get_messages=get_messages, estimate_tokens=estimate))
    with operation_profile() as profile:
        value = _preflight_estimate(session)
        rows = {tuple(row['operation']): row for row in profile.take()}
    stages = ['lookup', 'messages', 'schemas', 'count']
    expected = stages if failure in (None, 'fallback') else stages[:stages.index(failure) + 1]
    if failure is not None:
        expected += ['estimate']
    if failure == 'fallback':
        expected += ['messages']
    assert events == expected
    assert value == (100 if failure is None else chars_div_4(messages) if failure == 'fallback' else 777)
    prefix = ('request.initial_preflight',)
    assert rows[prefix]['raised'] == 0 and rows[prefix]['self_ms'] == 1
    assert {path[-1] for path in rows} <= {
        'request.initial_preflight', 'request.messages', 'request.count', 'request.estimate_fallback'}
    assert sum(row['raised'] for row in rows.values()) == int(failure in ('messages', 'schemas', 'count', 'fallback'))
    assert (prefix + ('request.estimate_fallback',) in rows) is (failure is not None)


_LOCAL_MESSAGES = [{"role": "user", "content": "hi there λ"}]
_LOCAL_TEMPLATE = "{{bos_token}} {% for m in messages %}{{m.role}} {{m.content}} {% endfor %}{% if tools %}{{tools|tojson}}{% endif %} {{eos_token}}"


def _local_counter(tokenizer):
    counter = LocalTokenizer.__new__(LocalTokenizer)
    counter._id, counter._tok, counter._chat_template = "offline-test", tokenizer, None
    counter.last_basis = "local_tokenizer_estimate"
    return counter


def _native_tokenizer(kind="wordlevel", cls=None):
    hf = pytest.importorskip("transformers")
    native = pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    vocab = {word: index for index, word in enumerate(
        ["[UNK]", "<s>", "</s>", "user", "hi", "there", "λ", "a", "b", "ab", "abab"])}
    if kind == "bpe":
        model = native.models.BPE(vocab, [("a", "b"), ("ab", "ab")], unk_token="[UNK]")
    elif kind == "wordpiece":
        model = native.models.WordPiece(vocab, unk_token="[UNK]")
    elif kind == "unigram":
        model = native.models.Unigram([(word, -float(index)) for word, index in vocab.items()], unk_id=0)
    else:
        model = native.models.WordLevel(vocab, unk_token="[UNK]")
    backend = native.Tokenizer(model)
    backend.pre_tokenizer = native.pre_tokenizers.WhitespaceSplit()
    return (cls or hf.PreTrainedTokenizerFast)(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="<s>",
        eos_token="</s>", chat_template=_LOCAL_TEMPLATE)


def _observe_encodings(monkeypatch):
    base = pytest.importorskip("transformers").PreTrainedTokenizerBase
    original, calls = base.__call__, []

    def observed(self, text, *args, **kwargs):
        calls.append((text, kwargs.copy()))
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(base, "__call__", observed)
    return calls


@pytest.mark.parametrize("kind", ["wordlevel", "wordpiece", "unigram", "bpe"])
def test_local_native_models_reuse_exact_rendered_encoding(monkeypatch, kind):
    from copy import deepcopy

    counter = _local_counter(_native_tokenizer(kind))
    expected = counter.count(_LOCAL_MESSAGES)
    calls = _observe_encodings(monkeypatch)
    with operation_profile() as profile, request_count_reuse():
        assert counter.count(_LOCAL_MESSAGES) == counter.count(deepcopy(_LOCAL_MESSAGES)) == expected
        rows = {tuple(row['operation']): row for row in profile.take()}
    assert set(rows) == {('tokenizer.render',), ('tokenizer.encode',)}
    assert rows[('tokenizer.render',)]['calls'] == 2
    assert rows[('tokenizer.encode',)]['calls'] == 1
    assert len(calls) == 1  # admission must work, not merely preserve counts via fallback
    assert calls[0][1]["add_special_tokens"] is False
    assert counter.last_basis == "local_tokenizer_estimate"


@pytest.mark.parametrize("shape", ["list", "dict", "batch_encoding"])
def test_local_legacy_count_shapes_and_optional_dependencies(monkeypatch, shape):
    import builtins
    from collections import UserDict

    class BatchShape(UserDict):
        @property
        def input_ids(self):
            return self["input_ids"]

    class LegacyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is True
            return ([1, 2, 3, 4] if shape == "list" else
                    {"input_ids": [1, 2, 3, 4]} if shape == "dict" else
                    BatchShape({"input_ids": [1, 2, 3, 4]}))

    original_import = builtins.__import__

    def no_optional_import(name, *args, **kwargs):
        if name in {"transformers", "tokenizers.models"}:
            raise ImportError("optional dependency absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optional_import)
    with operation_profile() as profile, request_count_reuse():
        assert _local_counter(LegacyTokenizer()).count(_LOCAL_MESSAGES) == 4
        rows = profile.take()
    assert len(rows) == 1 and rows[0]['operation'] == ['tokenizer.template_count']
    assert rows[0]['calls'] == 1 and rows[0]['raised'] == 0


@pytest.mark.parametrize("kind", ["slow", "renderer", "encoder", "batch_encoder", "dropout"])
def test_local_unsupported_encoding_keeps_original_path(monkeypatch, kind):
    hf = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")

    class SlowCharacters(hf.PreTrainedTokenizer):
        def __init__(self):
            self.vocab = {char: index for index, char in enumerate(["[UNK]", *" abcdefghijklmnopqrstuvwxyzλ<>/"])}
            super().__init__(unk_token="[UNK]", chat_template=_LOCAL_TEMPLATE)

        @property
        def vocab_size(self):
            return len(self.vocab)

        def get_vocab(self):
            return dict(self.vocab, **self.added_tokens_encoder)

        def _tokenize(self, text, **kwargs):
            return list(text)

        def _convert_token_to_id(self, token):
            return self.vocab.get(token, 0)

        def _convert_id_to_token(self, index):
            return next((key for key, value in self.vocab.items() if value == index), "[UNK]")

    class CustomRenderer(hf.PreTrainedTokenizerFast):
        def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
            assert tokenize is True
            return [11, 12, 13]

    class CustomEncoder(hf.PreTrainedTokenizerFast):
        def __call__(self, text, **kwargs):
            self.calls = getattr(self, "calls", 0) + 1
            return hf.BatchEncoding({"input_ids": [1] * self.calls})

    class CustomBatchEncoder(hf.PreTrainedTokenizerFast):
        def _batch_encode_plus(self, *args, **kwargs):
            return super()._batch_encode_plus(*args, **kwargs)

    if kind == "slow":
        tokenizer = SlowCharacters()
    elif kind == "dropout":
        tokenizer = _native_tokenizer("bpe")
        tokenizer.backend_tokenizer.model.dropout = 0.5
    else:
        tokenizer = _native_tokenizer(cls={"renderer": CustomRenderer, "encoder": CustomEncoder,
                                          "batch_encoder": CustomBatchEncoder}[kind])
    counter = _local_counter(tokenizer)
    calls = _observe_encodings(monkeypatch)
    with request_count_reuse():
        values = [counter.count(_LOCAL_MESSAGES), counter.count(_LOCAL_MESSAGES)]
        assert _request_count_reuse.get() == ()
    if kind == "renderer":
        assert values == [3, 3]
    elif kind == "encoder":
        assert values == [1, 2]
    else:
        assert len(calls) == 2


def test_local_scope_lifetime_and_direct_calls(monkeypatch):
    counter = _local_counter(_native_tokenizer())
    expected = counter.count(_LOCAL_MESSAGES)
    calls = _observe_encodings(monkeypatch)
    with monkeypatch.context() as direct:
        direct.setattr("scripts.llm_solver.harness.local_tokenizer._deterministic_encoding_binding",
                       lambda _: pytest.fail("direct calls must not inspect binding metadata"))
        assert counter.count(_LOCAL_MESSAGES) == counter.count(_LOCAL_MESSAGES) == expected
    with request_count_reuse():
        assert counter.count(_LOCAL_MESSAGES) == counter.count(_LOCAL_MESSAGES) == expected
        with request_count_reuse():
            assert counter.count(_LOCAL_MESSAGES) == expected
        assert counter.count(_LOCAL_MESSAGES) == expected
        clear_request_count_reuse()
        assert counter.count(_LOCAL_MESSAGES) == expected
    assert _request_count_reuse.get() is None
    assert len(calls) == 5


@pytest.mark.parametrize("mutation", ["added_token", "replacement", "template", "message", "tools", "split_special_tokens"])
def test_local_reuse_rechecks_mutable_inputs(mutation):
    tokenizer = _native_tokenizer()
    counter = _local_counter(tokenizer)
    messages = [{"role": "user", "content": "unseen phrase"}]
    tools = [{"type": "function", "function": {"name": "read", "description": "short"}}]
    with request_count_reuse():
        counter.count(messages, tools)
        old_key = _request_count_reuse.get()[0]
        if mutation == "added_token":
            tokenizer.add_tokens(["unseen phrase"])
        elif mutation == "replacement":
            counter._tok = _native_tokenizer("wordpiece")
            assert old_key[2][0] is tokenizer
        elif mutation == "template":
            counter._chat_template = "hi hi hi hi"
        elif mutation == "message":
            messages[0]["content"] += " there there there"
        elif mutation == "tools":
            tools[0]["function"]["description"] = "long " * 20
        else:
            tokenizer.split_special_tokens = True
        value = counter.count(messages, tools)
        assert _request_count_reuse.get()[0] != old_key
    assert value == counter.count(messages, tools)


def test_local_dynamic_template_and_failed_template_recovery(monkeypatch):
    from datetime import datetime

    counter = _local_counter(_native_tokenizer())
    counter._chat_template = "{% for i in range(strftime_now('%S')|int) %}hi {% endfor %}"
    with operation_profile() as profile, request_count_reuse():
        for second in (1, 7):
            clock = type("Clock", (), {"now": staticmethod(lambda: datetime(2026, 10, 17, 12, 0, second))})
            monkeypatch.setattr("transformers.utils.chat_template_utils.datetime", clock)
            assert counter.count(_LOCAL_MESSAGES) == second
        counter._chat_template = "{{raise_exception('unsupported shape')}}"
        value = counter.count(_LOCAL_MESSAGES)
        assert counter.last_basis == "local_message_estimate" and _request_count_reuse.get() == ()
        rows = {tuple(row['operation']): row for row in profile.take()}
    assert rows[('tokenizer.render',)]['calls'] == 3 and rows[('tokenizer.render',)]['raised'] == 1
    assert rows[('tokenizer.encode',)]['calls'] == 2
    assert rows[('tokenizer.fallback_encode',)]['calls'] == 1
    assert rows[('tokenizer.fallback_encode',)]['raised'] == 0
    assert counter.count(_LOCAL_MESSAGES) == value
    counter._chat_template = None
    with request_count_reuse():
        assert counter.count(_LOCAL_MESSAGES) == 6
        assert counter.last_basis == "local_tokenizer_estimate"


def test_local_and_backend_counts_do_not_share_evidence():
    client = SimpleNamespace(
        cfg=SimpleNamespace(model="offline-test", request_dialect="llama", timeout_connect=1),
        client=SimpleNamespace(base_url="http://offline.invalid/v1"), profile=None)
    backend = BackendTokenCounter(client)
    backend._request_count = Mock(return_value={"input_tokens": 923})
    payload = {"model": "offline-test", "messages": _LOCAL_MESSAGES}
    local = _local_counter(_native_tokenizer())
    with request_count_reuse():
        assert backend.count_payload(payload) == backend.count_payload(payload) == 923
        assert backend.calls == 1
        assert local.count(_LOCAL_MESSAGES) == local.count(_LOCAL_MESSAGES) == 6
        assert backend.count_payload(payload) == 923 and backend.calls == 2
        assert local.count(_LOCAL_MESSAGES) == 6
    assert local.last_basis == "local_tokenizer_estimate"
    assert backend.last["count_basis"] == "backend_input_tokens"


@pytest.mark.parametrize("decay", [False, True])
def test_local_halflife_preflight_reuses_only_identical_projection(monkeypatch, decay):
    from copy import deepcopy
    from scripts.llm_solver.harness.context_strategies.halflife_context import HalfLifeContext
    from scripts.llm_solver.harness.request_counting import bind_session_counter

    messages = [{"role": "system", "content": "hi there"}]
    messages += [{"role": role, "content": "hi there " * 30}
                 for _ in range(12) for role in ("assistant", "tool")]

    def make_session():
        counter = _local_counter(_native_tokenizer())
        context = HalfLifeContext(context_size=8192, context_limit_tokens=32 if decay else 0,
                                  cap_7_chars=100, cap_15_chars=80)
        context.replace_all_messages(deepcopy(messages))
        context._turn_count = 12
        session = SimpleNamespace(context=context,
            cfg=SimpleNamespace(context_size=8192, max_tokens=64, context_fill_ratio=.95,
                                tokenizer_id="offline-test"),
            client=object(), model_tool_schemas=[], _session_number=1, _current_turn=12,
            _emit=lambda *args, **kwargs: None)
        bind_session_counter(session, counter)
        return session

    baseline, candidate = make_session(), make_session()
    expected = _preflight_estimate(baseline)
    calls = _observe_encodings(monkeypatch)
    with request_count_reuse():
        assert _preflight_estimate(candidate) == expected
    assert len(calls) == (2 if decay else 1)
    assert candidate.context.get_messages() == baseline.context.get_messages()
    assert candidate.context.snapshot_messages() == messages
    assert candidate._preflight_count_authoritative is False


def test_missing_chat_renderer_preserves_message_estimate():
    class MissingRenderer:
        def apply_chat_template(self, *args, **kwargs):
            raise ImportError("chat template renderer unavailable")

        def encode(self, text, **kwargs):
            return list(text)

    counter = _local_counter(MissingRenderer())
    expected = counter.count(_LOCAL_MESSAGES)
    with request_count_reuse():
        assert counter.count(_LOCAL_MESSAGES) == expected
        assert counter.last_basis == "local_message_estimate"
        assert _request_count_reuse.get() == ()
