"""Backend metadata follows the active transport and preserves failed probes."""
import hashlib
import json

import httpx
import openai
import pytest

from _config_helpers import make_config
from scripts.llm_solver._main_helpers import _write_server_metadata
from scripts.llm_solver.server.client import LlamaClient


@pytest.fixture(autouse=True)
def forbid_separate_http_client(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *args, **kwargs:
                        pytest.fail("metadata bypassed the mocked SDK transport"))


def client_with_transport(handler):
    cfg = make_config(base_url="http://declared.invalid/v1", timeout_connect=3)
    client = LlamaClient(cfg)
    client.client.close()
    client.client = openai.OpenAI(
        base_url="http://active.invalid/route/v1", api_key="fixture-key",
        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client


def test_metadata_uses_active_authenticated_route_and_saved_snapshot(tmp_path):
    paths = []

    def handler(request):
        assert request.url.host == "active.invalid"
        assert request.headers["authorization"] == "Bearer fixture-key"
        assert request.extensions["timeout"]["read"] == 3
        paths.append(request.url.path)
        if request.url.path.endswith("/props"):
            return httpx.Response(200, json={"model_path": "observed-model"})
        if request.url.path.endswith("/slots"):
            return httpx.Response(200, json=[{"n_ctx": 4096}])
        return httpx.Response(200, json={"data": [{"id": "reported-alias"}]})

    client = client_with_transport(handler)
    path, digest = _write_server_metadata(tmp_path, client)
    assert paths == ["/route/props", "/route/slots", "/route/v1/models"]
    raw = path.read_bytes()
    snapshot = json.loads(raw)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert snapshot["base_url"] == "http://active.invalid/route/v1/"
    assert snapshot["root_url"] == "http://active.invalid/route"
    assert snapshot["capture_status"] == "available"
    assert snapshot["endpoints"]["/props"]["json"]["model_path"] == "observed-model"
    assert snapshot["captured_at"]
    assert "fixture-key" not in raw.decode()


@pytest.mark.parametrize("failure", ["unauthorized", "invalid_json", "connection"])
def test_unavailable_metadata_is_recorded_without_inventing_backend_facts(tmp_path, failure):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if failure == "connection":
            raise httpx.ConnectError("fixture failure", request=request)
        if failure == "unauthorized":
            return httpx.Response(401, json={"error": {"message": "unavailable"}})
        return httpx.Response(200, text="not JSON")

    client = client_with_transport(handler)
    path, digest = _write_server_metadata(tmp_path, client)
    assert len(paths) == 3
    assert path is not None and digest
    snapshot = json.loads(path.read_text())
    assert snapshot["capture_status"] == "unavailable"
    assert all("json" not in record for record in snapshot["endpoints"].values())
    if failure == "unauthorized":
        assert all(record["status_code"] == 401 for record in snapshot["endpoints"].values())


def test_failed_endpoint_does_not_hide_other_available_metadata():
    def handler(request):
        if request.url.path.endswith("/props"):
            return httpx.Response(404, json={"error": {"message": "not supported"}})
        return httpx.Response(200, json={"data": []})

    client = client_with_transport(handler)
    snapshot = client.query_server_metadata()
    assert snapshot["capture_status"] == "available"
    assert snapshot["endpoints"]["/props"]["status_code"] == 404
    assert snapshot["endpoints"]["/v1/models"]["json"] == {"data": []}
