import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config = {
        "models": {
            "test-model": {
                "api_base": "https://example.com",
                "api_key": "sk-secret",
                "upstream_model": "test-model",
            }
        }
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    from llm_proxy import config_loader
    from llm_proxy.state import init_state

    monkeypatch.setattr(config_loader, "CONFIG_PATH", config_path)
    init_state(config_loader.load_config())

    from llm_proxy.main import app

    yield TestClient(app), config_path


def test_update_model_preserves_api_key_when_empty(client):
    tc, config_path = client
    r = tc.put("/api/models/test-model", json={"display_name": "Updated", "api_key": ""})
    assert r.status_code == 200

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["models"]["test-model"]["api_key"] == "sk-secret"
    assert saved["models"]["test-model"]["display_name"] == "Updated"


def test_update_model_sets_api_key_when_provided(client):
    tc, config_path = client
    r = tc.put("/api/models/test-model", json={"api_key": "sk-new"})
    assert r.status_code == 200

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["models"]["test-model"]["api_key"] == "sk-new"


def test_update_model_allows_empty_api_key_for_new_model(client):
    tc, config_path = client
    r = tc.put("/api/models/new-model", json={"api_base": "https://example.com", "api_key": ""})
    assert r.status_code == 200

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["models"]["new-model"]["api_key"] == ""


def test_create_model_without_api_key_field_fails_reload(client):
    """UI 新建模型若省略 api_key 字段，reload 会因 build_model_map KeyError 失败。"""
    tc, config_path = client
    r = tc.put("/api/models/new-model", json={"api_base": "https://example.com"})
    assert r.status_code == 500
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["models"]["new-model"]["api_base"] == "https://example.com"
    assert "api_key" not in saved["models"]["new-model"]
