"""API tests with the model mocked out."""

from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fake News" in response.text


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(app_module, "predict_news", lambda text: ("FAKE", 88.5))
    response = client.get("/predict", params={"news": "Aliens run the government"})
    assert response.status_code == 200
    assert "FAKE" in response.text
    assert "88.5" in response.text


def test_predict_escapes_html_input(monkeypatch):
    monkeypatch.setattr(app_module, "predict_news", lambda text: ("REAL", 60.0))
    response = client.get("/predict", params={"news": "<script>alert(1)</script>"})
    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
