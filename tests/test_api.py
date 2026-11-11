from fastapi.testclient import TestClient

import vietlott_ml.api as api

client = TestClient(api.app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_lotteries_empty_data(monkeypatch):
    monkeypatch.setattr(api, "load_canonical", lambda lottery: [])
    response = client.get("/api/lotteries")
    assert response.status_code == 200
    body = response.json()
    assert {item["key"] for item in body} == {"mega645", "power655"}
    assert all(item["draw"] is None for item in body)


def test_latest_invalid_lottery():
    response = client.get("/api/latest/khong_ton_tai")
    assert response.status_code == 404


def test_predict_random(monkeypatch):
    fake = {
        "lottery": "mega645",
        "numbers": [1, 2, 3, 4, 5, 6],
        "model": "random",
    }
    monkeypatch.setattr(api, "generate_numbers", lambda *a, **k: fake)
    response = client.post(
        "/api/predict", json={"lottery": "mega645", "model": "random"}
    )
    assert response.status_code == 200
    assert response.json()["numbers"] == [1, 2, 3, 4, 5, 6]


def test_predict_invalid_model():
    response = client.post(
        "/api/predict", json={"lottery": "mega645", "model": "khong_ton_tai"}
    )
    assert response.status_code == 400


def test_metrics_invalid_lottery():
    response = client.get("/api/metrics/khong_ton_tai")
    assert response.status_code == 404
