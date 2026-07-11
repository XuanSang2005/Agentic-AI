"""GATE Phase 6 (ops): liveness/readiness, admin auth khoá-mặc-định, JSON logging.

Không cần DB, không build index: dùng client "lạnh" (không chạy lifespan,
_service ép về None) — chính là trạng thái đang-warmup mà probe phải phân biệt.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.logging_config import JsonFormatter


@pytest.fixture()
def cold_client(monkeypatch):
    """Client KHÔNG lifespan + _service=None — mô phỏng process vừa lên, index chưa build."""
    monkeypatch.setattr(api_main, "_service", None)
    return TestClient(api_main.app)


def test_health_is_liveness_200_before_index_ready(cold_client):
    """Gate a: /health 200 NGAY cả khi index chưa build (không trigger build)."""
    resp = cold_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["ready"] is False
    assert "pois" not in body               # chưa build thì không có gì để đếm
    assert api_main._service is None        # probe KHÔNG được trigger build


def test_ready_503_during_warmup(cold_client):
    """Gate a: /ready 503 warming_up lúc startup — LB chưa gửi traffic tới."""
    resp = cold_client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "warming_up"


def test_admin_locked_when_no_token(cold_client, monkeypatch):
    """Gate b: ADMIN_TOKEN trống → 503 admin_disabled (KHÔNG mở như search mock-mode)."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    resp = cold_client.post("/admin/pois/batch", json=[])
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "admin_disabled"


def test_admin_wrong_or_missing_token_401(cold_client, monkeypatch):
    """Gate b+e: token đặt nhưng header sai/thiếu → 401 (so constant-time)."""
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret-admin")
    assert cold_client.post("/admin/pois/batch", json=[]).status_code == 401
    resp = cold_client.post("/admin/pois/batch", json=[],
                            headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_admin_auth_runs_before_source_check(cold_client, monkeypatch):
    """Gate b: token ĐÚNG → qua auth, chặn tiếp ở gate sau (xlsx → 503
    ingestion_unavailable) — chứng minh auth đứng TRƯỚC mọi xử lý."""
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret-admin")
    monkeypatch.setattr("src.config.DATA_SOURCE", "xlsx")
    resp = cold_client.post("/admin/pois/batch", json=[],
                            headers={"Authorization": "Bearer s3cret-admin"})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "ingestion_unavailable"


def test_request_middleware_logs_latency_and_status(cold_client, caplog):
    """Gate c: mỗi request phát log có method/path/status/latency_ms;
    KHÔNG log query string (lat/lon người dùng)."""
    with caplog.at_level(logging.INFO, logger="tasco.api"):
        cold_client.get("/health", params={"lat": 10.7, "lon": 106.7})
    rec = next(r for r in caplog.records
               if r.name == "tasco.api" and r.getMessage() == "request")
    assert rec.method == "GET" and rec.path == "/health"
    assert rec.status == 200 and isinstance(rec.latency_ms, float)
    assert "lat" not in JsonFormatter().format(rec).replace("latency_ms", "")


def test_json_formatter_emits_valid_json_with_extras():
    """Gate c: formatter ra JSON hợp lệ, extra fields nằm phẳng trong payload."""
    rec = logging.LogRecord("tasco.test", logging.INFO, __file__, 1,
                            "sự kiện", (), None)
    rec.latency_ms = 12.5
    rec.status = 200
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "sự kiện" and payload["level"] == "INFO"
    assert payload["latency_ms"] == 12.5 and payload["status"] == 200


def test_no_print_left_in_src():
    """Gate c: không còn print() trong src/ — mọi output vận hành qua logging."""
    src = Path(__file__).resolve().parent.parent / "src"
    offenders = [str(p) for p in src.rglob("*.py")
                 if re.search(r"\bprint\(", p.read_text(encoding="utf-8"))]
    assert not offenders, f"print() còn sót trong: {offenders}"
