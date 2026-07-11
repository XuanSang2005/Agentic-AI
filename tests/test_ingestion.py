"""GATE Phase 4a (hành vi mới): ingest batch → Postgres + atomic reindex.

Chạy khi DATA_SOURCE=postgres + DB đã seed (như gate Phase 3); mặc định skip.
POI test dùng prefix ZTEST, teardown xoá sạch + clear cache để các test sau
(equivalence/smoke) thấy DB nguyên trạng.
"""
from __future__ import annotations

import pytest

from src import config

pytestmark = pytest.mark.skipif(
    config.DATA_SOURCE != "postgres",
    reason="cần DATA_SOURCE=postgres + Postgres đã seed",
)

_ZPREFIX = "ZTEST"

_BATCH = [
    {
        "poi_id": "ZTEST01", "name": "Ztestcafe Nguyễn Huệ", "category": "Quán cà phê",
        "city": "TP.HCM", "district": "Quận 1", "address": "1 Nguyễn Huệ, Quận 1",
        "lat": 10.774, "lon": 106.704, "rating": 4.5, "review_count": 10,
        "popularity_score": 50.0, "price_level": 2, "opening_hours": "07:00-22:00",
        "attributes": ["yên tĩnh", "wifi"], "tags": ["coffee"],
        "description": "Quán cà phê thử nghiệm ingestion",
    },
    {
        "poi_id": "ZTEST02", "name": "Ztestcafe Đồng Khởi", "category": "Quán cà phê",
        "city": "TP.HCM", "district": "Quận 1", "lat": 10.776, "lon": 106.702,
        "rating": 4.0,
    },
]


@pytest.fixture(scope="module")
def client():
    try:
        import psycopg
        with psycopg.connect(config.database_url()) as conn:
            conn.execute("SELECT 1 FROM pois LIMIT 1")
    except Exception as e:
        pytest.skip(f"Postgres không sẵn sàng: {e}")

    from fastapi.testclient import TestClient
    from src.api import main as api_main

    npy_before = set(config.EMBEDDING_CACHE_DIR.glob("*.npy"))
    with TestClient(api_main.app) as c:
        yield c

    # Teardown: trả DB + mọi data-cache + embedding cache về nguyên trạng
    # để các module test sau (equivalence/smoke) thấy hệ như chưa ingest.
    import psycopg
    from src.search import _clear_data_caches
    with psycopg.connect(config.database_url()) as conn:
        conn.execute("DELETE FROM pois WHERE poi_id LIKE %s", (_ZPREFIX + "%",))
    _clear_data_caches()
    for f in set(config.EMBEDDING_CACHE_DIR.glob("*.npy")) - npy_before:
        f.unlink()


def _db_rows(where_like: str):
    import psycopg
    with psycopg.connect(config.database_url()) as conn:
        return conn.execute(
            "SELECT poi_id, row_order, status FROM pois WHERE poi_id LIKE %s"
            " ORDER BY poi_id", (where_like,)).fetchall()


def test_b_push_batch_accepted_and_persisted(client):
    """Gate b: push batch → accepted đúng, Postgres có record (row_order nối tiếp,
    status pending), chỉ POI mới bị encode."""
    n_before = client.get("/health").json()["pois"]
    resp = client.post("/admin/pois/batch", json=_BATCH)
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["received"] == 2 and report["accepted"] == 2
    assert report["accepted_ids"] == ["ZTEST01", "ZTEST02"]
    assert report["rejected"] == []
    # Reindex 1 lần cho cả batch; POI cũ KHÔNG re-encode — chỉ 2 doc mới
    assert report["reindex"]["pois"] == n_before + 2
    assert report["reindex"]["encoded_new"] == 2

    rows = _db_rows(_ZPREFIX + "%")
    assert [r[0] for r in rows] == ["ZTEST01", "ZTEST02"]
    assert all(r[2] == "pending" for r in rows), "status mới phải là pending (4b verify)"
    orders = sorted(r[1] for r in rows)
    assert orders == [n_before, n_before + 1], "row_order phải nối tiếp max hiện có"


def test_c_new_poi_visible_in_search(client):
    """Gate c: sau push, serve THẤY POI mới qua atomic swap (không restart)."""
    resp = client.get("/v1/search", params={"q": "ztestcafe quận 1", "limit": 5})
    ids = [r["id"] for r in resp.json()["results"]]
    assert any("ZTEST01" in i or "ZTEST02" in i for i in ids), f"POI mới không thấy: {ids}"
    assert client.get("/health").json()["pois"] == 113


def test_d_idempotent_reingest(client):
    """Gate d: push lại cùng batch → upsert, không trùng, row_order giữ nguyên."""
    before = _db_rows(_ZPREFIX + "%")
    resp = client.post("/admin/pois/batch", json=_BATCH)
    report = resp.json()
    assert report["accepted"] == 2
    assert report["reindex"]["pois"] == 113          # không phình
    assert report["reindex"]["encoded_new"] == 0     # document không đổi → 0 encode
    assert _db_rows(_ZPREFIX + "%") == before        # row_order/status y nguyên


def test_e_partial_batch_commits_valid_rejects_invalid(client):
    """Gate e: 2 đúng + 1 méo → 2 accepted (commit), 1 rejected kèm lý do.
    (LỰA CHỌN: reject ở tầng validate → record hợp lệ vẫn commit — validate là
    deterministic per-record; rollback sạch chỉ dành cho lỗi DB, xem gate f.)"""
    batch = [
        dict(_BATCH[0], poi_id="ZTEST03", name="Ztestcafe Ba"),
        {"poi_id": "ZTEST04", "category": "Quán cà phê", "city": "TP.HCM"},  # thiếu name
        dict(_BATCH[0], poi_id="ZTEST05", name="Ztestcafe Năm"),
    ]
    report = client.post("/admin/pois/batch", json=batch).json()
    assert report["accepted"] == 2 and report["accepted_ids"] == ["ZTEST03", "ZTEST05"]
    assert len(report["rejected"]) == 1
    rej = report["rejected"][0]
    assert rej["index"] == 1 and rej["poi_id"] == "ZTEST04"
    assert any(e["field"] == "name" for e in rej["errors"])
    assert {r[0] for r in _db_rows("ZTEST0%")} == {"ZTEST01", "ZTEST02", "ZTEST03", "ZTEST05"}


def test_f_db_error_rolls_back_whole_batch_and_no_reindex(client):
    """Gate f: lỗi DB GIỮA batch (record 2 tràn INT4 — qua được Pydantic, chết ở
    Postgres) → rollback sạch cả batch, index không chứa record nào của batch."""
    n_before = client.get("/health").json()["pois"]
    batch = [
        dict(_BATCH[0], poi_id="ZTEST06", name="Ztestcafe Sáu"),
        dict(_BATCH[0], poi_id="ZTEST07", name="Ztestcafe Bảy",
             review_count=2**40),  # vượt INTEGER — psycopg raise giữa transaction
    ]
    resp = client.post("/admin/pois/batch", json=batch)
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "ingestion_db_error"
    # Atomicity DB ↔ index: ZTEST06 không ở DB, không ở index
    assert _db_rows("ZTEST06") == [] and _db_rows("ZTEST07") == []
    assert client.get("/health").json()["pois"] == n_before
    ids = [r["id"] for r in
           client.get("/v1/search", params={"q": "ztestcafe sáu"}).json()["results"]]
    assert not any("ZTEST06" in i for i in ids)
