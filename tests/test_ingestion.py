"""GATE Phase 4a (b–f) + 4b (i, k): ingest batch → verify → Postgres + atomic reindex.

Chạy khi DATA_SOURCE=postgres + DB đã seed (như gate Phase 3); mặc định skip.
AWS được MOCK (FakeGeo inject qua geocode.set_client) — không mạng thật.
POI test dùng prefix ZTEST, teardown xoá sạch + clear cache để các test sau
(equivalence/smoke) thấy DB nguyên trạng.
"""
from __future__ import annotations

import pytest

from src import config
from tests.test_verify import FakeGeo

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
        "city": "TP.HCM", "district": "Quận 1", "address": "2 Đồng Khởi, Quận 1",
        "lat": 10.776, "lon": 106.702, "rating": 4.0,
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
    from src.verify import geocode

    # Mock AWS: happy-path (echo BiasPosition, Overall 0.98) — statuses tất định,
    # không mạng thật; test i/k override hành vi per-address ngay trên fake này.
    geocode.set_client(FakeGeo())

    import os
    import time as _time

    # Admin auth (Phase 6): bật ingestion bằng token test, gắn header mặc định.
    os.environ["ADMIN_TOKEN"] = "test-admin-token"

    npy_before = set(config.EMBEDDING_CACHE_DIR.glob("*.npy"))
    with TestClient(api_main.app) as c:
        c.headers.update({"Authorization": "Bearer test-admin-token"})
        # Lifespan build NỀN (Phase 6) — chờ service + poller lên rồi TẮT poller:
        # test điều khiển reload bằng poll-ONCE (deterministic), không vòng lặp thật.
        svc = api_main._get_service()  # block tới khi build xong
        deadline = _time.time() + 30
        while svc._poller is None and _time.time() < deadline:
            _time.sleep(0.05)
        svc.stop_version_poller()
        yield c
    geocode.set_client(None)
    os.environ.pop("ADMIN_TOKEN", None)

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
    status theo verify — mock happy → verified), chỉ POI mới bị encode."""
    n_before = client.get("/health").json()["pois"]
    resp = client.post("/admin/pois/batch", json=_BATCH)
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["received"] == 2 and report["accepted"] == 2
    assert report["accepted_ids"] == ["ZTEST01", "ZTEST02"]
    assert report["rejected"] == []
    assert report["verified"] == 2 and report["unverified"] == 0
    # Reindex 1 lần cho cả batch; POI cũ KHÔNG re-encode — chỉ 2 doc mới
    assert report["reindex"]["pois"] == n_before + 2
    assert report["reindex"]["encoded_new"] == 2

    rows = _db_rows(_ZPREFIX + "%")
    assert [r[0] for r in rows] == ["ZTEST01", "ZTEST02"]
    assert all(r[2] == "verified" for r in rows), "mock happy → status verified thật"
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


def test_i_aws_error_flags_unverified_but_batch_survives(client):
    """Gate i (4b): AWS ném exception cho 1 record → record đó VẪN ghi với
    status='unverified' + reason lỗi; record khác verify bình thường."""
    from src.verify import geocode
    fake = geocode.get_client()
    fake.rules["8 Lê Lợi, Quận 1"] = TimeoutError("connect timeout")

    batch = [
        dict(_BATCH[0], poi_id="ZTEST08", name="Ztestcafe Tám",
             address="8 Nguyễn Trãi, Quận 1"),
        dict(_BATCH[0], poi_id="ZTEST09", name="Ztestcafe Chín",
             address="8 Lê Lợi, Quận 1"),
    ]
    report = client.post("/admin/pois/batch", json=batch).json()
    assert report["accepted"] == 2
    assert report["verified"] == 1 and report["unverified"] == 1
    by_id = {v["poi_id"]: v for v in report["verification"]}
    assert by_id["ZTEST08"]["status"] == "verified"
    assert by_id["ZTEST09"]["status"] == "unverified"
    assert "verify failed:" in by_id["ZTEST09"]["reason"] and "timeout" in by_id["ZTEST09"]["reason"]
    assert {(r[0], r[2]) for r in _db_rows("ZTEST08")} == {("ZTEST08", "verified")}
    assert {(r[0], r[2]) for r in _db_rows("ZTEST09")} == {("ZTEST09", "unverified")}


def test_k_mixed_batch_all_commit_flag_not_reject(client):
    """Gate k (4b): 1 verified + 1 unverified(score thấp) + 1 lỗi-AWS →
    cả 3 đều COMMIT (flag không reject), report phân loại đúng từng cái."""
    from src.verify import geocode
    fake = geocode.get_client()
    fake.rules["11 Hai Bà Trưng, Quận 1"] = {"overall": 0.4}
    fake.rules["12 Pasteur, Quận 1"] = ConnectionError("network unreachable")

    batch = [
        dict(_BATCH[0], poi_id="ZTEST10", name="Ztestcafe Mười",
             address="10 Nguyễn Du, Quận 1"),
        dict(_BATCH[0], poi_id="ZTEST11", name="Ztestcafe Mười Một",
             address="11 Hai Bà Trưng, Quận 1"),
        dict(_BATCH[0], poi_id="ZTEST12", name="Ztestcafe Mười Hai",
             address="12 Pasteur, Quận 1"),
    ]
    report = client.post("/admin/pois/batch", json=batch).json()
    assert report["accepted"] == 3 and report["rejected"] == []
    assert report["verified"] == 1 and report["unverified"] == 2
    by_id = {v["poi_id"]: v for v in report["verification"]}
    assert by_id["ZTEST10"]["status"] == "verified"
    assert by_id["ZTEST11"]["status"] == "unverified"
    assert "match score 0.40" in by_id["ZTEST11"]["reason"]
    assert by_id["ZTEST12"]["status"] == "unverified"
    assert by_id["ZTEST12"]["reason"].startswith("verify failed:")
    statuses = {r[0]: r[2] for r in _db_rows("ZTEST1%")}
    assert statuses == {"ZTEST10": "verified", "ZTEST11": "unverified",
                        "ZTEST12": "unverified"}


def test_m_batch_too_large_rejected_before_verify(client, monkeypatch):
    """Harden 1: batch > max_batch_size → 413 NGAY, không gọi AWS, không chạm DB."""
    import dataclasses

    from src import config as cfg
    from src.verify import geocode

    s = cfg.settings()
    small = dataclasses.replace(s, ingestion=dataclasses.replace(
        s.ingestion, max_batch_size=2))
    monkeypatch.setattr("src.config.settings", lambda: small)

    def _must_not_be_called(*a, **kw):
        raise AssertionError("verify_batch KHÔNG được gọi khi batch quá cỡ")
    monkeypatch.setattr(geocode, "verify_batch", _must_not_be_called)

    n_db_before = len(_db_rows("%"))
    batch = [dict(_BATCH[0], poi_id=f"ZTESTBIG{i}") for i in range(3)]  # 3 > 2
    resp = client.post("/admin/pois/batch", json=batch)
    assert resp.status_code == 413
    err = resp.json()["error"]
    assert err["code"] == "batch_too_large"
    assert err["details"] == {"received": 3, "max": 2}
    assert len(_db_rows("%")) == n_db_before  # DB không đổi


def test_n_auth_constant_time_same_behavior(client, monkeypatch):
    """Harden 2: compare_digest KHÔNG đổi hành vi — token đúng pass, sai/thiếu 401."""
    monkeypatch.setenv("TASCO_BEARER_TOKEN", "sekret-bearer")
    monkeypatch.setenv("TASCO_API_KEY", "sekret-key")

    ok = client.get("/v1/search", params={"q": "cafe"},
                    headers={"Authorization": "Bearer sekret-bearer"})
    assert ok.status_code == 200
    ok2 = client.get("/v1/search", params={"q": "cafe"},
                     headers={"X-API-Key": "sekret-key"})
    assert ok2.status_code == 200
    bad = client.get("/v1/search", params={"q": "cafe"},
                     headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401
    missing = client.get("/v1/search", params={"q": "cafe"})
    assert missing.status_code == 401

    monkeypatch.delenv("TASCO_BEARER_TOKEN")
    monkeypatch.delenv("TASCO_API_KEY")
    assert client.get("/v1/search", params={"q": "cafe"}).status_code == 200  # mock mode


def test_o_reindex_fail_after_commit_returns_200_with_warning(client, monkeypatch):
    """Harden 3: reindex ném lỗi SAU commit → 200 + warning (data ĐÃ persist),
    KHÔNG 500, record vẫn nằm trong DB."""
    from src.api import main as api_main

    def _boom():
        raise RuntimeError("encode POI mới fail (giả lập)")
    monkeypatch.setattr(api_main._get_service(), "reindex", _boom)

    batch = [dict(_BATCH[0], poi_id="ZTEST13", name="Ztestcafe Mười Ba")]
    resp = client.post("/admin/pois/batch", json=batch)
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["accepted"] == 1
    assert report["reindex"]["status"] == "failed"
    assert "đã commit" in report["reindex"]["warning"]
    assert "encode POI mới fail" in report["reindex"]["reason"]
    # DB vẫn giữ record — reindex fail không rollback commit
    assert [r[0] for r in _db_rows("ZTEST13")] == ["ZTEST13"]


def test_p_advisory_lock_serializes_and_releases(client):
    """Harden 4: 2 ingest tuần tự chạy đúng qua advisory lock (row_order nối
    tiếp), lock xact-scoped nhả sạch sau commit (pg_locks trống)."""
    import psycopg

    r1 = client.post("/admin/pois/batch",
                     json=[dict(_BATCH[0], poi_id="ZTEST14", name="Ztestcafe Mười Bốn")])
    r2 = client.post("/admin/pois/batch",
                     json=[dict(_BATCH[0], poi_id="ZTEST15", name="Ztestcafe Mười Lăm")])
    assert r1.status_code == 200 and r2.status_code == 200

    rows = {r[0]: r[1] for r in _db_rows("ZTEST1%")}
    assert rows["ZTEST15"] == rows["ZTEST14"] + 1, "row_order phải nối tiếp qua 2 batch"
    with psycopg.connect(config.database_url()) as conn:
        n_locks = conn.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'").fetchone()[0]
    assert n_locks == 0, "advisory lock phải tự nhả khi transaction kết thúc"


# --- Phase 5: data_version dùng chung + reload đa-instance ---


def _db_version() -> int:
    from src.data_loader import current_data_version
    return current_data_version()


def test_q_version_bumps_with_commit_not_with_rollback(client):
    """Gate 5a: ingest commit → version +1 đúng 1 lần/batch; batch rollback
    (lỗi DB) → version KHÔNG tăng; report + _loaded_version mang version mới."""
    from src.api import main as api_main

    v0 = _db_version()
    report = client.post("/admin/pois/batch", json=[
        dict(_BATCH[0], poi_id="ZTEST16", name="Ztestcafe Mười Sáu"),
        dict(_BATCH[0], poi_id="ZTEST17", name="Ztestcafe Mười Bảy"),
    ]).json()
    assert _db_version() == v0 + 1, "1 batch (2 record) = bump đúng 1 lần"
    assert report["data_version"] == v0 + 1
    assert api_main._get_service().loaded_version == v0 + 1

    # Batch lỗi DB (INT4 overflow) → rollback → version giữ nguyên
    resp = client.post("/admin/pois/batch", json=[
        dict(_BATCH[0], poi_id="ZTEST18", review_count=2**40)])
    assert resp.status_code == 500
    assert _db_version() == v0 + 1, "rollback thì version KHÔNG tăng (atomic với data)"


def test_r_poll_once_reloads_when_remote_version_newer(client):
    """Gate 5b: 'instance khác' ghi DB + bump version (gọi thẳng upsert_pois,
    không qua endpoint → instance này KHÔNG tự reindex) → poll-once phát hiện
    version mới → reload, thấy POI mới, _loaded_version cập nhật."""
    from src import ingestion
    from src.api import main as api_main

    svc = api_main._get_service()
    n_before = svc.n_pois
    valid, rejected = ingestion.validate_batch([
        dict(_BATCH[0], poi_id="ZTEST19", name="Ztestcafe Mười Chín")])
    assert not rejected
    new_version = ingestion.upsert_pois(valid, ["verified"])  # ghi thẳng DB
    assert svc.loaded_version < new_version  # instance này chưa biết gì

    out = svc.check_and_reload_once()
    assert out["reloaded"] is True and out["version"] == new_version
    assert svc.loaded_version == new_version
    assert svc.n_pois == n_before + 1  # index đã thấy POI của "instance khác"
    ids = [r["id"] for r in
           client.get("/v1/search", params={"q": "ztestcafe mười chín"}).json()["results"]]
    assert any("ZTEST19" in i for i in ids)


def test_s_poll_once_skips_when_version_unchanged(client, monkeypatch):
    """Gate 5c: version không đổi → poll-once KHÔNG rebuild (reindex không
    được gọi — không tốn compute)."""
    from src.api import main as api_main

    svc = api_main._get_service()

    def _must_not_be_called():
        raise AssertionError("reindex KHÔNG được gọi khi version không đổi")
    monkeypatch.setattr(svc, "reindex", _must_not_be_called)

    out = svc.check_and_reload_once()
    assert out["reloaded"] is False and out["version"] == svc.loaded_version


def test_t_search_serves_old_snapshot_during_reload(client, monkeypatch):
    """Gate 5d: reload đang chạy (kẹt ở load_pois) → search vẫn trả kết quả
    bình thường từ snapshot cũ, không lỗi, không nửa vời."""
    import threading

    from src import search as search_mod
    from src.api import main as api_main

    svc = api_main._get_service()
    release = threading.Event()
    real_load = search_mod.load_pois

    def slow_load():
        release.wait(timeout=30)  # kẹt reload tại đây, deterministic
        return real_load()
    monkeypatch.setattr(search_mod, "load_pois", slow_load)

    before = [r["id"] for r in
              client.get("/v1/search", params={"q": "quán cà phê yên tĩnh"}).json()["results"]]
    t = threading.Thread(target=svc.reindex)
    t.start()
    try:
        # reload đang kẹt giữa chừng — search vẫn phục vụ snapshot cũ y nguyên
        during = [r["id"] for r in
                  client.get("/v1/search", params={"q": "quán cà phê yên tĩnh"}).json()["results"]]
        assert during == before
    finally:
        release.set()
        t.join(timeout=60)
    assert not t.is_alive(), "reindex phải hoàn tất sau khi thả khoá"


def test_u_poller_thread_daemon_and_clean_shutdown(client):
    """Gate 5e: poller là daemon (không chặn shutdown) và tắt sạch, không rò
    thread. Lifespan đã start 1 poller — kiểm nó rồi stop/start lại thủ công."""
    import threading

    from src.api import main as api_main

    svc = api_main._get_service()
    svc.stop_version_poller()  # tắt poller của lifespan (nếu đang chạy)
    svc.start_version_poller(interval_seconds=60)  # interval dài — không fire trong test
    poller = next(t for t in threading.enumerate() if t.name == "tasco-version-poller")
    assert poller.daemon is True
    svc.stop_version_poller()
    assert not poller.is_alive(), "stop phải join sạch thread"
    assert not any(t.name == "tasco-version-poller" for t in threading.enumerate())
    svc.start_version_poller()  # trả lại trạng thái như lifespan để teardown bình thường


def test_v_ready_200_when_built_and_db_reachable(client):
    """Gate 6a (nửa sau): service đã build + Postgres chạm được → /ready 200
    kèm pois + data_version; /health cũng 200 với ready=true."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready" and body["pois"] >= 111
    assert body["data_version"] >= 1
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["ready"] is True
