"""FastAPI service — đúng Search API contract trong docs/tasco_api.pdf.

- GET /v1/search (+ alias /search, /v1/geocode-search) → {query, results[], meta}
- GET /health
- Auth pluggable theo PDF: đặt env TASCO_BEARER_TOKEN / TASCO_API_KEY thì service
  YÊU CẦU header tương ứng (Authorization: Bearer … hoặc X-API-Key: …);
  không đặt → mock mode chấp nhận mọi request. KHÔNG hardcode credentials.
- Deterministic offline: toàn bộ chạy trên data/ bundled + embedding .npy cache,
  0 LLM call, không cần network — "deterministic mock data for tests and demos".

Chạy: make api  (uvicorn src.api.main:app --port 8000)
"""
from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

from typing import Optional

from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from src import config
from src.api.dto import (ErrorBody, ErrorResponse, SearchMeta, SearchResponse,
                         to_place_result)
from src.data_loader import normalize_vi
from src.logging_config import setup_logging
from src.search import SearchService

MAX_LIMIT = config.settings().search.api_max_limit  # "max recommended 20" theo PDF

setup_logging()
logger = logging.getLogger("tasco.api")

_service: SearchService | None = None
_service_lock = threading.Lock()


def _get_service() -> SearchService:
    """Build service đúng 1 lần (thread-safe) — caller CHỜ nếu đang build."""
    global _service
    with _service_lock:
        if _service is None:
            _service = SearchService()
        return _service


def _service_ready() -> bool:
    """Index đã build + warmup xong chưa — KHÔNG trigger build (probe phải rẻ)."""
    return _service is not None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build NỀN (daemon thread) để /health và /ready trả lời được NGAY trong lúc
    # warmup ~7s — LB phân biệt "sống" vs "sẵn sàng". Request search đến sớm
    # sẽ chờ ở _get_service (lock) thay vì lỗi.
    def _build() -> None:
        svc = _get_service()
        svc.start_version_poller()  # Phase 5 — no-op khi DATA_SOURCE=xlsx
        logger.info("service_ready", extra={"pois": svc.n_pois,
                                            "data_version": svc.loaded_version})
    threading.Thread(target=_build, daemon=True, name="tasco-startup").start()
    yield
    if _service is not None:
        _service.stop_version_poller()


app = FastAPI(
    title="Tasco Maps — AI Semantic Search & Ranking (Track 2)",
    version="1.0.0",
    description="Search-and-ranking service cho POI tiếng Việt. "
                "Contract: docs/tasco_api.pdf. Deterministic, offline, 0 LLM call.",
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """Đo latency mọi request. Chỉ log PATH — KHÔNG log query string
    (?lat/lon là vị trí người dùng, không đưa vào log)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    logger.info("request", extra={
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    })
    return response


def _error(status: int, code: str, message: str, request: Request,
           details: Optional[dict] = None) -> JSONResponse:
    """ErrorResponse đúng shape PDF; echo X-Request-Id nếu client gửi."""
    rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details),
                         requestId=rid)
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return _error(400, "invalid_request", "Missing or invalid parameter", request,
                  details={"errors": [
                      {"field": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]}
                      for e in exc.errors()]})


@app.exception_handler(Exception)
async def on_internal_error(request: Request, exc: Exception):
    return _error(500, "internal_error", "Unexpected service error", request)


def _secret_eq(given: str, expected: str) -> bool:
    """So secret CONSTANT-TIME (hmac.compare_digest trên bytes — chống timing
    attack); accept/reject y hệt so sánh == cũ."""
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def _check_auth(request: Request) -> JSONResponse | None:
    """401 nếu env cấu hình token/key mà header không khớp; mock mode nếu không đặt env."""
    bearer = config.bearer_token()
    api_key = config.service_api_key()
    if not bearer and not api_key:
        return None  # mock mode — PDF: "accept requests with or without authentication"
    auth_header = request.headers.get("Authorization", "")
    if bearer and _secret_eq(auth_header, f"Bearer {bearer}"):
        return None
    if api_key and _secret_eq(request.headers.get("X-API-Key", ""), api_key):
        return None
    return _error(401, "unauthorized", "Missing or invalid token/key", request)


def _check_admin_auth(request: Request) -> JSONResponse | None:
    """Auth admin (Phase 6, đóng TODO pass-through 4a) — token RIÊNG (env
    ADMIN_TOKEN, không dùng chung TASCO_BEARER_TOKEN), so constant-time.

    KHÁC search có chủ đích: search không token → mock MỞ; admin không token
    → KHOÁ 503 (admin ghi DB + tốn tiền AWS, không được default mở).
    """
    token = config.admin_token()
    if not token:
        return _error(503, "admin_disabled",
                      "ADMIN_TOKEN chưa đặt — ingestion bị khoá (bắt buộc đặt env để bật)",
                      request)
    if _secret_eq(request.headers.get("Authorization", ""), f"Bearer {token}"):
        return None
    return _error(401, "unauthorized", "Missing or invalid admin token", request)


@app.post("/admin/pois/batch")
def ingest_pois_batch(request: Request, records: list[dict] = Body(...)):
    """Batch ingest POI: validate từng record (4a) → VERIFY qua AWS Location
    (4b, đồng bộ, trước commit) → upsert Postgres trong 1 transaction (4a) →
    reindex MỘT LẦN atomic swap (4a). Auth: Bearer ADMIN_TOKEN (Phase 6).

    Partial-success: record méo bị reject kèm lý do, record hợp lệ vẫn vào;
    verify là "flag không reject" — AWS lỗi/score thấp → vẫn ghi với
    status='unverified' + reason; lỗi DB giữa batch → rollback sạch + KHÔNG
    reindex (verify xong không cứu được atomicity — vẫn rollback như 4a).
    """
    from src import ingestion
    from src.verify import geocode

    t0 = time.perf_counter()
    denied = _check_admin_auth(request)  # auth TRƯỚC mọi thứ — kể cả check nguồn
    if denied is not None:
        return denied
    if config.DATA_SOURCE != "postgres":
        return _error(503, "ingestion_unavailable",
                      "Ingestion cần DATA_SOURCE=postgres (nguồn xlsx là read-only)",
                      request)
    # Chặn batch quá cỡ TRƯỚC validate/verify — không tốn AWS call, không chạm DB
    max_batch = config.settings().ingestion.max_batch_size
    if len(records) > max_batch:
        return _error(413, "batch_too_large",
                      f"Batch {len(records)} records vượt giới hạn {max_batch} — chia nhỏ batch",
                      request, details={"received": len(records), "max": max_batch})
    valid, rejected = ingestion.validate_batch(records)
    verifications = geocode.verify_batch(valid) if valid else []
    report = {
        "received": len(records),
        "accepted": len(valid),
        "accepted_ids": [p.poi_id for p in valid],
        "verified": sum(v["status"] == "verified" for v in verifications),
        "unverified": sum(v["status"] == "unverified" for v in verifications),
        "verification": [
            {"poi_id": p.poi_id, "status": v["status"], "reason": v["reason"],
             "overall_score": v["overall_score"], "place_type": v["place_type"]}
            for p, v in zip(valid, verifications)
        ],
        "rejected": rejected,
        "reindex": None,
    }
    if valid:
        try:
            # 1 transaction — lỗi là rollback sạch (status theo verify per-record;
            # data_version bump cùng transaction — Phase 5)
            new_version = ingestion.upsert_pois(valid, [v["status"] for v in verifications])
        except Exception as e:
            # KHÔNG reindex: index phải khớp đúng thứ đã commit (= không gì cả)
            return _error(500, "ingestion_db_error",
                          "Batch rolled back, không record nào được ghi", request,
                          details={"reason": str(e).strip()})
        report["data_version"] = new_version
        try:
            report["reindex"] = _get_service().reindex()
            # Instance này đã ở version mới — không chờ poll; instance KHÁC
            # bắt kịp qua poller (eventual, trễ ≤ version_poll_seconds).
            _get_service().set_loaded_version(new_version)
        except Exception as e:
            # DB ĐÃ commit → ingest thành công thật; index stale chỉ tạm thời
            # (tự lành khi restart — load từ DB). Trả 500 ở đây gây hiểu nhầm
            # "không có gì xảy ra" trong khi data đã persist.
            logger.exception("reindex fail SAU khi batch đã commit — index tạm stale")
            report["reindex"] = {
                "status": "failed",
                "warning": "records đã commit vào DB nhưng index chưa refresh — "
                           "sẽ cập nhật ở lần reindex/restart sau",
                "reason": str(e).strip(),
            }
    n_unverified = report["unverified"]
    logger.info("ingest_batch", extra={
        "received": report["received"], "accepted": report["accepted"],
        "rejected": len(report["rejected"]),
        "verified": report["verified"], "unverified": n_unverified,
        # tín hiệu chất lượng data nguồn — unverified cao là data bẩn/toạ độ lệch
        "unverified_ratio": round(n_unverified / max(len(valid), 1), 3),
        "data_version": report.get("data_version"),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    })
    return report


@app.get("/", include_in_schema=False)
@app.get("/demo", include_in_schema=False)
def demo_page():
    """UI demo tĩnh (demo/index.html) — same-origin với API, offline 100%."""
    return FileResponse(config.ROOT / "demo" / "index.html", media_type="text/html")


@app.get("/health")
def health():
    """LIVENESS: process sống → 200 NGAY, kể cả khi index CHƯA build xong
    (không phụ thuộc _get_service — không trigger build). LB dùng để biết
    "đừng kill"; đã sẵn sàng nhận traffic hay chưa là việc của /ready."""
    body = {"status": "ok", "ready": _service_ready(),
            "embeddingModel": config.EMBEDDING_MODEL,
            "deterministic": True, "llmCalls": 0}
    if _service_ready():
        body["pois"] = _service.n_pois
    return body


@app.get("/ready")
def ready(request: Request):
    """READINESS: 200 CHỈ khi index đã build + warmup xong (và Postgres chạm
    được, khi DATA_SOURCE=postgres). Lúc startup ~7s → 503 để LB chưa gửi
    traffic tới. DB check nhẹ tay: SELECT 1, timeout ngắn."""
    if not _service_ready():
        return _error(503, "warming_up", "Index đang build/warmup — chưa nhận traffic",
                      request)
    body = {"status": "ready", "pois": _service.n_pois}
    if config.DATA_SOURCE == "postgres":
        try:
            import psycopg
            with psycopg.connect(config.database_url(), connect_timeout=2) as conn:
                conn.execute("SELECT 1")
        except Exception as e:
            return _error(503, "db_unreachable", "Postgres không chạm được", request,
                          details={"reason": str(e).strip()[:200]})
        body["data_version"] = _service.loaded_version
    return body


@app.get("/v1/search", response_model=SearchResponse, response_model_exclude_none=True)
@app.get("/search", include_in_schema=False)
@app.get("/v1/geocode-search", include_in_schema=False)
def search(
    request: Request,
    q: str = Query(..., min_length=1, description="User query (bắt buộc)",
                   examples=["quán cà phê yên tĩnh để làm việc"]),
    lat: Optional[float] = Query(None, description="Focus latitude for local ranking"),
    lon: Optional[float] = Query(None, description="Focus longitude for local ranking"),
    radiusMeters: Optional[float] = Query(None, gt=0, description="Search radius quanh focus point"),
    bbox: Optional[str] = Query(None, description="minLon,minLat,maxLon,maxLat"),
    category: Optional[str] = Query(None, description="Optional category filter (vd: Quán cà phê)"),
    limit: int = Query(10, ge=1, description="Default 10, max recommended 20"),
    lang: str = Query("vi", description="Default vi"),
    explain: bool = Query(False, description="Kèm QueryPlan + signal breakdown mỗi kết quả"),
):
    """Free-text search — GET /v1/search?q=cafe%20gần%20hồ%20gươm&limit=5

    Ví dụ response (rút gọn):
    ```json
    {
      "query": "cafe gần hồ gươm",
      "results": [{
        "id": "poi:C003", "type": "poi", "name": "Cộng Cà Phê Hồ Gươm",
        "label": "Cộng Cà Phê Hồ Gươm", "address": "Đinh Tiên Hoàng, Hoàn Kiếm, Hà Nội",
        "category": "Quán cà phê", "coordinates": {"lat": 21.03, "lon": 105.852},
        "score": 0.83, "source": "mock", "tags": ["coffee", "lake-view", "tourist"]
      }],
      "meta": {"limit": 5, "lang": "vi"}
    }
    ```
    """
    denied = _check_auth(request)
    if denied is not None:
        return denied
    limit = min(limit, MAX_LIMIT)

    hits = _get_service().search(q, lat=lat, lon=lon, limit=limit, explain=explain)

    # Post-filter theo contract (không đụng thuật toán ranking)
    if category:
        want = normalize_vi(category)
        hits = [h for h in hits if normalize_vi(h.poi.category) == want]
    if bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
        except ValueError:
            return _error(400, "invalid_request",
                          "bbox phải là minLon,minLat,maxLon,maxLat", request,
                          details={"field": "bbox"})
        hits = [h for h in hits
                if min_lat <= h.poi.lat <= max_lat and min_lon <= h.poi.lon <= max_lon]
    if radiusMeters is not None and lat is not None and lon is not None:
        hits = [h for h in hits
                if h.distance_meters is not None and h.distance_meters <= radiusMeters]

    results = [to_place_result(h) for h in hits[:limit]]
    # Log query + n_results (POI query không nhạy cảm); KHÔNG log lat/lon người dùng
    logger.info("search", extra={"query": q, "n_results": len(results), "limit": limit})
    return SearchResponse(query=q, results=results, meta=SearchMeta(limit=limit, lang=lang))
