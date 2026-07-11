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

import uuid
from contextlib import asynccontextmanager

from typing import Optional

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from src import config
from src.api.dto import (ErrorBody, ErrorResponse, SearchMeta, SearchResponse,
                         to_place_result)
from src.data_loader import normalize_vi
from src.search import SearchService

MAX_LIMIT = config.settings().search.api_max_limit  # "max recommended 20" theo PDF

_service: SearchService | None = None


def _get_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_service()  # build index + load embedding cache 1 lần lúc startup
    yield


app = FastAPI(
    title="Tasco Maps — AI Semantic Search & Ranking (Track 2)",
    version="1.0.0",
    description="Search-and-ranking service cho POI tiếng Việt. "
                "Contract: docs/tasco_api.pdf. Deterministic, offline, 0 LLM call.",
    lifespan=lifespan,
)


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


def _check_auth(request: Request) -> JSONResponse | None:
    """401 nếu env cấu hình token/key mà header không khớp; mock mode nếu không đặt env."""
    bearer = config.bearer_token()
    api_key = config.service_api_key()
    if not bearer and not api_key:
        return None  # mock mode — PDF: "accept requests with or without authentication"
    auth_header = request.headers.get("Authorization", "")
    if bearer and auth_header == f"Bearer {bearer}":
        return None
    if api_key and request.headers.get("X-API-Key", "") == api_key:
        return None
    return _error(401, "unauthorized", "Missing or invalid token/key", request)


def _admin_auth() -> None:
    """PASS-THROUGH có chủ đích (Phase 4a) — điểm cắm auth cho endpoint admin.

    TODO(trước khi expose internet): auth thật cho admin — token riêng (không
    dùng chung TASCO_BEARER_TOKEN của search) hoặc mTLS/IAM tuỳ hạ tầng deploy.
    """


@app.post("/admin/pois/batch")
def ingest_pois_batch(request: Request, records: list[dict] = Body(...),
                      _auth: None = Depends(_admin_auth)):
    """Batch ingest POI (Phase 4a): validate từng record → upsert Postgres trong
    1 transaction → reindex MỘT LẦN (atomic swap) để serve thấy POI mới.

    Partial-success: record méo bị reject kèm lý do, record hợp lệ vẫn vào;
    lỗi DB giữa batch → rollback sạch + KHÔNG reindex. POI mới mang
    status='pending' (chưa verify — Phase 4b).
    """
    from src import ingestion

    if config.DATA_SOURCE != "postgres":
        return _error(503, "ingestion_unavailable",
                      "Ingestion cần DATA_SOURCE=postgres (nguồn xlsx là read-only)",
                      request)
    valid, rejected = ingestion.validate_batch(records)
    report = {
        "received": len(records),
        "accepted": len(valid),
        "accepted_ids": [p.poi_id for p in valid],
        "rejected": rejected,
        "reindex": None,
    }
    if valid:
        try:
            ingestion.upsert_pois(valid)  # 1 transaction — lỗi là rollback sạch
        except Exception as e:
            # KHÔNG reindex: index phải khớp đúng thứ đã commit (= không gì cả)
            return _error(500, "ingestion_db_error",
                          "Batch rolled back, không record nào được ghi", request,
                          details={"reason": str(e).strip()})
        report["reindex"] = _get_service().reindex()
    return report


@app.get("/", include_in_schema=False)
@app.get("/demo", include_in_schema=False)
def demo_page():
    """UI demo tĩnh (demo/index.html) — same-origin với API, offline 100%."""
    return FileResponse(config.ROOT / "demo" / "index.html", media_type="text/html")


@app.get("/health")
def health():
    """Health check — kèm thông tin chế độ deterministic/offline."""
    svc = _get_service()
    return {"status": "ok", "pois": svc.n_pois, "embeddingModel": config.EMBEDDING_MODEL,
            "deterministic": True, "llmCalls": 0}


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

    return SearchResponse(
        query=q,
        results=[to_place_result(h) for h in hits[:limit]],
        meta=SearchMeta(limit=limit, lang=lang),
    )
