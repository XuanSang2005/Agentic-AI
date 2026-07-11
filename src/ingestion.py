"""Phase 4a: batch ingest POI → Postgres (data động, chưa AWS verify — đó là 4b).

Thiết kế:
- Validate TỪNG record (Pydantic, extra=forbid): record méo gom vào rejected kèm
  lý do, KHÔNG giết batch. LỰA CHỌN partial-commit: reject ở tầng VALIDATE là
  deterministic per-record → các record hợp lệ vẫn commit (một transaction);
  còn lỗi ở tầng DB là lỗi hạ tầng không dự đoán được theo record → rollback
  SẠCH cả batch (caller không được reindex).
- Upsert theo poi_id (idempotent — push lại không tạo trùng). row_order của POI
  MỚI nối tiếp MAX hiện có (subquery trong cùng transaction thấy các insert
  trước đó → tăng dần tất định); POI đã có GIỮ NGUYÊN row_order — thứ tự corpus
  không xê dịch, vector cũ tái dùng được.
- status='pending' (PLACEHOLDER — Phase 4b sẽ verify qua AWS Location rồi mới
  chuyển trạng thái; loader/serve hiện chưa đọc status). Update record cũ cũng
  reset về pending: data đổi là phải verify lại.
- KHÔNG parse/normalize thứ hai: module này chỉ ghi field thô đã strip (_text
  của data_loader); document/norm_document/is_synthetic luôn do data_loader
  _finalize dựng khi load lại — một code path duy nhất, document không thể lệch.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src import config
from src.data_loader import _text


class PoiIn(BaseModel):
    """Record POI đầu vào — field khớp 1:1 cột pois; thiếu field bắt buộc/sai
    kiểu/field lạ → rejected."""
    model_config = ConfigDict(extra="forbid")

    poi_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    city: str = Field(min_length=1)
    brand: str = ""
    sub_category: str = ""
    district: str = ""
    address: str = ""
    lat: float = 0.0
    lon: float = 0.0
    rating: float = 0.0
    review_count: int = 0
    popularity_score: float = 0.0
    price_level: int = 0
    opening_hours: str = ""
    attributes: list[str] = []
    tags: list[str] = []
    description: str = ""


# row_order: POI mới = MAX+1 tại thời điểm insert (trong cùng tx nên tuần tự);
# ON CONFLICT KHÔNG update row_order (giữ vị trí corpus) nhưng reset status.
_UPSERT = """
    INSERT INTO pois (poi_id, row_order, name, brand, category, sub_category,
                      city, district, address, lat, lon, rating, review_count,
                      popularity_score, price_level, opening_hours, attributes,
                      tags, description, is_synthetic, status)
    VALUES (%s, (SELECT COALESCE(MAX(row_order), -1) + 1 FROM pois),
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s::jsonb, %s::jsonb, %s, %s, 'pending')
    ON CONFLICT (poi_id) DO UPDATE SET
        name = EXCLUDED.name, brand = EXCLUDED.brand,
        category = EXCLUDED.category, sub_category = EXCLUDED.sub_category,
        city = EXCLUDED.city, district = EXCLUDED.district,
        address = EXCLUDED.address, lat = EXCLUDED.lat, lon = EXCLUDED.lon,
        rating = EXCLUDED.rating, review_count = EXCLUDED.review_count,
        popularity_score = EXCLUDED.popularity_score,
        price_level = EXCLUDED.price_level,
        opening_hours = EXCLUDED.opening_hours,
        attributes = EXCLUDED.attributes, tags = EXCLUDED.tags,
        description = EXCLUDED.description,
        is_synthetic = EXCLUDED.is_synthetic,
        status = 'pending'
"""


def validate_batch(records: list[dict]) -> tuple[list[PoiIn], list[dict]]:
    """(valid, rejected) — rejected kèm index + poi_id (nếu đọc được) + lỗi từng field."""
    valid: list[PoiIn] = []
    rejected: list[dict] = []
    for i, raw in enumerate(records):
        try:
            valid.append(PoiIn.model_validate(raw))
        except ValidationError as e:
            rejected.append({
                "index": i,
                "poi_id": raw.get("poi_id") if isinstance(raw, dict) else None,
                "errors": [{"field": ".".join(str(x) for x in err["loc"]),
                            "reason": err["msg"]} for err in e.errors()],
            })
    return valid, rejected


def upsert_pois(pois: list[PoiIn]) -> None:
    """Ghi cả batch trong MỘT transaction — lỗi giữa chừng → psycopg tự rollback
    sạch (with connect: commit khi thoát êm, rollback khi exception nổi lên)."""
    import json

    import psycopg

    with psycopg.connect(config.database_url()) as conn:
        with conn.cursor() as cur:
            for p in pois:
                cur.execute(_UPSERT, (
                    _text(p.poi_id), _text(p.name), _text(p.brand),
                    _text(p.category), _text(p.sub_category), _text(p.city),
                    _text(p.district), _text(p.address), p.lat, p.lon, p.rating,
                    p.review_count, p.popularity_score, p.price_level,
                    _text(p.opening_hours),
                    json.dumps([_text(a) for a in p.attributes], ensure_ascii=False),
                    json.dumps([_text(t) for t in p.tags], ensure_ascii=False),
                    _text(p.description),
                    _text(p.poi_id).startswith("G"),  # cùng luật is_synthetic với loader
                ))
