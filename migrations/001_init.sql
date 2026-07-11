-- Schema bảng pois — cột khớp 1:1 field POI dataclass (src/data_loader.py).
--
-- ⚠ lat/lon/rating/popularity_score là DOUBLE PRECISION (float8 = IEEE double,
-- round-trip đúng bit với float Python) — KHÔNG NUMERIC/Decimal: Decimal lệch
-- float khi tính signal.
--
-- ⚠ row_order = vị trí dòng trong nguồn gốc (xlsx). Loader ORDER BY row_order
-- vì thứ tự corpus quyết định hash embedding cache (.npy) — xlsx KHÔNG sort
-- theo poi_id (C→R→H→A→S→M→G) nên ORDER BY poi_id sẽ đảo corpus và phá cache.
--
-- document/norm_document KHÔNG lưu — derived field, loader tự dựng (1 code path
-- duy nhất cho mọi nguồn). is_synthetic lưu để query tay/BI, loader vẫn tự suy.
-- status cho ingestion phase sau — loader hiện chưa đọc.

CREATE TABLE IF NOT EXISTS pois (
    poi_id           TEXT PRIMARY KEY,
    row_order        INTEGER NOT NULL UNIQUE,
    name             TEXT NOT NULL DEFAULT '',
    brand            TEXT NOT NULL DEFAULT '',
    category         TEXT NOT NULL DEFAULT '',
    sub_category     TEXT NOT NULL DEFAULT '',
    city             TEXT NOT NULL DEFAULT '',
    district         TEXT NOT NULL DEFAULT '',
    address          TEXT NOT NULL DEFAULT '',
    lat              DOUBLE PRECISION NOT NULL DEFAULT 0,
    lon              DOUBLE PRECISION NOT NULL DEFAULT 0,
    rating           DOUBLE PRECISION NOT NULL DEFAULT 0,
    review_count     INTEGER NOT NULL DEFAULT 0,
    popularity_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    price_level      INTEGER NOT NULL DEFAULT 0,
    opening_hours    TEXT NOT NULL DEFAULT '',
    attributes       JSONB NOT NULL DEFAULT '[]',
    tags             JSONB NOT NULL DEFAULT '[]',
    description      TEXT NOT NULL DEFAULT '',
    is_synthetic     BOOLEAN NOT NULL DEFAULT FALSE,
    status           TEXT NOT NULL DEFAULT 'active'
);
