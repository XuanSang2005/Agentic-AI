"""Seed Postgres từ xlsx — cầu nối chạy 1 lần (idempotent: upsert theo poi_id).

Đọc qua ĐÚNG nhánh parse xlsx của data_loader (_pois_from_xlsx) rồi ghi thẳng
field đã parse → Postgres chứa data GIỐNG HỆT những gì xlsx loader trả, không
parse lại lần hai. row_order = vị trí dòng xlsx (bắt buộc — thứ tự corpus
quyết định hash embedding cache). document/norm_document KHÔNG ghi — derived,
loader tự dựng bằng cùng code path.

Chạy:  DATABASE_URL=... .venv/bin/python scripts/seed_postgres.py
       (không đặt DATABASE_URL → default docker-compose dev, port 5433)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from src import config
from src.data_loader import _pois_from_xlsx

_UPSERT = """
    INSERT INTO pois (poi_id, row_order, name, brand, category, sub_category,
                      city, district, address, lat, lon, rating, review_count,
                      popularity_score, price_level, opening_hours, attributes,
                      tags, description, is_synthetic)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s::jsonb, %s::jsonb, %s, %s)
    ON CONFLICT (poi_id) DO UPDATE SET
        row_order = EXCLUDED.row_order, name = EXCLUDED.name,
        brand = EXCLUDED.brand, category = EXCLUDED.category,
        sub_category = EXCLUDED.sub_category, city = EXCLUDED.city,
        district = EXCLUDED.district, address = EXCLUDED.address,
        lat = EXCLUDED.lat, lon = EXCLUDED.lon, rating = EXCLUDED.rating,
        review_count = EXCLUDED.review_count,
        popularity_score = EXCLUDED.popularity_score,
        price_level = EXCLUDED.price_level,
        opening_hours = EXCLUDED.opening_hours,
        attributes = EXCLUDED.attributes, tags = EXCLUDED.tags,
        description = EXCLUDED.description,
        is_synthetic = EXCLUDED.is_synthetic
"""


def main() -> None:
    pois = _pois_from_xlsx()
    url = config.database_url()
    with psycopg.connect(url) as conn:
        # Áp TOÀN BỘ migrations theo thứ tự (docker-compose tự áp lúc init;
        # đường khác — brew/cloud — thì đây). Mọi file đều idempotent.
        migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
        for sql_file in sorted(migrations_dir.glob("*.sql")):
            conn.execute(sql_file.read_text(encoding="utf-8"))
        with conn.cursor() as cur:
            for i, p in enumerate(pois):
                cur.execute(_UPSERT, (
                    p.id, i, p.name, p.brand, p.category, p.sub_category,
                    p.city, p.district, p.address, p.lat, p.lon, p.rating,
                    p.review_count, p.popularity, p.price_level, p.opening_hours,
                    json.dumps(p.attributes, ensure_ascii=False),
                    json.dumps(p.tags, ensure_ascii=False),
                    p.description, p.is_synthetic,
                ))
        n = conn.execute("SELECT count(*) FROM pois").fetchone()[0]
    host = url.split("@")[-1] if "@" in url else url
    print(f"Seeded {len(pois)} POI từ xlsx → Postgres ({host}); bảng pois hiện có {n} dòng.")


if __name__ == "__main__":
    main()
