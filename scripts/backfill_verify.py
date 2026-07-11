"""Backfill verify TOÀN BỘ POI trong Postgres — chạy TAY, một lần (re-run được).

Đưa data CŨ (status='active', chưa từng qua verify) về cùng chuẩn với data
ingest mới: verified/unverified thật từ AWS Location. Sau backfill, badge
hiển thị trên MỌI kết quả search, không chỉ POI ingest sau này.

- Dùng ĐÚNG bộ máy verify của ingestion (verify_batch: threadpool + retry +
  flag-không-reject) và CÙNG cách ghép query như smoke đã tune ngưỡng
  (address + ", " + city).
- UPDATE status theo poi_id trong MỘT transaction + bump data_version →
  mọi serve instance tự reload (Phase 5), không cần restart.
- Cần AWS_LOCATION_API_KEY (.env tự nạp); ~1 request/POI.
- Dòng G synthetic có địa chỉ giả → unverified là ĐÚNG (verify đang bắt fake).

Chạy:  .venv/bin/python scripts/backfill_verify.py [--dry-run]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import _pois_from_postgres
from src.ingestion import _BUMP_VERSION
from src.verify.geocode import verify_batch


def main() -> None:
    dry = "--dry-run" in sys.argv
    pois = _pois_from_postgres()
    print(f"Verify {len(pois)} POI qua AWS Location (threadpool "
          f"{config.settings().verify.max_workers}, flag không reject)...")

    # Cùng shape query với smoke/ngưỡng đã tune: address + city (address trong
    # data không chứa city; city giúp AWS match đúng vùng)
    probes = [SimpleNamespace(address=f"{p.address}, {p.city}" if p.address else "",
                              lat=p.lat, lon=p.lon) for p in pois]
    results = verify_batch(probes)

    by_status = Counter(r["status"] for r in results)
    changed = sum(1 for p, r in zip(pois, results) if p.status != r["status"])
    print(f"  → {dict(by_status)} | {changed}/{len(pois)} POI đổi status")
    for p, r in zip(pois, results):
        if r["status"] == "unverified":
            print(f"    ✗ {p.id:<8} {r['reason']}")

    if dry:
        print("DRY RUN — không ghi gì.")
        return

    import psycopg
    with psycopg.connect(config.database_url()) as conn:
        with conn.cursor() as cur:
            for p, r in zip(pois, results):
                cur.execute("UPDATE pois SET status = %s WHERE poi_id = %s",
                            (r["status"], p.id))
        new_version = conn.execute(_BUMP_VERSION).fetchone()[0]
    print(f"Đã ghi status + bump data_version → {new_version} "
          f"(serve instances tự reload trong ≤ 1 chu kỳ poll).")


if __name__ == "__main__":
    main()
