"""SMOKE THẬT (chạy TAY, KHÔNG trong CI): gọi AWS Location Geocode v2 THẬT với
~10 địa chỉ Việt Nam thật từ dataset — đo ĐỘ PHỦ Việt Nam của verify.

Cần env AWS_LOCATION_API_KEY (API key của AWS Location, KHÔNG IAM/SigV4 —
xem TODO trong src/verify/geocode.py). Mỗi lần chạy tốn ~10 request
(SingleUse). In Overall + toạ độ matched + verdict từng địa chỉ.

⚠ Nếu nhiều địa chỉ THẬT ra unverified/score thấp → phủ VN kém: BÁO LẠI để
quyết định ngưỡng/nguồn — ĐỪNG tự hạ threshold trong config cho "đẹp số".

Chạy:  .venv/bin/python scripts/smoke_geocode.py [N]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import _pois_from_xlsx
from src.verify.geocode import geocode_verify


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    # POI thật (không dòng G synthetic — địa chỉ G là giả, unverified là ĐÚNG)
    pois = [p for p in _pois_from_xlsx() if not p.is_synthetic and p.address][:n]

    print(f"Smoke geocode {len(pois)} địa chỉ thật (AWS Location, region từ config):\n")
    n_verified = 0
    for p in pois:
        try:
            r = geocode_verify(f"{p.address}, {p.city}", p.lat, p.lon)
        except Exception as e:
            print(f"  ✗ {p.id:<5} EXCEPTION: {e}")
            continue
        ok = r["status"] == "verified"
        n_verified += ok
        pos = r["matched_position"]
        pos_s = f"({pos['lat']:.5f},{pos['lon']:.5f})" if pos else "—"
        print(f"  {'✓' if ok else '✗'} {p.id:<5} overall={r['overall_score']}"
              f" dist={r.get('distance_m', '—')}m type={r['place_type']} {pos_s}")
        print(f"      {p.address}, {p.city}"
              + (f"\n      → {r['reason']}" if r["reason"] else ""))
    print(f"\nVerified {n_verified}/{len(pois)}."
          " Phủ kém (nhiều unverified) → báo lại, đừng tự hạ threshold.")


if __name__ == "__main__":
    main()
