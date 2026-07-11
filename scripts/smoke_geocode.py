"""SMOKE THẬT (chạy TAY, KHÔNG trong CI): gọi AWS Location Geocode v2 THẬT với
~10 địa chỉ Việt Nam thật từ dataset — đo ĐỘ PHỦ Việt Nam của verify.

Cần env AWS_LOCATION_API_KEY (API key của AWS Location, KHÔNG IAM/SigV4 —
xem TODO trong src/verify/geocode.py). Mỗi lần chạy tốn ~10 request
(SingleUse). In Overall + toạ độ matched + verdict từng địa chỉ.

Fail được PHÂN LOẠI theo nguyên nhân — KHÔNG gộp thành "phủ kém" một chiều:
  1. AWS KHÔNG resolve được (lỗi HTTP / không có ResultItems) → phủ kém THẬT.
  2. Resolve được nhưng Overall < ngưỡng → địa chỉ khớp, văn phong khác —
     KHÔNG phải phủ kém.
  3. Resolve tốt (score đạt) nhưng toạ độ lệch > ngưỡng (hoặc place type lạ)
     → TOẠ ĐỘ DATASET lệch (data issue) — verify đang làm đúng việc.
KHÔNG tự hạ threshold — chỉ report, người đọc quyết.

Chạy:  .venv/bin/python scripts/smoke_geocode.py [N]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import _pois_from_xlsx
from src.verify.geocode import geocode_verify


def _classify(result: dict | None, error: Exception | None) -> str:
    """verified | no_resolve | score_low | coords_off — ưu tiên theo thứ tự đó."""
    if error is not None or result is None:
        return "no_resolve"
    if result["status"] == "verified":
        return "verified"
    if result["overall_score"] is None:      # unverified không có kết quả geocode
        return "no_resolve"
    if result["overall_score"] < config.settings().verify.match_score_threshold:
        return "score_low"
    return "coords_off"                       # score đạt, chỉ lệch toạ độ/place type


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    # POI thật (không dòng G synthetic — địa chỉ G là giả, unverified là ĐÚNG)
    pois = [p for p in _pois_from_xlsx() if not p.is_synthetic and p.address][:n]

    print(f"Smoke geocode {len(pois)} địa chỉ thật (AWS Location, region từ config):\n")
    groups: dict[str, list[str]] = {"verified": [], "no_resolve": [],
                                    "score_low": [], "coords_off": []}
    scores: list[float] = []
    for p in pois:
        result, error = None, None
        try:
            result = geocode_verify(f"{p.address}, {p.city}", p.lat, p.lon)
        except Exception as e:
            error = e
        kind = _classify(result, error)
        groups[kind].append(p.id)
        if error is not None:
            print(f"  ✗ {p.id:<5} EXCEPTION: {error}")
            continue
        if result["overall_score"] is not None:
            scores.append(result["overall_score"])
        ok = kind == "verified"
        pos = result["matched_position"]
        pos_s = f"({pos['lat']:.5f},{pos['lon']:.5f})" if pos else "—"
        print(f"  {'✓' if ok else '✗'} {p.id:<5} overall={result['overall_score']}"
              f" dist={result.get('distance_m', '—')}m type={result['place_type']} {pos_s}")
        print(f"      {p.address}, {p.city}"
              + (f"\n      → {result['reason']}" if result["reason"] else ""))

    cfg = config.settings().verify
    total = len(pois)
    print(f"\n{'=' * 60}\nSUMMARY ({total} địa chỉ, ngưỡng score {cfg.match_score_threshold}"
          f" / distance {cfg.max_distance_m:.0f}m):")
    print(f"  ✓ verified                              : {len(groups['verified'])}/{total}"
          f"  {groups['verified']}")
    print(f"  1. AWS KHÔNG resolve (phủ kém THẬT)     : {len(groups['no_resolve'])}/{total}"
          f"  {groups['no_resolve']}")
    print(f"  2. resolve, score < ngưỡng (văn phong)  : {len(groups['score_low'])}/{total}"
          f"  {groups['score_low']}")
    print(f"  3. score đạt, toạ độ lệch (data issue)  : {len(groups['coords_off'])}/{total}"
          f"  {groups['coords_off']}")
    if scores:
        print(f"\n  Phân bố Overall (n={len(scores)}): min={min(scores):.2f}"
              f" max={max(scores):.2f}")
        print("  scores:", " ".join(f"{s:.2f}" for s in sorted(scores)))
    print("\n  KHÔNG tự hạ threshold — số liệu trên để người đọc quyết ngưỡng.")


if __name__ == "__main__":
    main()
