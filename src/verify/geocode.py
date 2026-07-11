"""Phase 4b: verify POI ingest bằng AWS Location Geocode v2 (geo-places).

NGUYÊN TẮC "flag không reject": verify là bước CÓ THỂ HỎNG — AWS lỗi/timeout/
throttle thì POI VẪN được ghi với status='unverified' + reason; không bao giờ
làm sập ingestion. Ba lớp bảo vệ trong verify_batch:
  1. try/except TỪNG record — một record lỗi không kéo sập batch.
  2. Song song có giới hạn (ThreadPoolExecutor, max_workers từ config).
  3. Retry backoff khi ThrottlingException (delays từ config); hết retry →
     unverified, không fail oan.

Client AWS INJECT ĐƯỢC (set_client / param) — unit test thay fake, KHÔNG gọi
mạng thật. Credential từ env / ~/.aws (boto3 default chain, KHÔNG hardcode);
region + ngưỡng từ config/settings.yaml (verify).

⚠ AWS Position là [lng, lat] — NGƯỢC với (lat, lon) của hệ; convert 2 chiều
đều nằm gọn trong module này.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from src import config
from src.ranking.signals import haversine_km

_VERIFIED = "verified"
_UNVERIFIED = "unverified"

_client = None  # singleton lazy; test thay bằng set_client(fake)


def _boto_client():
    import boto3
    return boto3.client("geo-places", region_name=config.settings().verify.aws_region)


def get_client():
    global _client
    if _client is None:
        _client = _boto_client()
    return _client


def set_client(client) -> None:
    """Inject client (fake trong test; None → quay về boto3 thật)."""
    global _client
    _client = client


def _is_throttling(exc: Exception) -> bool:
    """Nhận ThrottlingException của botocore lẫn fake test (theo Code hoặc tên class)."""
    code = getattr(exc, "response", {}) or {}
    code = code.get("Error", {}).get("Code", "") if isinstance(code, dict) else ""
    return code == "ThrottlingException" or type(exc).__name__ == "ThrottlingException"


def _unverified(reason: str, **extra) -> dict:
    return {"status": _UNVERIFIED, "matched_position": None, "overall_score": None,
            "place_type": None, "reason": reason, **extra}


def geocode_verify(address: str, claimed_lat: float, claimed_lon: float,
                   client=None) -> dict:
    """1 địa chỉ → {status, matched_position, overall_score, place_type, reason}.

    verified khi ĐỦ CẢ BA: Overall ≥ threshold, khoảng cách claim↔matched ≤
    max_distance_m, PlaceType thuộc danh sách chấp nhận — else unverified với
    reason cụ thể từng điều kiện fail. Exception AWS để NỔ LÊN — caller
    (verify_batch) quyết định flag; retry throttling xử lý tại đây.
    """
    cfg = config.settings().verify
    if not (address or "").strip():
        return _unverified("no address to verify")

    client = client if client is not None else get_client()
    attempt = 0
    while True:
        try:
            resp = client.geocode(
                QueryText=address,
                BiasPosition=[claimed_lon, claimed_lat],  # AWS là [lng, lat] — ĐỪNG đảo
                Filter={"IncludeCountries": ["VNM"]},
                IntendedUse="SingleUse",  # verify thuần, không lưu kết quả AWS
                MaxResults=1,
            )
            break
        except Exception as e:
            if _is_throttling(e) and attempt < len(cfg.throttle_retry_delays):
                time.sleep(cfg.throttle_retry_delays[attempt])
                attempt += 1
                continue
            raise

    items = resp.get("ResultItems") or []
    if not items:
        return _unverified("no geocode result for address")
    item = items[0]
    matched_lon, matched_lat = float(item["Position"][0]), float(item["Position"][1])
    overall = float((item.get("MatchScores") or {}).get("Overall") or 0.0)
    place_type = str(item.get("PlaceType") or "")
    dist_m = haversine_km(claimed_lat, claimed_lon, matched_lat, matched_lon) * 1000

    reasons = []
    if overall < cfg.match_score_threshold:
        reasons.append(f"match score {overall:.2f} < {cfg.match_score_threshold}")
    if dist_m > cfg.max_distance_m:
        reasons.append(f"matched position {dist_m:.0f}m away (> {cfg.max_distance_m:.0f}m)")
    if place_type not in cfg.place_types:
        reasons.append(f"place type {place_type!r} not in {list(cfg.place_types)}")

    return {
        "status": _UNVERIFIED if reasons else _VERIFIED,
        "matched_position": {"lat": matched_lat, "lon": matched_lon},
        "overall_score": overall,
        "place_type": place_type,
        "distance_m": round(dist_m, 1),
        "reason": "; ".join(reasons) or None,
    }


def verify_batch(pois, client=None) -> list[dict]:
    """Verify N record song song (giữ NGUYÊN thứ tự input). Không exception nào
    thoát ra ngoài: record lỗi AWS → unverified + reason, record khác đi tiếp."""
    def one(p) -> dict:
        try:
            return geocode_verify(p.address, p.lat, p.lon, client=client)
        except Exception as e:  # timeout/network/credential/lỗi lạ — flag không reject
            return _unverified(f"verify failed: {e}")

    workers = min(config.settings().verify.max_workers, max(len(pois), 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, pois))
