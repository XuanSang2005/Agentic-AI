"""Tính từng signal cho (QueryPlan, POI), mỗi cái trong [0,1].

⚠ TUYỆT ĐỐI không dùng POI.is_synthetic làm signal — down-rank G phải đến từ
category/city/attr consistency generalizable (bộ private có thể khác).
Signal thiếu thông tin → trả 0.5 TRUNG TÍNH (không thưởng không phạt).

v3 (joint-semantic-competition): category_match & attr_match support continuous
weights from JointMetadataIndex. price_check, rating_check, time_check signals
added for numeric constraint routing.
"""
from __future__ import annotations

import math
import re

from src.data_loader import POI, normalize_vi
from src.understanding.query_plan import QueryPlan

# Cache attrs đã normalize theo poi_id (data tĩnh trong 1 process)
_NORM_ATTRS: dict[str, frozenset[str]] = {}


def _poi_attrs(poi: POI) -> frozenset[str]:
    if poi.id not in _NORM_ATTRS:
        _NORM_ATTRS[poi.id] = frozenset(normalize_vi(a) for a in poi.attributes)
    return _NORM_ATTRS[poi.id]


def _has_attr(poi: POI, attr_str: str) -> bool:
    """POI khớp attribute nếu normalized attribute string match bất kỳ attribute nào."""
    return normalize_vi(attr_str) in _poi_attrs(poi)


def category_match(plan: QueryPlan, poi: POI) -> float:
    """Category match: dùng continuous weights nếu có, fallback binary match."""
    if plan.category_weights:
        # Continuous: trả weight của category POI nếu nó nằm trong plan weights
        return plan.category_weights.get(poi.category, 0.0)
    if not plan.categories:
        return 1.0  # không đoán được category → không phạt ai
    return 1.0 if poi.category in plan.categories else 0.0


def attr_match(plan: QueryPlan, poi: POI) -> float:
    """Attribute match: dùng continuous weights nếu có, fallback binary match.

    Không yêu cầu gì → 0.5 trung tính. Negated attributes trừ điểm.
    """
    if plan.attribute_weights:
        # Continuous: sum weights of POI's matching attributes / total weights
        total_weight = sum(plan.attribute_weights.values())
        if total_weight == 0:
            score = 0.5
        else:
            matched_weight = sum(
                plan.attribute_weights.get(a, 0.0) for a in poi.attributes
            )
            score = matched_weight / total_weight
        # Deduct negated attribute weights
        if plan.negated_attribute_weights:
            neg_total = sum(plan.negated_attribute_weights.values())
            neg_hits = sum(
                plan.negated_attribute_weights.get(a, 0.0)
                for a in poi.attributes
                if a in plan.negated_attribute_weights
            )
            if neg_total > 0:
                score -= neg_hits / max(neg_total, 1)
        return min(1.0, max(0.0, score))

    # Fallback: binary matching
    req = plan.attr_concepts
    score = (sum(_has_attr(poi, a) for a in req) / len(req)) if req else 0.5
    if plan.neg_concepts:
        neg_hits = sum(_has_attr(poi, a) for a in plan.neg_concepts)
        score -= neg_hits / max(len(req), 1)
    return min(1.0, max(0.0, score))


def city_match(plan: QueryPlan, poi: POI) -> float:
    if plan.city is None:
        return 0.5  # query không nói city → trung tính
    return 1.0 if poi.city == plan.city else 0.0


def rating_norm(plan: QueryPlan, poi: POI) -> float:
    return poi.rating / 5.0


def popularity(plan: QueryPlan, poi: POI) -> float:
    """Chỉ thưởng popularity khi query bật cờ (nổi tiếng/best/ngon); còn lại trung tính."""
    return poi.popularity / 100.0 if plan.want_pop else 0.5


def name_match(plan: QueryPlan, poi: POI) -> float:
    """Khớp CHÍNH XÁC tên/brand (substring word-boundary trên bản bỏ dấu) — 1.0 hoặc 0.0.

    Cố tình STRICT thay vì token-overlap chung: tên POI chứa city/district token
    ("Aeon Mall Quận 1") mà tính overlap sẽ ăn điểm oan từ query location.
    """
    padded_query = f" {plan.norm_query} "
    for field in (poi.name, poi.brand):
        norm = normalize_vi(field) if field else ""
        if norm and f" {norm} " in padded_query:
            return 1.0
    return 0.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_score(plan: QueryPlan, poi: POI) -> float:
    """1/(1+km) tới resolved_coord (landmark/district centroid); không có coord → 0.5.

    POI nằm ĐÚNG district query hỏi → 1.0 thẳng, không qua centroid — centroid
    tính từ median toàn quận (có dòng G kéo lệch ~1-4km) chỉ đáng tin làm gradient
    cho POI NGOÀI quận, không đáng tin làm thước đo trong quận.
    """
    if plan.district and poi.district == plan.district and poi.city == (plan.city or poi.city):
        return 1.0
    if plan.resolved_coord is None:
        return 0.5
    if not poi.lat and not poi.lon:
        return 0.5
    km = haversine_km(plan.resolved_coord[0], plan.resolved_coord[1], poi.lat, poi.lon)
    return 1.0 / (1.0 + km)


# --- Numeric constraint signals ---

_HOURS_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


def _parse_closing_minutes(opening_hours: str) -> int | None:
    """Parse closing time from opening_hours string to minutes since midnight.

    Handles overnight closing (e.g. 18:00-02:00 → 26*60 = 1560 minutes).
    """
    s = opening_hours.strip()
    if s == "24/7":
        return 24 * 60  # 1440 — always open
    m = _HOURS_RE.match(s)
    if not m:
        return None
    open_min = int(m.group(1)) * 60 + int(m.group(2))
    close_min = int(m.group(3)) * 60 + int(m.group(4))
    # If close < open, it wraps past midnight (e.g., 18:00-02:00)
    if close_min <= open_min:
        close_min += 24 * 60
    return close_min


def _compare(value: float, limit: float, op: str) -> bool:
    """Dynamic comparison operator: le/ge/eq."""
    if op == "le":
        return value <= limit
    if op == "ge":
        return value >= limit
    if op == "eq":
        return abs(value - limit) < 0.01
    return value <= limit  # fallback


# VND → price_level mapping (duplicated from rules.py for independence)
_PRICE_LEVEL_MAP = [
    (50_000, 1),
    (150_000, 2),
    (300_000, 3),
]


def _vnd_to_level(vnd: float) -> int:
    for threshold, level in _PRICE_LEVEL_MAP:
        if vnd <= threshold:
            return level
    return 4


def price_check(plan: QueryPlan, poi: POI) -> float:
    """Check price constraint: compare POI's price_level against plan's limit."""
    if plan.price_limit is None:
        return 0.5  # no price constraint → neutral
    target_level = _vnd_to_level(plan.price_limit)
    if not poi.price_level:
        return 0.5  # no data → neutral
    return 1.0 if _compare(poi.price_level, target_level, plan.price_op) else 0.0


def rating_check(plan: QueryPlan, poi: POI) -> float:
    """Check rating constraint: compare POI's rating against plan's limit."""
    if plan.rating_limit is None:
        return 0.5  # no rating constraint → neutral
    return 1.0 if _compare(poi.rating, plan.rating_limit, plan.rating_op) else 0.0


def time_check(plan: QueryPlan, poi: POI) -> float:
    """Check time constraint: compare POI's closing time against plan's limit."""
    if plan.time_limit_minutes is None:
        return 0.5  # no time constraint → neutral
    closing = _parse_closing_minutes(poi.opening_hours)
    if closing is None:
        return 0.0  # can't parse → fail
    return 1.0 if _compare(closing, plan.time_limit_minutes, plan.time_op) else 0.0


def _parse_open_close_minutes(opening_hours: str) -> tuple[int, int] | None:
    """Parse opening_hours 'HH:MM-HH:MM' → (open_min, close_min); '24/7' → (0, 1440)."""
    s = opening_hours.strip()
    if s == "24/7":
        return (0, 1440)
    m = _HOURS_RE.match(s)
    if not m:
        return None
    open_min = int(m.group(1)) * 60 + int(m.group(2))
    close_min = int(m.group(3)) * 60 + int(m.group(4))
    return (open_min, close_min)


def is_open_at(opening_hours: str, current_minutes: int) -> bool:
    """Check if POI is open at the given minutes-since-midnight."""
    parsed = _parse_open_close_minutes(opening_hours)
    if parsed is None:
        return False
    open_min, close_min = parsed
    if close_min <= open_min:
        # Wraps past midnight (e.g., 18:00-02:00)
        return current_minutes >= open_min or current_minutes < close_min
    return open_min <= current_minutes < close_min


def open_now_check(plan: QueryPlan, poi: POI) -> float:
    """Score 1.0 if POI is currently open, 0.0 if closed, 0.5 if no constraint."""
    if not plan.current_time_open:
        return 0.5  # no "đang mở cửa" → neutral
    from datetime import datetime
    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    return 1.0 if is_open_at(poi.opening_hours, current_minutes) else 0.0


# Registry để reranker dot-product với weights.
# bm25/dense tính riêng trong reranker (điểm theo query, normalize trong candidate pool).
SIGNAL_FUNCS = {
    "category": category_match,
    "attr": attr_match,
    "city": city_match,
    "rating": rating_norm,
    "pop": popularity,
    "distance": distance_score,
    "name": name_match,
    "price_check": price_check,
    "rating_check": rating_check,
    "time_check": time_check,
    "open_now_check": open_now_check,
}

