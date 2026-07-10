"""Tính từng signal cho (QueryPlan, POI), mỗi cái trong [0,1].

⚠ TUYỆT ĐỐI không dùng POI.is_synthetic làm signal — down-rank G phải đến từ
category/city/attr consistency generalizable (bộ private có thể khác).
Signal thiếu thông tin → trả 0.5 TRUNG TÍNH (không thưởng không phạt).
"""
from __future__ import annotations

import math

from src.data_loader import POI, normalize_vi
from src.understanding.query_plan import QueryPlan
from src.understanding.rules import concept_tokens

# Cache attrs đã normalize theo poi_id (data tĩnh trong 1 process)
_NORM_ATTRS: dict[str, frozenset[str]] = {}


def _poi_attrs(poi: POI) -> frozenset[str]:
    if poi.id not in _NORM_ATTRS:
        _NORM_ATTRS[poi.id] = frozenset(normalize_vi(a) for a in poi.attributes)
    return _NORM_ATTRS[poi.id]


def _has_concept(poi: POI, concept_id: str) -> bool:
    """POI khớp concept nếu có BẤT KỲ token thành viên nào (concept-expansion)."""
    return bool(concept_tokens().get(concept_id, frozenset()) & _poi_attrs(poi))


def category_match(plan: QueryPlan, poi: POI) -> float:
    if not plan.categories:
        return 1.0  # không đoán được category → không phạt ai
    return 1.0 if poi.category in plan.categories else 0.0


def attr_match(plan: QueryPlan, poi: POI) -> float:
    """Tỉ lệ concept yêu cầu được POI thỏa (match ở mức CONCEPT, không phải token thô).

    Không yêu cầu gì → 0.5 trung tính. Mỗi neg concept POI dính → trừ 1 phần tương ứng.
    """
    req = plan.attr_concepts
    score = (sum(_has_concept(poi, c) for c in req) / len(req)) if req else 0.5
    if plan.neg_concepts:
        neg_hits = sum(_has_concept(poi, c) for c in plan.neg_concepts)
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
}
