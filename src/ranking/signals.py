"""Tính từng signal cho (QueryPlan, POI), mỗi cái trong [0,1].

⚠ TUYỆT ĐỐI không dùng POI.is_synthetic làm signal — down-rank G phải đến từ
category/city/attr consistency generalizable (bộ private có thể khác).
Signal thiếu thông tin → trả 0.5 TRUNG TÍNH (không thưởng không phạt).
"""
from __future__ import annotations

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


# Registry để reranker dot-product với weights (bm25 tính riêng vì cần candidate set)
SIGNAL_FUNCS = {
    "category": category_match,
    "attr": attr_match,
    "city": city_match,
    "rating": rating_norm,
    "pop": popularity,
}
