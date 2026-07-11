"""Constraint-composition reasoning: tách câu nhiều ràng buộc, chấm từng cái, giải thích thỏa/nới.

Lớp ANNOTATION trên kết quả rerank hiện có — KHÔNG thay reranker, KHÔNG đổi thứ
tự kết quả (ranking bất biến → eval/stress bất biến theo cấu trúc). Rule-based
thuần trên field CÓ SẴN trong data: attributes, opening_hours, district/city,
price_level, category. Deterministic, offline, 0 LLM.

Constraint types + priority (cho thứ tự NỚI — nới priority thấp trước):
  - location / category : priority 2 (cao — sai chỗ/sai loại là sai hẳn)
  - attribute / time / price : priority 1 (thấp — nới được khi thiếu lựa chọn)

"Đang mở cửa" (open-now) CỐ Ý không làm: cần giờ hệ thống → phá deterministic.
Nếu cần thì nhận giờ tường minh qua API param (future).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config
from src.data_loader import POI, normalize_vi
from src.ranking.signals import haversine_km
from src.understanding.query_plan import QueryPlan
from src.understanding.rules import concept_label, concept_tokens, landmark_label

# Ngưỡng + mapping concept→kiểu constraint từ config/settings.yaml (constraints).
# Concept đặc thù được nâng thành constraint CÓ KIỂU riêng (không phải attribute
# thuần): chấm bằng field cấu trúc thay vì chỉ token match.
_CONS = config.settings().constraints
_TIME_CONCEPTS = _CONS.time_concepts
_PRICE_CONCEPTS = _CONS.price_concepts
_PRICE_MAX_LEVEL = _CONS.price_max_level  # price_level trong data: 1..4

SATISFIED_THRESHOLD = _CONS.satisfied_threshold  # score ≥ ngưỡng → "thỏa"; dưới → "nới"


@dataclass
class Constraint:
    type: str            # category | attribute | time | location | price
    value: str           # nhãn người-đọc-được ("Quán cà phê", "yên tĩnh", "gần hồ gươm")
    key: str             # machine key (canonical category / concept id / landmark key)
    priority: int        # 2 cao (location/category), 1 thấp (attribute/time/price)
    negated: bool = False          # "không quá đông" → thỏa khi KHÔNG có
    data: dict = field(default_factory=dict)  # payload (coord landmark, city…)


def parse_constraints(plan: QueryPlan) -> list[Constraint]:
    """QueryPlan (đã có sẵn từ L1) → danh sách ràng buộc có kiểu."""
    out: list[Constraint] = []

    for cat in sorted(plan.categories):
        out.append(Constraint("category", cat, cat, priority=2))

    for cid in sorted(plan.attr_concepts):
        if cid in _TIME_CONCEPTS:
            out.append(Constraint("time", concept_label(cid), cid, priority=1))
        elif cid in _PRICE_CONCEPTS:
            out.append(Constraint("price", concept_label(cid), cid, priority=1))
        else:
            out.append(Constraint("attribute", concept_label(cid), cid, priority=1))
    for cid in sorted(plan.neg_concepts):
        out.append(Constraint("attribute", f"không {concept_label(cid)}", cid,
                              priority=1, negated=True))

    # Location: lấy MỨC CỤ THỂ NHẤT (landmark > district > city) làm 1 ràng buộc
    if plan.landmark and plan.resolved_coord:
        out.append(Constraint("location", f"gần {landmark_label(plan.landmark)}",
                              plan.landmark, priority=2,
                              data={"coord": plan.resolved_coord, "city": plan.city}))
    elif plan.district:
        out.append(Constraint("location", f"ở {plan.district}", plan.district,
                              priority=2, data={"district": plan.district,
                                                "city": plan.city}))
    elif plan.city:
        out.append(Constraint("location", f"ở {plan.city}", plan.city,
                              priority=2, data={"city": plan.city}))
    return out


# --- chấm từng ràng buộc ---

_HOURS_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


def _parse_hours(s: str) -> tuple[int, int] | None:
    """'HH:MM-HH:MM' → (open_min, close_min); '24/7' → (0, 1440); lạ → None."""
    if s.strip() == "24/7":
        return (0, 1440)
    m = _HOURS_RE.match(s.strip())
    if not m:
        return None
    o = int(m.group(1)) * 60 + int(m.group(2))
    c = int(m.group(3)) * 60 + int(m.group(4))
    return (o, c)


def _poi_has_concept(poi: POI, concept_id: str) -> bool:
    tokens = concept_tokens().get(concept_id, frozenset())
    return bool(tokens & {normalize_vi(a) for a in poi.attributes})


def score_constraint(poi: POI, c: Constraint) -> float:
    """POI thỏa ràng buộc bao nhiêu, [0,1]. Deterministic, chỉ đọc field data."""
    if c.type == "category":
        return 1.0 if poi.category == c.key else 0.0

    if c.type == "attribute":
        has = _poi_has_concept(poi, c.key)
        return (0.0 if has else 1.0) if c.negated else (1.0 if has else 0.0)

    if c.type == "time":
        hours = _parse_hours(poi.opening_hours)
        if c.key == "hai_bon_bay":
            return 1.0 if poi.opening_hours.strip() == "24/7" else 0.0
        # mo_khuya: 24/7 hoặc đóng qua đêm (close < open) → 1.0;
        # đóng ≥ late_close_full → 1.0; ≥ late_close_partial → 0.5; còn lại 0.
        if _poi_has_concept(poi, c.key):
            return 1.0  # token "mở khuya/mở muộn" tự khai trong attributes
        if hours is None:
            return 0.0
        o, close = hours
        if close == 1440 or close < o or close >= _CONS.late_close_full_minutes:
            return 1.0
        return 0.5 if close >= _CONS.late_close_partial_minutes else 0.0

    if c.type == "price":
        max_level = _PRICE_MAX_LEVEL[c.key]
        if poi.price_level and poi.price_level <= max_level:
            return 1.0
        return 1.0 if _poi_has_concept(poi, c.key) else 0.0

    if c.type == "location":
        if "coord" in c.data:
            km = haversine_km(c.data["coord"][0], c.data["coord"][1], poi.lat, poi.lon)
            if km <= _CONS.near_km_full:
                return 1.0
            if km <= _CONS.near_km_partial:
                return 0.7
            return 0.2 if poi.city == c.data.get("city") else 0.0
        if "district" in c.data:
            if poi.district == c.data["district"] and poi.city == (c.data.get("city") or poi.city):
                return 1.0
            return 0.5 if poi.city == c.data.get("city") else 0.0
        return 1.0 if poi.city == c.data.get("city") else 0.0

    return 0.0


def annotate(plan: QueryPlan, poi: POI) -> dict | None:
    """Bảng thỏa/nới cho 1 kết quả — đưa thẳng vào explanation["constraints"].

    "Nới" = ràng buộc score < ngưỡng, sắp theo priority TĂNG dần (nới cái
    ít quan trọng trước — attribute/time/price rồi mới location/category).
    """
    constraints = parse_constraints(plan)
    if not constraints:
        return None
    detail = []
    for c in constraints:
        s = score_constraint(poi, c)
        detail.append({"type": c.type, "label": c.value, "score": round(s, 2),
                       "satisfied": s >= SATISFIED_THRESHOLD, "priority": c.priority})
    relaxed = [d["label"] for d in sorted(detail, key=lambda d: d["priority"])
               if not d["satisfied"]]
    return {
        "total": len(detail),
        "satisfied": sum(d["satisfied"] for d in detail),
        "detail": detail,
        "relaxed": relaxed,
    }
