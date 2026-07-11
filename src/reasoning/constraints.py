"""Constraint-composition reasoning: tách câu nhiều ràng buộc, chấm từng cái, giải thích thỏa/nới.

Lớp ANNOTATION trên kết quả rerank hiện có — KHÔNG thay reranker, KHÔNG đổi thứ
tự kết quả (ranking bất biến → eval/stress bất biến theo cấu trúc). Rule-based
thuần trên field CÓ SẴN trong data: attributes, opening_hours, district/city,
price_level, category. Deterministic, offline, 0 LLM.

Constraint types + priority (cho thứ tự NỚI — nới priority thấp trước):
  - location / category : priority 2 (cao — sai chỗ/sai loại là sai hẳn)
  - attribute / time / price : priority 1 (thấp — nới được khi thiếu lựa chọn)

v2 (dynamic-vector-attributes): plan mang RAW attribute string (không còn concept
id) — phân loại time/price bằng keyword substring trên normalize_vi(attribute).
Thêm constraint có kiểu số: price_limit / rating_limit / time_limit / open_now.
open_now dùng giờ hệ thống (datetime.now) — nhánh DUY NHẤT không deterministic,
chỉ chạy khi query nói "đang mở cửa".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config
from src.data_loader import POI, normalize_vi
from src.ranking.signals import haversine_km
from src.understanding.query_plan import QueryPlan
from src.understanding.rules import landmark_label

# Ngưỡng + keyword phân loại constraint từ config/settings.yaml (constraints).
# Attribute đặc thù được nâng thành constraint CÓ KIỂU riêng (không phải attribute
# thuần): chấm bằng field cấu trúc thay vì chỉ token match. So khớp bằng substring
# trên normalize_vi(attribute_string) — hoạt động với attribute strings gốc từ
# dataset ("mở khuya", "giá rẻ", "miễn phí", "24/7").
_CONS = config.settings().constraints
_TIME_KEYWORDS = _CONS.time_keywords
_PRICE_KEYWORDS = _CONS.price_keywords
_PRICE_MAX_LEVEL = _CONS.price_max_level  # price_level trong data: 1..4

SATISFIED_THRESHOLD = _CONS.satisfied_threshold  # score ≥ ngưỡng → "thỏa"; dưới → "nới"


@dataclass
class Constraint:
    type: str            # category | attribute | time | location | price | *_limit | open_now
    value: str           # nhãn người-đọc-được ("Quán cà phê", "yên tĩnh", "gần hồ gươm")
    key: str             # machine key (canonical category / raw attribute / landmark key)
    priority: int        # 2 cao (location/category), 1 thấp (attribute/time/price)
    negated: bool = False          # "không quá đông" → thỏa khi KHÔNG có
    data: dict = field(default_factory=dict)  # payload (coord landmark, city…)


def _classify_attr(attr_str: str) -> str:
    """Phân loại attribute string thành constraint type: time, price, hoặc attribute."""
    norm = normalize_vi(attr_str)
    if any(kw in norm for kw in _TIME_KEYWORDS):
        return "time"
    if any(kw in norm for kw in _PRICE_KEYWORDS):
        return "price"
    return "attribute"


def parse_constraints(plan: QueryPlan) -> list[Constraint]:
    """QueryPlan (đã có sẵn từ L1) → danh sách ràng buộc có kiểu."""
    out: list[Constraint] = []

    for cat in sorted(plan.categories):
        out.append(Constraint("category", cat, cat, priority=2))

    for attr_str in sorted(plan.attr_concepts):
        ctype = _classify_attr(attr_str)
        out.append(Constraint(ctype, attr_str, attr_str, priority=1))
    for attr_str in sorted(plan.neg_concepts):
        out.append(Constraint("attribute", f"không {attr_str}", attr_str,
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

    # Numeric constraints: price, rating, time limits
    if plan.price_limit:
        op_label = "≤" if plan.price_op == "le" else "≥" if plan.price_op == "ge" else "="
        out.append(Constraint("price_limit", f"giá {op_label} {plan.price_limit}", f"{plan.price_op}:{plan.price_limit}", priority=1))

    if plan.rating_limit:
        op_label = "≥" if plan.rating_op == "ge" else "≤" if plan.rating_op == "le" else "="
        out.append(Constraint("rating_limit", f"đánh giá {op_label} {plan.rating_limit}", f"{plan.rating_op}:{plan.rating_limit}", priority=1))

    if plan.time_limit_minutes:
        h = plan.time_limit_minutes // 60
        m = plan.time_limit_minutes % 60
        time_str = f"{h:02d}:{m:02d}"
        op_label = "trước" if plan.time_op == "le" else "sau" if plan.time_op == "ge" else "="
        out.append(Constraint("time_limit", f"mở cửa {op_label} {time_str}", f"{plan.time_op}:{plan.time_limit_minutes}", priority=1))

    # Currently-open constraint
    if plan.current_time_open:
        from datetime import datetime
        now = datetime.now()
        time_str = f"{now.hour:02d}:{now.minute:02d}"
        out.append(Constraint("open_now", f"đang mở cửa ({time_str})",
                              "current_time", priority=1))

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


def _poi_has_attr(poi: POI, attr_str: str) -> bool:
    """POI khớp attribute nếu normalized attribute string match bất kỳ attribute nào."""
    return normalize_vi(attr_str) in frozenset(normalize_vi(a) for a in poi.attributes)


def score_constraint(poi: POI, c: Constraint) -> float:
    """POI thỏa ràng buộc bao nhiêu, [0,1]. Deterministic, chỉ đọc field data."""
    if c.type == "category":
        return 1.0 if poi.category == c.key else 0.0

    if c.type == "attribute":
        has = _poi_has_attr(poi, c.key)
        return (0.0 if has else 1.0) if c.negated else (1.0 if has else 0.0)

    if c.type == "time":
        hours = _parse_hours(poi.opening_hours)
        norm_key = normalize_vi(c.key)
        if "24/7" in norm_key or "hai bon bay" in norm_key:
            return 1.0 if poi.opening_hours.strip() == "24/7" else 0.0
        # mo khuya: 24/7 hoặc đóng qua đêm (close < open) → 1.0;
        # đóng ≥ late_close_full → 1.0; ≥ late_close_partial → partial; còn lại 0.
        if _poi_has_attr(poi, c.key):
            return 1.0  # token "mở khuya/mở muộn" tự khai trong attributes
        if hours is None:
            return 0.0
        o, close = hours
        if close == 1440 or close < o or close >= _CONS.late_close_full_minutes:
            return 1.0
        return (_CONS.time_partial_score
                if close >= _CONS.late_close_partial_minutes else 0.0)

    if c.type == "price":
        norm_key = normalize_vi(c.key)
        max_level = 1  # default fallback
        for kw, ml in _PRICE_MAX_LEVEL.items():
            if kw in norm_key:
                max_level = ml
                break
        if poi.price_level and poi.price_level <= max_level:
            return 1.0
        return 1.0 if _poi_has_attr(poi, c.key) else 0.0

    if c.type == "location":
        if "coord" in c.data:
            km = haversine_km(c.data["coord"][0], c.data["coord"][1], poi.lat, poi.lon)
            if km <= _CONS.near_km_full:
                return 1.0
            if km <= _CONS.near_km_partial:
                return _CONS.location_partial_score
            return _CONS.location_same_city_score if poi.city == c.data.get("city") else 0.0
        if "district" in c.data:
            if poi.district == c.data["district"] and poi.city == (c.data.get("city") or poi.city):
                return 1.0
            return (_CONS.location_city_fallback_score
                    if poi.city == c.data.get("city") else 0.0)
        return 1.0 if poi.city == c.data.get("city") else 0.0

    if c.type == "price_limit":
        op, val = c.key.split(":")
        val = int(float(val))
        if not poi.price_level:
            return 0.0
        if op == "le":
            return 1.0 if poi.price_level <= val else 0.0
        if op == "ge":
            return 1.0 if poi.price_level >= val else 0.0
        return 1.0 if poi.price_level == val else 0.0

    if c.type == "rating_limit":
        op, val = c.key.split(":")
        val = float(val)
        if not poi.rating:
            return 0.0
        if op == "ge":
            return 1.0 if poi.rating >= val else 0.0
        if op == "le":
            return 1.0 if poi.rating <= val else 0.0
        return 1.0 if poi.rating == val else 0.0

    if c.type == "time_limit":
        op, val = c.key.split(":")
        val = int(float(val))
        hours = _parse_hours(poi.opening_hours)
        if not hours:
            return 0.0
        o, close = hours
        if op == "le":
            return 1.0 if o <= val else 0.0
        if op == "ge":
            if close == 1440 or close < o:
                return 1.0
            return 1.0 if close >= val else 0.0
        return 0.0

    if c.type == "open_now":
        from src.ranking.signals import is_open_at
        from datetime import datetime
        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        return 1.0 if is_open_at(poi.opening_hours, current_minutes) else 0.0

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
