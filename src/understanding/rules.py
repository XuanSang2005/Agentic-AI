"""Rule-based extractor: query thô → QueryPlan (category, attr concepts, city, polarity, pop).

Chạy TRƯỚC và độc lập với LLM — cũng là planner chính ở deterministic mode.
Mọi matching trên text đã normalize_vi (bỏ dấu), word-boundary 2 đầu để tránh
match giữa từ ("hn" không ăn vào "hnx", "dong" không ăn vào "dong da" nhờ surface đa từ).
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from src import config
from src.data_loader import normalize_vi
from src.understanding.query_plan import QueryPlan

# --- City: pattern trên text đã bỏ dấu. Thứ tự = ưu tiên khi (hiếm) match nhiều city.
_CITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Hà Nội", re.compile(r"(?<![a-z0-9])(ha noi|hanoi|hn)(?![a-z0-9])")),
    ("TP.HCM", re.compile(r"(?<![a-z0-9])(tp\.? ?hcm|hcm|sai gon|saigon|ho chi minh|sg)(?![a-z0-9])")),
    ("Đà Nẵng", re.compile(r"(?<![a-z0-9])(da nang|danang)(?![a-z0-9])")),
    ("Đà Lạt", re.compile(r"(?<![a-z0-9])(da lat|dalat)(?![a-z0-9])")),
]

# --- Polarity: "không quá đông / không ồn / vắng" → cần yên tĩnh + PHỦ ĐỊNH đông khách.
_NEG_QUIET = re.compile(
    r"khong (qua )?(dong|on)( khach| duc| nguoi| ao)?|vang ve|it nguoi|it dong")

# --- Popularity: cờ bật popularity_score, KHÔNG phải attribute.
_POP = re.compile(r"(?<![a-z0-9])(noi tieng|famous|best|ngon|hot|dang di|dang den)(?![a-z0-9])")


def _boundary_pattern(surface: str) -> re.Pattern:
    """Surface (đã normalize) → regex word-boundary; escape để '24/7', 'tp.hcm' an toàn."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])")


@lru_cache(maxsize=1)
def _category_rules() -> list[tuple[re.Pattern, str]]:
    """[(pattern, canonical_category)] từ categories.yaml — surface dài match trước."""
    raw = yaml.safe_load(config.CATEGORIES_YAML.read_text(encoding="utf-8"))
    rules = []
    for entry in raw.values():
        canonical = entry["canonical"]
        for syn in entry["synonyms"]:
            rules.append((normalize_vi(str(syn)), canonical))
    return [(_boundary_pattern(s), cat) for s, cat in sorted(rules, key=lambda x: -len(x[0]))]


@lru_cache(maxsize=1)
def concept_tokens() -> dict[str, frozenset[str]]:
    """concept id → tập token thành viên ĐÃ normalize (để match phía POI attributes)."""
    raw = yaml.safe_load(config.ATTRIBUTE_CONCEPTS_YAML.read_text(encoding="utf-8"))
    return {cid: frozenset(normalize_vi(str(t)) for t in entry.get("tokens", []))
            for cid, entry in raw.items()}


@lru_cache(maxsize=1)
def _concept_rules() -> list[tuple[re.Pattern, str]]:
    """[(pattern, concept_id)] — trigger = tokens ∪ surface, đã normalize."""
    raw = yaml.safe_load(config.ATTRIBUTE_CONCEPTS_YAML.read_text(encoding="utf-8"))
    rules = []
    for cid, entry in raw.items():
        triggers = {normalize_vi(str(t))
                    for t in list(entry.get("tokens", [])) + list(entry.get("surface", []) or [])}
        rules.extend((t, cid) for t in triggers if t)
    return [(_boundary_pattern(t), cid) for t, cid in sorted(rules, key=lambda x: -len(x[0]))]


def _match_consuming(norm_text: str, rules: list[tuple[re.Pattern, str]]) -> set[str]:
    """Match longest-first và TIÊU THỤ span: surface dài match trước thì surface con
    nằm đè lên span đó không fire nữa ("họp nhóm"→phong_hop chặn "nhóm"→nhom;
    "trung tâm thương mại" chặn "trung tâm"). rules phải được sort dài→ngắn sẵn.
    """
    taken: list[tuple[int, int]] = []
    out: set[str] = set()
    for pat, value in rules:
        for m in pat.finditer(norm_text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue  # đè lên span đã tiêu thụ
            taken.append((s, e))
            out.add(value)
    return out


def extract_plan(query: str) -> QueryPlan:
    """Query thô → QueryPlan. Deterministic, không LLM, không đọc expected_ids."""
    norm = normalize_vi(query)
    plan = QueryPlan(query=query, norm_query=norm)

    for city, pat in _CITY_PATTERNS:
        if pat.search(norm):
            plan.city = city
            break

    # Consume span RIÊNG từng loại: category và concept được phép cùng ăn 1 đoạn text
    # ("ăn tối" → category Nhà hàng + concept an_uong là chủ đích).
    plan.categories = _match_consuming(norm, _category_rules())
    plan.attr_concepts = _match_consuming(norm, _concept_rules())

    # Polarity: suy ra yên tĩnh + phủ định đông khách; gỡ khỏi attrs nếu lỡ match dương.
    if _NEG_QUIET.search(norm):
        plan.attr_concepts.add("yen_tinh")
        plan.neg_concepts.add("dong_khach")
    plan.attr_concepts -= plan.neg_concepts

    plan.want_pop = bool(_POP.search(norm))
    return plan
