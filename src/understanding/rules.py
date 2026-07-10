"""Rule-based extractor: query thô → QueryPlan (category, attr concepts, city, polarity, pop).

Chạy TRƯỚC và độc lập với LLM — cũng là planner chính ở deterministic mode.
Mọi matching trên text đã normalize_vi (bỏ dấu), word-boundary 2 đầu để tránh
match giữa từ ("hn" không ăn vào "hnx", "dong" không ăn vào "dong da" nhờ surface đa từ).
"""
from __future__ import annotations

import re
import statistics
from functools import lru_cache

import yaml

from src import config
from src.data_loader import load_pois, normalize_vi
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


# --- Landmark (gazetteer) + district ---
# Cue bắt buộc trước landmark: "gần/near/cạnh/sát/quanh" trong cửa sổ 15 ký tự.
# Bẫy P009: "trên đường đi hạ long" không có cue "gần" → không resolve, đúng chủ đích.
_NEAR_CUE = re.compile(r"(?<![a-z0-9])(gan|near|canh|sat|quanh)(?![a-z0-9])")


@lru_cache(maxsize=1)
def _gazetteer_rules() -> list[tuple[re.Pattern, dict]]:
    """[(name_pattern, entry)] — tên dài match trước ("biển mỹ khê" trước "biển")."""
    raw = yaml.safe_load(config.GAZETTEER_YAML.read_text(encoding="utf-8"))
    rules = []
    for key, entry in raw.items():
        info = {"key": key, "lat": float(entry["lat"]), "lon": float(entry["lon"]),
                "city": entry["city"]}
        rules.extend((normalize_vi(str(n)), info) for n in entry["names"])
    return [(_boundary_pattern(n), info) for n, info in sorted(rules, key=lambda x: -len(x[0]))]


@lru_cache(maxsize=1)
def _district_rules() -> list[tuple[re.Pattern, tuple[str, str]]]:
    """[(pattern, (district, city))] sinh từ data — "Quận N" thêm biến thể qN/district N."""
    pairs = {(p.district, p.city) for p in load_pois() if p.district}
    rules = []
    for district, city in pairs:
        surfaces = {normalize_vi(district)}
        m = re.fullmatch(r"quan (\d+)", normalize_vi(district))
        if m:
            surfaces |= {f"q{m.group(1)}", f"q.{m.group(1)}", f"district {m.group(1)}"}
        rules.extend((s, (district, city)) for s in surfaces)
    return [(_boundary_pattern(s), dc) for s, dc in sorted(rules, key=lambda x: -len(x[0]))]


@lru_cache(maxsize=1)
def district_centroids() -> dict[tuple[str, str], tuple[float, float]]:
    """(city, district) → (median lat, median lon) trên TOÀN BỘ POI trong quận.

    Median (không phải mean) để vài dòng toạ độ lệch không kéo trôi centroid.
    Lưu ý: signals chấm POI CÙNG district = distance 1.0 trực tiếp — centroid chỉ
    làm gradient cho POI khác quận, nên sai số ~1km ở đây không nguy hiểm.
    """
    by_district: dict[tuple[str, str], list] = {}
    for p in load_pois():
        if p.district and p.lat and p.lon:
            by_district.setdefault((p.city, p.district), []).append(p)
    return {key: (statistics.median(p.lat for p in ps), statistics.median(p.lon for p in ps))
            for key, ps in by_district.items()}


def _detect_landmark(plan: QueryPlan, norm: str) -> tuple[int, int] | None:
    """Resolve landmark → trả span đã match để CONSUME: "gần hồ xuân hương" là
    location chính xác (distance signal lo), không được rớt xuống concept "gần hồ"
    generic rồi cộng oan attr cho POI cạnh hồ KHÁC thành phố.
    """
    for pat, info in _gazetteer_rules():
        m = pat.search(norm)
        if not m:
            continue
        if not _NEAR_CUE.search(norm[max(0, m.start() - 15):m.start()]):
            continue  # không có "gần/near" ngay trước → chỉ là ngữ cảnh (bẫy P009)
        if plan.city is not None and plan.city != info["city"]:
            continue  # "gần biển" nhưng query nói city không có biển → không resolve
        plan.landmark = info["key"]
        plan.resolved_coord = (info["lat"], info["lon"])
        plan.city = plan.city or info["city"]
        return m.span()
    return None


def _detect_district(plan: QueryPlan, norm: str) -> None:
    for pat, (district, city) in _district_rules():
        if pat.search(norm):
            if plan.city is not None and plan.city != city:
                continue
            plan.district = district
            plan.city = plan.city or city
            if plan.resolved_coord is None:  # landmark (điểm chính xác) được ưu tiên
                plan.resolved_coord = district_centroids().get((city, district))
            return


def _match_consuming(norm_text: str, rules: list[tuple[re.Pattern, str]],
                     pre_taken: tuple[tuple[int, int], ...] = ()) -> set[str]:
    """Match longest-first và TIÊU THỤ span: surface dài match trước thì surface con
    nằm đè lên span đó không fire nữa ("họp nhóm"→phong_hop chặn "nhóm"→nhom;
    "trung tâm thương mại" chặn "trung tâm"). rules phải được sort dài→ngắn sẵn.
    pre_taken: span đã bị tầng trước tiêu thụ (vd landmark) — không match đè lên.
    """
    taken: list[tuple[int, int]] = list(pre_taken)
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

    _detect_landmark(plan, norm)   # cần cue "gần/near", có city guard
    _detect_district(plan, norm)   # district → city + centroid (khi chưa có landmark)

    # Consume span RIÊNG từng loại: category và concept được phép cùng ăn 1 đoạn text
    # ("ăn tối" → category Nhà hàng + concept an_uong là chủ đích).
    #
    # ĐÃ ĐO nhưng chưa bật — landmark-span consumption (pre_taken=landmark span cho
    # concept): fix leak "cafe gần hồ xuân hương" ăn attr `gan_ho` rồi trả quán cạnh
    # hồ KHÁC city, NHƯNG làm P002 rơi vào dense-bait G003 (thua 0.016). Bật lại khi
    # có reranker semantic mạnh hơn phân xử được bait (LLM rerank / e5-base).
    plan.categories = _match_consuming(norm, _category_rules())
    plan.attr_concepts = _match_consuming(norm, _concept_rules())

    # Polarity: suy ra yên tĩnh + phủ định đông khách; gỡ khỏi attrs nếu lỡ match dương.
    if _NEG_QUIET.search(norm):
        plan.attr_concepts.add("yen_tinh")
        plan.neg_concepts.add("dong_khach")
    plan.attr_concepts -= plan.neg_concepts

    plan.want_pop = bool(_POP.search(norm))
    return plan
