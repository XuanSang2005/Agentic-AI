"""Rule-based extractor: query thô → QueryPlan (category, attr concepts, city, polarity, pop).

Chạy TRƯỚC và độc lập với LLM — cũng là planner chính ở deterministic mode.
Mọi matching trên text đã normalize_vi (bỏ dấu), word-boundary 2 đầu để tránh
match giữa từ ("hn" không ăn vào "hnx", "dong" không ăn vào "dong da" nhờ surface đa từ).

v2 (dynamic-vector-attributes): attribute matching chuyển từ static YAML rules
sang subtractive parsing + vector radius search. `attribute_concepts.yaml` không
được đọc ở bất kỳ code path nào — file giữ lại trong repo để tham khảo.
"""
from __future__ import annotations

import re
import statistics
from functools import lru_cache

import yaml

from src import config
from src.data_loader import load_pois, normalize_vi
from src.understanding.diacritics import restore_diacritics
from src.understanding.query_plan import QueryPlan

# --- City: pattern trên text đã bỏ dấu. Thứ tự = ưu tiên khi (hiếm) match nhiều city.
_CITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Hà Nội", re.compile(r"(?<![a-z0-9])(ha noi|hanoi|hn)(?![a-z0-9])")),
    ("TP.HCM", re.compile(r"(?<![a-z0-9])(tp\.? ?hcm|hcm|sai gon|saigon|ho chi minh|sg)(?![a-z0-9])")),
    ("Đà Nẵng", re.compile(r"(?<![a-z0-9])(da nang|danang)(?![a-z0-9])")),
    ("Đà Lạt", re.compile(r"(?<![a-z0-9])(da lat|dalat)(?![a-z0-9])")),
]

# --- Polarity: dynamic negation cues and antonym mapping ---
_NEGATION_CUES = re.compile(
    r"\b(?P<cue>khong\s+qua|khong\s+co|khong(?!\s+(?:gian|khi))|chua|tranh|dung)\s+"
    r"(?:\b(?:de|va|nhung|gan|tai|o|cho|co|ma|noi)\b\s+)?"
    r"(?P<target>(?:(?!\b(?:de|va|nhung|gan|tai|o|cho|co|ma|noi)\b)[a-z0-9]+)"
    r"(?:\s+(?:(?!\b(?:de|va|nhung|gan|tai|o|cho|co|ma|noi)\b)[a-z0-9]+)){0,2})\b"
)
_ANTONYM_MAP = {
    "đông khách": "yên tĩnh",
    "ồn ào": "yên tĩnh",
    "náo nhiệt": "yên tĩnh",
    "đắt đỏ": "giá rẻ",
    "đắt": "giá rẻ",
}

# --- Popularity: cờ bật popularity_score, KHÔNG phải attribute.
_POP = re.compile(r"(?<![a-z0-9])(noi tieng|famous|best|ngon|hot|dang di|dang den)(?![a-z0-9])")

# --- Stop words: loại bỏ khi subtractive parsing, TRÊN TEXT ĐÃ NORMALIZE.
_STOP_WORDS = frozenset({
    # Verbs of search / intent
    "can", "tim", "muon", "chi", "di", "hoi", "xin",
    # Prepositions / proximity cues
    "o", "tai", "gan", "sat", "canh", "xung", "quanh", "phia",
    "tren", "duoi", "ben",
    # Grammatical fillers
    "co", "la", "va", "de", "nhung", "cac", "mot", "vai",
    "nhat", "noi", "ma", "thi", "cung", "duoc", "bi", "boi",
    # Question words / particles
    "nao", "dau", "gi", "sao", "khong", "day", "kia",
    # Demonstratives / misc
    "nay", "do", "ay", "qua", "rat", "lam", "nhu", "voi",
    # Near cues already handled by landmark detection
    "near",
})


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
def _category_surface_norms() -> frozenset[str]:
    """Tất cả surface forms (normalized) của categories — để subtractive parsing."""
    raw = yaml.safe_load(config.CATEGORIES_YAML.read_text(encoding="utf-8"))
    surfaces = set()
    for entry in raw.values():
        for syn in entry["synonyms"]:
            surfaces.add(normalize_vi(str(syn)))
    return frozenset(surfaces)


def landmark_label(key: str) -> str:
    return _landmark_labels().get(key, key)


@lru_cache(maxsize=1)
def _landmark_labels() -> dict[str, str]:
    """gazetteer key → tên hiển thị (name đầu tiên, CÓ DẤU)."""
    raw = yaml.safe_load(config.GAZETTEER_YAML.read_text(encoding="utf-8"))
    return {key: str(entry["names"][0]) for key, entry in raw.items()}


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
def _gazetteer_surface_norms() -> frozenset[str]:
    """Tất cả surface forms (normalized) của landmarks — để subtractive parsing."""
    raw = yaml.safe_load(config.GAZETTEER_YAML.read_text(encoding="utf-8"))
    surfaces = set()
    for entry in raw.values():
        for n in entry["names"]:
            surfaces.add(normalize_vi(str(n)))
    return frozenset(surfaces)


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
def _district_surface_norms() -> frozenset[str]:
    """Tất cả surface forms (normalized) của districts — để subtractive parsing."""
    pairs = {(p.district, p.city) for p in load_pois() if p.district}
    surfaces = set()
    for district, _ in pairs:
        surfaces.add(normalize_vi(district))
        m = re.fullmatch(r"quan (\d+)", normalize_vi(district))
        if m:
            surfaces |= {f"q{m.group(1)}", f"q.{m.group(1)}", f"district {m.group(1)}"}
    return frozenset(surfaces)


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


def _match_consuming_tracked(norm_text: str, rules: list[tuple[re.Pattern, str]],
                             pre_taken: list[tuple[int, int]] | None = None,
                             ) -> tuple[set[str], list[tuple[int, int]]]:
    """Như _match_consuming nhưng TRẢ THÊM danh sách span đã tiêu thụ."""
    taken: list[tuple[int, int]] = list(pre_taken or [])
    out: set[str] = set()
    for pat, value in rules:
        for m in pat.finditer(norm_text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            out.add(value)
    return out, taken


# --- Subtractive parsing ---

def _subtract_known_spans(norm: str, consumed_spans: list[tuple[int, int]]) -> str:
    """Loại bỏ các span đã consumed, stop words → trả lại candidate attribute text.

    Làm việc trên TEXT ĐÃ NORMALIZE. Kết quả là các từ còn lại không thuộc
    category/city/landmark/district/stop words.
    """
    # Build mask: True = character available
    mask = [True] * len(norm)
    for s, e in consumed_spans:
        for i in range(s, min(e, len(norm))):
            mask[i] = False
    # Reconstruct text from available characters
    remaining = "".join(c if mask[i] else " " for i, c in enumerate(norm))
    # Filter stop words from remaining tokens
    words = remaining.split()
    words = [w for w in words if w not in _STOP_WORDS]
    return " ".join(words).strip()


def _collect_city_spans(norm: str, plan: QueryPlan) -> list[tuple[int, int]]:
    """Tìm lại span của city match trên text normalized."""
    spans = []
    if plan.city is not None:
        for _, pat in _CITY_PATTERNS:
            m = pat.search(norm)
            if m:
                spans.append(m.span())
                break
    return spans


def _collect_entity_word_spans(norm: str, surfaces: frozenset[str]) -> list[tuple[int, int]]:
    """Tìm span cho mỗi surface form trong text normalized."""
    spans = []
    for surface in sorted(surfaces, key=len, reverse=True):
        pat = _boundary_pattern(surface)
        for m in pat.finditer(norm):
            s, e = m.span()
            if not any(s < te and e > ts for ts, te in spans):
                spans.append((s, e))
    return spans


# --- Superlative sort patterns (normalized / no-diacritics) ---
_SORT_PATTERNS = [
    # price ascending (cheapest): e.g. "rẻ nhất", "giá rẻ nhất", "giá thấp nhất"
    (re.compile(r"\b(?<!danh\s)gia\s+(?:re|thap)\s+nhat\b|\b(?:chi\s+phi|tien)\s+(?:re|thap)\s+nhat\b|\bre\s+nhat\b"), "price", "asc"),
    (re.compile(r"\b(?<!danh\s)(?:gia|chi\s+phi|tien)?\s*(?:it\s+tien|it\s+ton)\s+nhat\b"), "price", "asc"),
    # price descending (most expensive): e.g. "đắt nhất", "giá đắt nhất", "giá cao nhất"
    (re.compile(r"\b(?<!danh\s)gia\s+(?:dat|cao)\s+nhat\b|\b(?:chi\s+phi|tien)\s+(?:dat|cao)\s+nhat\b|\bdat\s+nhat\b"), "price", "desc"),
    # rating descending (best rated): e.g. "tốt nhất", "cao nhất", "đánh giá cao nhất", "sao cao nhất"
    (re.compile(r"\b(?:danh\s+gia\s+|diem\s+|sao\s+|star\s+)?(?:tot|cao)\s+nhat\b"), "rating", "desc"),
    # rating ascending (worst rated): e.g. "thấp nhất", "đánh giá thấp nhất", "tệ nhất"
    (re.compile(r"\b(?:danh\s+gia\s+|diem\s+|sao\s+|star\s+)?(?:thap|te)\s+nhat\b"), "rating", "asc"),
]

# --- Currently-open pattern (normalized) ---
_OPEN_NOW_RE = re.compile(r"\bdang\s+mo\s+cua\b")


def _extract_sort_intent(plan: QueryPlan, norm: str) -> str:
    """Detect superlative sort phrases, set plan.sort_by/sort_order, subtract from norm."""
    for pat, col, order in _SORT_PATTERNS:
        m = pat.search(norm)
        if m:
            plan.sort_by = col
            plan.sort_order = order
            # Blank out matched span with spaces to preserve offsets
            norm = norm[:m.start()] + " " * (m.end() - m.start()) + norm[m.end():]
            break
    return norm


def _extract_open_now(plan: QueryPlan, norm: str) -> str:
    """Detect 'đang mở cửa', set plan.current_time_open, subtract from norm."""
    m = _OPEN_NOW_RE.search(norm)
    if m:
        plan.current_time_open = True
        norm = norm[:m.start()] + " " * (m.end() - m.start()) + norm[m.end():]
    return norm


def _extract_negated_concepts(plan: QueryPlan, norm: str, attr_index) -> str:
    """Detect negation patterns, extract the target words, and map to negative/positive attributes dynamically."""
    if attr_index is None:
        return norm

    matches = list(_NEGATION_CUES.finditer(norm))
    spans_to_remove = []
    
    for m in matches:
        cs, ce = m.span("cue")
        if cs != -1 and ce != -1:
            accented_cue = plan.query[cs:ce].strip().lower()
            if accented_cue == "tranh":
                continue

        ts, te = m.span("target")
        if ts != -1 and te != -1:
            accented_target = plan.query[ts:te].strip()
            if accented_target:
                # Find closest attribute in the database via vector search
                matched_attrs = attr_index.radius_search(accented_target)[:1]
                if matched_attrs:
                    # Add to negated concepts and set weights
                    plan.neg_concepts.update(matched_attrs)
                    for attr in matched_attrs:
                        plan.negated_attribute_weights[attr] = 1.0
                        
                        # Antonym check for semantic routing
                        antonym = _ANTONYM_MAP.get(attr)
                        if antonym:
                            # Vector search antonym in db
                            positive_matches = attr_index.radius_search(antonym)
                            plan.attr_concepts.update(positive_matches)
                            for pos_attr in positive_matches:
                                plan.attribute_weights[pos_attr] = 1.0
            
            # Record span to replace with spaces
            spans_to_remove.append(m.span())
            
    # Subtractive query parsing
    result = list(norm)
    for s, e in spans_to_remove:
        for i in range(s, min(e, len(result))):
            result[i] = " "
    return "".join(result)


def extract_plan(query: str, attr_index=None, joint_index=None, column_anchor=None) -> QueryPlan:
    """Query thô → QueryPlan. Deterministic, không LLM, không đọc expected_ids.

    attr_index: AttributeIndex (từ dense.py) — khi có, attribute matching chạy
    subtractive parsing + vector radius search thay vì static YAML rules.

    joint_index: JointMetadataIndex — khi có, semantic competition chạy trên
    categories + attributes, trả continuous weights.

    column_anchor: ColumnAnchorIndex — khi có, numeric phrase extraction + E5 topic
    classification hoạt động.
    """
    norm = normalize_vi(query)
    plan = QueryPlan(query=query, norm_query=norm)

    # --- Superlative sort & open-now extraction (BEFORE all existing steps) ---
    norm = _extract_sort_intent(plan, norm)
    norm = _extract_open_now(plan, norm)
    norm = _extract_negated_concepts(plan, norm, attr_index)

    # --- Numeric constraint extraction (TRƯỚC mọi thứ khác) ---
    if column_anchor is not None:
        norm = _extract_numeric_constraints(plan, norm, column_anchor)

    for city, pat in _CITY_PATTERNS:
        if pat.search(norm):
            plan.city = city
            break

    landmark_span = _detect_landmark(plan, norm)   # cần cue "gần/near", có city guard
    _detect_district(plan, norm)   # district → city + centroid (khi chưa có landmark)

    # Category: match + track consumed spans
    plan.categories, cat_spans = _match_consuming_tracked(norm, _category_rules())

    if attr_index is not None:
        # --- Subtractive parsing: loại bỏ entities đã nhận diện, giữ candidate attrs ---
        consumed = list(cat_spans)
        consumed.extend(_collect_city_spans(norm, plan))
        if landmark_span is not None:
            consumed.append(landmark_span)
            # Also consume the near cue word before the landmark
            pre_window = norm[max(0, landmark_span[0] - 15):landmark_span[0]]
            cue_m = _NEAR_CUE.search(pre_window)
            if cue_m:
                offset = max(0, landmark_span[0] - 15)
                consumed.append((offset + cue_m.start(), offset + cue_m.end()))
        consumed.extend(_collect_entity_word_spans(norm, _district_surface_norms()))

        remaining = _subtract_known_spans(norm, consumed)

        if remaining.strip():
            # Restore accents on the isolated candidate text to ensure high E5 vector similarity
            accented_remaining = restore_diacritics(remaining)
            matched_attrs: set[str] = set()
            # Try the full remaining text first
            full_matches = attr_index.radius_search(accented_remaining)
            matched_attrs.update(full_matches)
            # Also try individual words/sub-phrases if multiple words remain
            rem_words = accented_remaining.split()
            if len(rem_words) > 1:
                for w in rem_words:
                    if len(w) >= 2:
                        matched_attrs.update(attr_index.radius_search(w))
            plan.attr_concepts = matched_attrs

            # --- Joint Semantic Competition (continuous weights) ---
            if joint_index is not None:
                _run_semantic_competition(plan, accented_remaining, joint_index)
    else:
        # Fallback: no attr_index → empty attributes (backward compat for bare tests)
        plan.attr_concepts = set()

    plan.attr_concepts -= plan.neg_concepts

    plan.want_pop = bool(_POP.search(norm))
    plan.clean_query = restore_diacritics(" ".join(norm.split()))
    return plan


# --- Numeric Segment Extraction ---

# Operator keywords (normalized) → comparison direction
_OP_LE = frozenset({"duoi", "truoc", "kem", "nho hon", "it hon", "duoi muc", "thap hon"})
_OP_GE = frozenset({"tren", "sau", "hon", "lon hon", "nhieu hon", "tren muc", "cao hon"})
_OP_EQ = frozenset({"tam", "khoang", "dung", "bang"})

# Grouped regex: (operator_prefix)? (digits) (unit_suffix)?
_NUMERIC_RE = re.compile(
    r"(?:(?P<op>duoi|tren|tam|khoang|sau|truoc|hon|kem)\s+)?"
    r"(?P<num>\d+(?:\.\d+)?)"
    r"(?:\s*(?P<unit>trieu|tr|star|sao|gio|g|h|k|am|pm|sang|chieu|toi|dem))?"
)

# VND → price_level mapping thresholds
_PRICE_LEVEL_MAP = [
    (50_000, 1),
    (150_000, 2),
    (300_000, 3),
]


def _vnd_to_price_level(vnd: float) -> int:
    """Map raw VND amount to database price_level (1-4)."""
    for threshold, level in _PRICE_LEVEL_MAP:
        if vnd <= threshold:
            return level
    return 4


def _parse_op(op_text: str | None) -> str:
    """Parse operator keyword to comparison direction."""
    if op_text is None:
        return "le"  # default: "dưới" (under) for price, "trên" (over) for rating
    op_norm = op_text.strip().lower()
    if op_norm in _OP_LE:
        return "le"
    if op_norm in _OP_GE:
        return "ge"
    if op_norm in _OP_EQ:
        return "eq"
    return "le"


def _extract_numeric_constraints(plan: QueryPlan, norm: str,
                                 column_anchor) -> str:
    """Extract numeric phrases from normalized query, classify topic, bind to plan.

    Returns the query string with numeric phrases subtracted.
    """
    matches = list(_NUMERIC_RE.finditer(norm))
    if not matches:
        return norm

    # Process matches in reverse order to preserve character positions during subtraction
    spans_to_remove: list[tuple[int, int]] = []

    for m in matches:
        s, e = m.span()
        # Skip numeric matches that are parts of a fraction or slash term (e.g. "24/7")
        if (s > 0 and norm[s - 1] == "/") or (e < len(norm) and norm[e] == "/"):
            continue

        full_phrase = m.group(0).strip()
        op_text = m.group("op")
        num_str = m.group("num")
        unit = m.group("unit")
        num_val = float(num_str)

        # Safeguard 1: Skip if number is < 10 and has neither op nor unit (e.g. "quan 1", "nguoi 3")
        if op_text is None and unit is None and num_val < 10:
            continue

        # Safeguard 2: Skip if preceded by address/district/ward markers (e.g. "quan 1", "p 5", "duong 3")
        prefix_part = norm[:s].rstrip(" .")
        last_word_match = re.search(r"[a-z0-9]+$", prefix_part)
        if last_word_match:
            last_word = last_word_match.group(0)
            if last_word in {"quan", "q", "phuong", "p", "duong", "d", "so", "ngo", "ngach", "hem"}:
                continue

        accented_phrase = plan.query[s:e]
        topic = column_anchor.classify(accented_phrase)

        # Override misclassified rating topic if number is not a valid rating (> 5)
        if topic == "rating" and num_val > 5.0:
            if num_val <= 24.0:
                topic = "time"
            else:
                topic = "price"

        # Override misclassified time topic if number is not a valid hour (> 24) and has no unit
        if topic == "time" and unit is None and num_val > 24.0:
            topic = "price"

        op = _parse_op(op_text)

        if topic == "price":
            # Convert to VND: handle k (thousand) and tr/trieu (million) multipliers
            if unit in ("k",):
                plan.price_limit = num_val * 1_000
            elif unit in ("tr", "trieu"):
                plan.price_limit = num_val * 1_000_000
            else:
                plan.price_limit = num_val * 1_000  # assume k if no unit
            plan.price_op = op if op_text else "le"  # default: price is "under X"

        elif topic == "rating":
            plan.rating_limit = num_val
            plan.rating_op = op if op_text else "ge"  # default: rating is "over X"

        elif topic == "time":
            # Convert to minutes since midnight
            hour = int(num_val)
            # Handle AM/PM and Vietnamese time-of-day markers
            if unit in ("sang",) and hour >= 1 and hour <= 12:
                minutes = hour * 60  # sáng: AM
            elif unit in ("chieu", "toi") and hour >= 1 and hour <= 12:
                minutes = (hour + 12) * 60 if hour != 12 else 12 * 60
            elif unit in ("dem",) and hour >= 1 and hour <= 6:
                minutes = hour * 60  # đêm: late night, treated as next day
            else:
                minutes = hour * 60
            plan.time_limit_minutes = minutes
            plan.time_op = op if op_text else "ge"  # default: open "until at least X"

        spans_to_remove.append((s, e))

    # Subtract all numeric phrases from the query
    result = list(norm)
    for s, e in spans_to_remove:
        for i in range(s, min(e, len(result))):
            result[i] = " "
    return "".join(result)


# --- Hierarchical Semantic Matching ---

def _run_semantic_competition(plan: QueryPlan, accented_remaining: str,
                              joint_index) -> None:
    """Hierarchically match remaining text against JointMetadataIndex to reduce noise.

    1. First try matching the entire remaining phrase. If a strong match is found,
       accept it and stop.
    2. Otherwise, fall back to sliding window trigrams and bigrams.
    """
    text = accented_remaining.strip()
    if not text:
        return

    # Phase 1: Try full phrase match first
    cat_w, attr_w = joint_index.search(text)
    # Check if there is any strong match (scaled score >= 0.15)
    has_strong_match = any(w >= 0.15 for w in cat_w.values()) or any(w >= 0.15 for w in attr_w.values())
    if has_strong_match:
        plan.category_weights.update(cat_w)
        plan.attribute_weights.update(attr_w)
        return

    # Phase 2: Fall back to bigram/trigram segmentation if the full phrase is composite
    words = text.split()
    if len(words) <= 1:
        plan.category_weights.update(cat_w)
        plan.attribute_weights.update(attr_w)
        return

    segments = []
    # Trigrams
    for i in range(len(words) - 2):
        segments.append(" ".join(words[i:i + 3]))
    # Bigrams
    for i in range(len(words) - 1):
        segments.append(" ".join(words[i:i + 2]))

    for segment in segments:
        cw, aw = joint_index.search(segment)
        for label, weight in cw.items():
            if weight > plan.category_weights.get(label, 0.0):
                plan.category_weights[label] = weight
        for label, weight in aw.items():
            if weight > plan.attribute_weights.get(label, 0.0):
                plan.attribute_weights[label] = weight


