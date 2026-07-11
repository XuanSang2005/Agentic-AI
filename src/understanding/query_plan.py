"""Query Plan — output chuẩn của L1 (rule-based bây giờ, LLM planner sau này cùng schema).

Mọi tầng sau (signals, reranker, filter) chỉ đọc QueryPlan, không đọc query thô.
Nhớ bẫy P009: "trên đường đi Hạ Long" là ngữ cảnh, KHÔNG map vào location filter —
landmark chỉ được resolve khi có cue "gần/near" (xem rules.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryPlan:
    query: str                                   # câu gốc (giữ dấu)
    norm_query: str                              # normalize_vi(query) — để match/debug
    categories: set[str] = field(default_factory=set)     # canonical category trong data
    attr_concepts: set[str] = field(default_factory=set)  # concept id (lexicon) yêu cầu
    neg_concepts: set[str] = field(default_factory=set)   # concept id bị PHỦ ĐỊNH
    city: str | None = None
    want_pop: bool = False                       # "nổi tiếng/best/ngon" → popularity signal
    # --- location chi tiết (Bước 2) ---
    district: str | None = None
    landmark: str | None = None                  # key trong gazetteer.yaml
    resolved_coord: tuple[float, float] | None = None  # (lat, lon) từ landmark/district
    # --- Continuous soft weights (Joint Semantic Competition) ---
    category_weights: dict[str, float] = field(default_factory=dict)   # category → score
    attribute_weights: dict[str, float] = field(default_factory=dict)  # attribute → score
    negated_attribute_weights: dict[str, float] = field(default_factory=dict)  # neg attr → score
    # --- Numeric constraints (Semantic Number Routing) ---
    price_limit: float | None = None             # raw VND amount (e.g. 100000)
    price_op: str = "le"                         # "le" | "ge" | "eq"
    rating_limit: float | None = None            # target rating (e.g. 4.5)
    rating_op: str = "ge"                        # "le" | "ge" | "eq"
    time_limit_minutes: int | None = None        # closing time in minutes since midnight
    time_op: str = "ge"                          # "le" | "ge" | "eq"
    # --- Superlative sorting ---
    sort_by: str | None = None                   # "price" | "rating" | None
    sort_order: str | None = None                # "asc" | "desc" | None
    # --- Currently-open filter ---
    current_time_open: bool = False              # True when query says "đang mở cửa"
    clean_query: str | None = None               # query with sorting and open patterns removed


