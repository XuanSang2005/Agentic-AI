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
