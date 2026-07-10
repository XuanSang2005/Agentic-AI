"""L3 re-rank: lấy top-N candidate của base retriever, chấm lại bằng tổ hợp signal có trọng số.

Implement CÙNG Protocol Retriever → swap thẳng vào eval harness không sửa gì.
Deterministic: không LLM, tie-break theo poi_id.
"""
from __future__ import annotations

from src.data_loader import POI
from src.ranking import signals
from src.understanding.rules import extract_plan

# Trọng số mặc định (Bước 1 — chưa distance). bm25 = điểm base đã chuẩn hóa theo max
# trong candidate set; phần còn lại là signal từ QueryPlan + POI.
DEFAULT_WEIGHTS = {
    "bm25": 0.32,
    "category": 0.26,
    "attr": 0.22,
    "city": 0.12,
    "rating": 0.05,
    "pop": 0.03,
}

# Bước 2: thêm distance (landmark/district), các weight khác giảm nhẹ tương ứng.
WEIGHTS_WITH_DISTANCE = {
    "bm25": 0.28,
    "category": 0.22,
    "attr": 0.20,
    "city": 0.10,
    "distance": 0.15,
    "rating": 0.03,
    "pop": 0.02,
}

N_CANDIDATES = 30  # đủ sâu: 60 câu eval đều có đáp án trong top-30 BM25 (recall@30 ~1.0)


class RerankRetriever:
    """Rerank trên candidate của base retriever (BM25 bây giờ, hybrid sau này)."""

    def __init__(self, pois: list[POI], base, weights: dict[str, float] | None = None,
                 n_candidates: int = N_CANDIDATES):
        self._by_id = {p.id: p for p in pois}
        self._base = base
        self._weights = weights or DEFAULT_WEIGHTS
        self._n = n_candidates

    def _candidates(self, query: str) -> list[tuple[str, float]]:
        """Top-N (poi_id, base_score); fallback điểm theo rank nếu base không có score."""
        if hasattr(self._base, "search_scored"):
            return self._base.search_scored(query, k=self._n)
        ids = self._base.search(query, k=self._n)
        return [(pid, 1.0 - i / max(len(ids), 1)) for i, pid in enumerate(ids)]

    def score_breakdown(self, query: str, poi_id: str, base_norm: float) -> dict[str, float]:
        """Điểm từng signal (đã nhân weight) — dùng cho explainability và debug."""
        plan = extract_plan(query)
        poi = self._by_id[poi_id]
        parts = {"bm25": self._weights.get("bm25", 0.0) * base_norm}
        for name, w in self._weights.items():
            if name != "bm25":
                parts[name] = w * signals.SIGNAL_FUNCS[name](plan, poi)
        return parts

    def search(self, query: str, k: int = 10) -> list[str]:
        plan = extract_plan(query)
        cands = self._candidates(query)
        if not cands:
            return []
        max_base = max(score for _, score in cands) or 1.0

        scored = []
        for pid, base_score in cands:
            poi = self._by_id[pid]
            total = self._weights.get("bm25", 0.0) * (base_score / max_base)
            for name, w in self._weights.items():
                if name != "bm25":
                    total += w * signals.SIGNAL_FUNCS[name](plan, poi)
            scored.append((pid, total))

        scored.sort(key=lambda x: (-x[1], x[0]))  # tie-break poi_id → deterministic
        return [pid for pid, _ in scored[:k]]
