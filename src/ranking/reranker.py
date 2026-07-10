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

# Slice dense-as-signal: dense_relevance là relevance CHÍNH; BM25 hạ mạnh (thủ phạm
# dính bait G ở P036/P042) + tách "name" = khớp chính xác tên/brand. Các weight
# structured (category/attr/city/distance/rating/pop) GIỮ NGUYÊN từ v2 — chúng đang
# bảo vệ mixed-lang/location queries, đừng đụng.
WEIGHTS_WITH_DENSE = {
    "dense": 0.32,
    "bm25": 0.06,
    "name": 0.06,
    "category": 0.22,
    "attr": 0.20,
    "city": 0.10,
    "distance": 0.15,
    "rating": 0.03,
    "pop": 0.02,
}

N_CANDIDATES = 30  # đủ sâu: 60 câu eval đều có đáp án trong top-30 BM25 (recall@30 ~1.0)


POOL_K = 25  # union pool: top-25 BM25 ∪ top-25 dense — tối đa recall cho reranker


class RerankRetriever:
    """Rerank trên candidate của base retriever; có dense → union pool 2 nguồn."""

    def __init__(self, pois: list[POI], base, weights: dict[str, float] | None = None,
                 n_candidates: int = N_CANDIDATES, dense=None, pool_k: int = POOL_K):
        self._by_id = {p.id: p for p in pois}
        self._base = base
        self._dense = dense
        self._weights = weights or (WEIGHTS_WITH_DENSE if dense else DEFAULT_WEIGHTS)
        self._n = n_candidates
        self._pool_k = pool_k

    def _candidates(self, query: str) -> list[tuple[str, float]]:
        """Top-N (poi_id, base_score); fallback điểm theo rank nếu base không có score."""
        if hasattr(self._base, "search_scored"):
            return self._base.search_scored(query, k=self._n)
        ids = self._base.search(query, k=self._n)
        return [(pid, 1.0 - i / max(len(ids), 1)) for i, pid in enumerate(ids)]

    @staticmethod
    def _minmax_in_pool(scores: dict[str, float], pool: list[str]) -> dict[str, float]:
        """Normalize [0,1] TRONG pool; kênh phẳng (max=min) → 0.5 trung tính."""
        vals = [scores.get(pid, 0.0) for pid in pool]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return {pid: 0.5 for pid in pool}
        return {pid: (scores.get(pid, 0.0) - lo) / (hi - lo) for pid in pool}

    def _score_pool(self, query: str,
                    user_coord: tuple[float, float] | None = None
                    ) -> list[tuple[str, float, dict[str, float]]]:
        """Union pool BM25 ∪ dense → [(poi_id, total, breakdown weighted-per-signal)].

        Nguồn sự thật DUY NHẤT của điểm số: search() và search_explained()
        đều đi qua đây — explain không bao giờ lệch khỏi ranking thật.

        user_coord: focus point từ API (?lat/lon — "local ranking" theo PDF). CHỈ làm
        fallback khi query text không tự resolve được location (landmark/district giữ
        ưu tiên). Eval không truyền → không ảnh hưởng con số eval.
        """
        n_all = len(self._by_id)
        bm25_all = self._base.search_scored(query, k=n_all)
        dense_all = self._dense.search_scored(query, k=n_all)
        pool = sorted({pid for pid, _ in bm25_all[:self._pool_k]}
                      | {pid for pid, _ in dense_all[:self._pool_k]})
        bm25_n = self._minmax_in_pool(dict(bm25_all), pool)
        dense_n = self._minmax_in_pool(dict(dense_all), pool)

        plan = extract_plan(query)
        if user_coord is not None and plan.resolved_coord is None:
            plan.resolved_coord = user_coord
        scored = []
        for pid in pool:
            poi = self._by_id[pid]
            parts = {
                "bm25_relevance": self._weights.get("bm25", 0.0) * bm25_n[pid],
                "dense_relevance": self._weights.get("dense", 0.0) * dense_n[pid],
            }
            for name, w in self._weights.items():
                if name not in ("bm25", "dense"):
                    parts[name] = w * signals.SIGNAL_FUNCS[name](plan, poi)
            scored.append((pid, sum(parts.values()), parts))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    @property
    def max_score(self) -> float:
        """Tổng weights — trần lý thuyết của total, để chuẩn hóa score hiển thị [0,1]."""
        return sum(self._weights.values())

    def search_explained(self, query: str, k: int = 10,
                         user_coord: tuple[float, float] | None = None) -> list[dict]:
        """Top-k kèm breakdown từng signal (explainability cho API ?explain=true)."""
        if self._dense is None:
            raise ValueError("search_explained cần cấu hình full pipeline (dense != None)")
        return [{"poi_id": pid, "total": total, "signals": parts}
                for pid, total, parts in self._score_pool(query, user_coord)[:k]]

    def search(self, query: str, k: int = 10) -> list[str]:
        if self._dense is not None:
            return [pid for pid, _, _ in self._score_pool(query)[:k]]
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
