"""BM25 (rank_bm25) trên norm_document của POI — baseline lexical, dòng 1 bảng ablation.

Tokenize CỐ TÌNH đơn giản: normalize_vi (bỏ dấu) + whitespace-split, cả 2 phía
query lẫn document — nên câu KHÔNG DẤU vẫn match. Upgrade tokenizer (pyvi) để sau.
Index TẤT CẢ POI kể cả dòng G — không lọc (down-rank là việc của L3, không phải ở đây).
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.data_loader import POI, normalize_vi


class BM25Retriever:
    """Implement Protocol `Retriever` (src.search) — swap được trong bảng ablation."""

    def __init__(self, pois: list[POI]):
        self._ids = [p.id for p in pois]
        # norm_document dựng sẵn ở data_loader — dùng chung với dense sau này
        corpus = [p.norm_document.split() for p in pois]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> list[str]:
        tokens = normalize_vi(query).split()
        scores = self._bm25.get_scores(tokens)
        # Sort ổn định + tie-break theo poi_id để kết quả DETERMINISTIC tuyệt đối
        order = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))
        return [self._ids[i] for i in order[:k]]
