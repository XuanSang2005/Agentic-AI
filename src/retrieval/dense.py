"""Dense retrieval: multilingual-e5-small, cosine brute-force (111 POI — không cần ANN/FAISS).

⚠ ENCODE GIỮ NGUYÊN DẤU tiếng Việt — NGƯỢC với BM25 (bỏ dấu 2 phía):
e5 train trên text tự nhiên có dấu, bỏ dấu làm HẠI dense. Normalization
khác nhau theo retriever là CHỦ ĐÍCH: BM25 dùng poi.norm_document,
dense dùng poi.document (bản gốc có dấu, cùng build từ data_loader).

Prefix e5 bắt buộc: "passage: " khi index, "query: " khi search — thiếu là tụt điểm.
Embedding cache ra .npy (key = model + hash corpus) để load lại nhanh, deterministic.
"""
from __future__ import annotations

import hashlib

import numpy as np

from src import config
from src.data_loader import POI
from src.understanding.diacritics import restore_diacritics


class DenseRetriever:
    """Implement Protocol `Retriever` (src.search) — swap được trong bảng ablation."""

    def __init__(self, pois: list[POI], model_name: str = config.EMBEDDING_MODEL):
        self._ids = [p.id for p in pois]
        self._model_name = model_name
        self._model = None  # lazy — chỉ load khi cache miss hoặc lúc encode query
        docs = [p.document for p in pois]  # GIỮ DẤU
        self._doc_emb = self._load_or_encode(docs)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _load_or_encode(self, docs: list[str]) -> np.ndarray:
        """Cache .npy theo (model, nội dung corpus) — data đổi là tự re-encode."""
        digest = hashlib.sha256(
            (self._model_name + "\n\x00".join(docs)).encode("utf-8")).hexdigest()[:16]
        slug = self._model_name.split("/")[-1]
        cache = config.EMBEDDING_CACHE_DIR / f"{slug}_{digest}.npy"
        if cache.exists():
            return np.load(cache)
        emb = self._get_model().encode(
            [f"passage: {d}" for d in docs],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        assert np.isfinite(emb).all(), "embedding có NaN/inf — model/encode hỏng"
        config.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb

    def search_scored(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (poi_id, cosine). Embeddings đã L2-normalize → cosine = dot.

        restore_diacritics CHỈ ở đây (query-side dense): e5 yếu với câu không dấu.
        BM25/rules vẫn nhận query gốc — không rò sang nhánh khác. Câu có dấu /
        English → restore là no-op. Cache .npy phía POI không đổi.
        """
        q_emb = self._get_model().encode(
            [f"query: {restore_diacritics(query)}"],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)[0]
        # einsum thay vì `@`: Accelerate BLAS (numpy 2.x/macOS) bắn RuntimeWarning
        # divide-by-zero giả trong matmul dù input/output finite (đã verify identical)
        scores = np.einsum("ij,j->i", self._doc_emb, q_emb)
        order = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))
        return [(self._ids[i], float(scores[i])) for i in order[:k]]

    def search(self, query: str, k: int = 10) -> list[str]:
        return [pid for pid, _ in self.search_scored(query, k)]
