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


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny


class DenseRetriever:
    """Implement Protocol `Retriever` (src.search) — swap được trong bảng ablation.

    Sử dụng Qdrant Local Mode (offline, embedded storage) để chạy vector search
    và hỗ trợ hybrid pre-filtering trực tiếp bằng metadata payload.
    """

    def __init__(self, pois: list[POI], model_name: str = config.EMBEDDING_MODEL):
        self._model_name = model_name
        self._model = None  # lazy — chỉ load khi cache miss hoặc lúc encode query
        docs = [p.document for p in pois]  # GIỮ DẤU
        self._doc_emb = self._load_or_encode(docs)

        # Khởi tạo Qdrant Client ở chế độ local (embedded database trong thư mục data)
        self._collection_name = "pois"
        config.QDRANT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self._client = QdrantClient(path=str(config.QDRANT_STORAGE_DIR))
        except Exception:
            # Fallback sang in-memory nếu thư mục bị lock bởi tiến trình khác (ví dụ: uvicorn)
            self._client = QdrantClient(location=":memory:")

        # Khởi tạo collection nếu chưa tồn tại
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        # Upsert POIs và payload metadata
        points = []
        for i, poi in enumerate(pois):
            points.append(PointStruct(
                id=i + 1,
                vector=self._doc_emb[i].tolist(),
                payload={
                    "id": poi.id,
                    "category": poi.category,
                    "city": poi.city,
                    "district": poi.district,
                    "price_level": poi.price_level
                }
            ))
        self._client.upsert(collection_name=self._collection_name, points=points)

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

    def search_scored(self, query: str, k: int = 10, plan=None) -> list[tuple[str, float]]:
        """Top-k (poi_id, cosine). Sử dụng local Qdrant collection với hybrid pre-filtering."""
        q_emb = self._get_model().encode(
            [f"query: {restore_diacritics(query)}"],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)[0]

        # Xây dựng bộ lọc hybrid pre-filtering dựa trên QueryPlan
        must_filters = []
        if plan is not None:
            if plan.city:
                must_filters.append(FieldCondition(key="city", match=MatchValue(value=plan.city)))
            if plan.district:
                must_filters.append(FieldCondition(key="district", match=MatchValue(value=plan.district)))
            if plan.categories:
                must_filters.append(FieldCondition(key="category", match=MatchAny(any=list(plan.categories))))

        query_filter = Filter(must=must_filters) if must_filters else None

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=q_emb.tolist(),
            query_filter=query_filter,
            limit=k
        )
        return [(r.payload["id"], float(r.score)) for r in results.points]

    def search(self, query: str, k: int = 10) -> list[str]:
        return [pid for pid, _ in self.search_scored(query, k)]


class AttributeIndex:
    """Index vector cho unique attributes của dataset — radius search thay YAML mapping.

    Dùng CHUNG model E5 với DenseRetriever (cùng embedding space). Cache riêng
    theo nội dung attribute list (hash) — data thay đổi là tự re-encode.
    """

    def __init__(self, unique_attrs: list[str], model_name: str = config.EMBEDDING_MODEL):
        self._attrs = unique_attrs
        self._model_name = model_name
        self._model = None
        self._attr_emb = self._load_or_encode(unique_attrs) if unique_attrs else np.empty((0, 0))

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _load_or_encode(self, attrs: list[str]) -> np.ndarray:
        """Cache .npy theo (model, nội dung attribute list) — attribute đổi là tự re-encode."""
        digest = hashlib.sha256(
            (self._model_name + "\n\x00".join(attrs)).encode("utf-8")).hexdigest()[:16]
        slug = self._model_name.split("/")[-1]
        cache = config.EMBEDDING_CACHE_DIR / f"attrs_{slug}_{digest}.npy"
        if cache.exists():
            return np.load(cache)
        emb = self._get_model().encode(
            [f"passage: {a}" for a in attrs],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        config.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb

    def radius_search(self, query_span: str,
                      threshold: float = config.ATTRIBUTE_SIMILARITY_THRESHOLD,
                      ) -> list[str]:
        """Tìm attributes có cosine similarity >= threshold với query_span.

        Giới hạn lấy tối đa top 2 để tránh gom các attribute nhiễu do không gian
        vector của E5-small nén sát nhau (vd: 'ăn uống' có thể kéo theo các hoạt động khác).
        """
        if not self._attrs or self._attr_emb.size == 0:
            return []
        q_emb = self._get_model().encode(
            [f"query: {query_span}"],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)[0]
        scores = np.einsum("ij,j->i", self._attr_emb, q_emb)
        matches = [(self._attrs[i], float(scores[i])) for i in range(len(self._attrs))]
        matches.sort(key=lambda x: -x[1])
        return [attr for attr, score in matches if score >= threshold][:2]


class JointMetadataIndex:
    """Index vector cho categories + attributes — similarity search trả continuous weights.

    Kết hợp cả categories và attributes vào cùng 1 embedding space. Mỗi segment query
    được so khớp với toàn bộ index; ReLu noise floor gate loại bỏ false positives.
    """

    # ReLu noise floor: similarities dưới ngưỡng này bị zeroed out
    NOISE_FLOOR = 0.835

    def __init__(self, unique_categories: list[str], unique_attrs: list[str],
                 model_name: str = config.EMBEDDING_MODEL):
        self._categories = list(unique_categories)
        self._attrs = list(unique_attrs)
        self._all_labels = self._categories + self._attrs
        self._n_cats = len(self._categories)
        self._model_name = model_name
        self._model = None
        self._emb = self._load_or_encode(self._all_labels) if self._all_labels else np.empty((0, 0))

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _load_or_encode(self, labels: list[str]) -> np.ndarray:
        digest = hashlib.sha256(
            (self._model_name + "\n\x00".join(labels)).encode("utf-8")).hexdigest()[:16]
        slug = self._model_name.split("/")[-1]
        cache = config.EMBEDDING_CACHE_DIR / f"joint_{slug}_{digest}.npy"
        if cache.exists():
            return np.load(cache)
        emb = self._get_model().encode(
            [f"passage: {l}" for l in labels],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        config.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb

    def _relu_scale(self, similarity: float) -> float:
        """ReLu noise floor gate: max(0, (sim - floor) / (1 - floor))."""
        return max(0.0, (similarity - self.NOISE_FLOOR) / (1.0 - self.NOISE_FLOOR))

    def search(self, query_span: str, top_k: int = 2
               ) -> tuple[dict[str, float], dict[str, float]]:
        """So khớp query segment vs toàn bộ categories & attributes.

        Returns:
            (category_weights, attribute_weights) — mỗi dict map label → ReLu-scaled score.
            Chỉ trả top_k matches mỗi namespace (category, attribute) có score > 0.
        """
        if not self._all_labels or self._emb.size == 0:
            return {}, {}
        q_emb = self._get_model().encode(
            [f"query: {query_span}"],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)[0]
        scores = np.einsum("ij,j->i", self._emb, q_emb)

        cat_weights: dict[str, float] = {}
        attr_weights: dict[str, float] = {}

        # Category matches (first n_cats entries)
        cat_scored = [(self._all_labels[i], float(scores[i])) for i in range(self._n_cats)]
        cat_scored.sort(key=lambda x: -x[1])
        for label, sim in cat_scored[:top_k]:
            scaled = self._relu_scale(sim)
            if scaled > 0:
                cat_weights[label] = scaled

        # Attribute matches (remaining entries)
        attr_scored = [(self._all_labels[i], float(scores[i]))
                       for i in range(self._n_cats, len(self._all_labels))]
        attr_scored.sort(key=lambda x: -x[1])
        for label, sim in attr_scored[:top_k]:
            scaled = self._relu_scale(sim)
            if scaled > 0:
                attr_weights[label] = scaled

        return cat_weights, attr_weights


class ColumnAnchorIndex:
    """Embed topic anchors (price, rating, time) — argmax competitive classification.

    Xác định một numeric phrase thuộc column nào bằng cách so sánh similarity
    với 3 anchor descriptions, chọn anchor cao nhất (argmax).
    """

    ANCHORS = {
        "price": "giá tiền, chi phí, tiền, đồng, vnd, k, triệu, rẻ, đắt, tốn",
        "rating": "điểm đánh giá, chất lượng, sao, star, rating, tốt, dở",
        "time": "thời gian, giờ đóng cửa, mở cửa, tiếng, muộn, khuya, h, g, giờ, sáng, tối, đêm",
    }

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self._model_name = model_name
        self._model = None
        self._anchor_keys = list(self.ANCHORS.keys())
        self._anchor_emb = self._encode_anchors()

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode_anchors(self) -> np.ndarray:
        texts = [f"passage: {self.ANCHORS[k]}" for k in self._anchor_keys]
        digest = hashlib.sha256(
            (self._model_name + "\n\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        slug = self._model_name.split("/")[-1]
        cache = config.EMBEDDING_CACHE_DIR / f"anchors_{slug}_{digest}.npy"
        if cache.exists():
            return np.load(cache)
        emb = self._get_model().encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        config.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb

    def classify(self, phrase: str) -> str:
        """Argmax competitive classification: trả 'price', 'rating', hoặc 'time'."""
        q_emb = self._get_model().encode(
            [f"query: {phrase}"],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)[0]
        scores = np.einsum("ij,j->i", self._anchor_emb, q_emb)
        best_idx = int(np.argmax(scores))
        return self._anchor_keys[best_idx]


