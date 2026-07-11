"""Orchestrator: SearchService — hàm duy nhất mà API và demo gọi.

Full pipeline: BM25 ∪ dense (union pool) → multi-signal rerank. Deterministic,
offline hoàn toàn (embedding từ .npy cache, 0 LLM call), không cần network/API key.
Trả kèm signal breakdown cho từng kết quả khi explain=True (explainability).

Điểm kiến trúc: MỌI retriever (BM25, dense, +rerank) implement CÙNG Protocol
`Retriever` — bảng ablation trong eval/run_eval.py swap object mà không sửa harness.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

from src.data_loader import POI, load_pois
from src.ranking.reranker import RerankRetriever
from src.ranking.signals import haversine_km
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src import config
from src.reasoning.constraints import annotate as annotate_constraints
from src.understanding.abbreviations import expand_abbreviations
from src.understanding.diacritics import restore_diacritics
from src.understanding.rules import concept_label, extract_plan, landmark_label
from src.understanding.typo_fix import correct_typos


@runtime_checkable
class Retriever(Protocol):
    """Interface chung cho mọi tầng retrieval/ranking trong bảng ablation."""

    def search(self, query: str, k: int = 10) -> list[str]:
        """Câu query thô → list poi_id đã xếp hạng (tốt nhất đứng đầu), tối đa k."""
        ...


@dataclass
class SearchHit:
    """1 kết quả đã chấm điểm; explanation chỉ có khi explain=True."""
    poi: POI
    score: float                                  # normalize [0,1] theo trần weights
    distance_meters: int | None = None            # chỉ set khi request có lat/lon
    explanation: dict | None = None               # {plan, signals} — breakdown thật


class _SearchIndex:
    """Bundle bất biến pois + retrievers — reindex build BẢN MỚI rồi swap nguyên
    con, TUYỆT ĐỐI không mutate index đang được request đọc."""

    def __init__(self, pois: list[POI], reuse_embeddings=None, model=None):
        self.pois = pois
        self.by_id = {p.id: p for p in pois}
        bm25 = BM25Retriever(pois)
        self.dense = DenseRetriever(pois, reuse_embeddings=reuse_embeddings, model=model)
        self.reranker = RerankRetriever(pois, base=bm25, dense=self.dense)


def _clear_data_caches() -> None:
    """Xoá mọi cache derive từ POI DATA trước khi rebuild (lexicon YAML không đổi
    thì cache theo file giữ nguyên). Gồm cả cache kết quả query-level của các bước
    hiểu câu — vocab đổi thì kết quả sửa/expand có thể đổi theo."""
    from src import data_loader
    from src.ranking import signals
    from src.understanding import abbreviations, diacritics, rules, typo_fix

    data_loader._load_pg_rows.cache_clear()
    signals._NORM_ATTRS.clear()
    rules._city_rules.cache_clear()
    rules._district_rules.cache_clear()
    rules.district_centroids.cache_clear()
    abbreviations._data_vocab.cache_clear()
    abbreviations._rules.cache_clear()
    abbreviations.expand_abbreviations.cache_clear()
    typo_fix._vocabs.cache_clear()
    typo_fix.correct_typos.cache_clear()
    diacritics._maps.cache_clear()
    diacritics.restore_diacritics.cache_clear()


class SearchService:
    """Build 1 lần lúc startup (encode/cache embedding), mỗi query sau đó sub-second.

    Reindex (Phase 4a): sau khi ingestion commit Postgres, gọi reindex() — build
    _SearchIndex MỚI từ data đã cập nhật rồi HOÁN ĐỔI self._index bằng một phép
    gán (atomic dưới GIL). Request đang chạy đã chụp reference index cũ ở đầu
    search() nên không bao giờ thấy index nửa vời.
    """

    def __init__(self, pois: list[POI] | None = None):
        self._reindex_lock = threading.Lock()  # serialize các lần rebuild
        self._index = _SearchIndex(pois if pois is not None else load_pois())
        # Warmup: model encode query load lazy — ép load NGAY tại startup để
        # request đầu tiên không gánh ~7s (embedding corpus thì đã có .npy cache).
        self._index.reranker.search("warmup", k=1)

    @property
    def n_pois(self) -> int:
        return len(self._index.pois)

    def reindex(self) -> dict:
        """Rebuild index từ nguồn data (MỘT lần cho cả batch) → atomic swap.

        CHỈ gọi sau khi transaction DB commit thành công — rollback thì đừng gọi,
        index phải khớp đúng thứ đã commit. Vector POI cũ + model chuyền lại từ
        index cũ: chỉ POI mới phải encode (kiểm bằng n_encoded trả về).
        """
        with self._reindex_lock:
            old = self._index
            _clear_data_caches()
            reuse = dict(zip(old.dense.docs, old.dense.doc_embeddings))
            new = _SearchIndex(load_pois(), reuse_embeddings=reuse,
                               model=old.dense.loaded_model)
            self._index = new  # atomic swap — từ đây request mới đọc index mới
            return {"pois": len(new.pois), "encoded_new": new.dense.n_encoded}

    def search(
        self,
        query: str,
        lat: float | None = None,
        lon: float | None = None,
        limit: int = config.settings().search.default_limit,
        explain: bool = False,
        # k nội bộ lớn hơn limit để post-filter (category/radius/bbox) không làm đói kết quả
        k_internal: int = config.settings().search.k_internal,
    ) -> list[SearchHit]:
        idx = self._index  # chụp reference MỘT LẦN — request này dùng trọn snapshot
        user_coord = (lat, lon) if lat is not None and lon is not None else None
        ranked = idx.reranker.search_explained(query, k=max(limit, k_internal),
                                               user_coord=user_coord)
        plan_dict = interpreted = normalized_query = expanded_query = typo_corrected = None
        if explain:
            # Chuỗi hiểu câu — TRÙNG với reranker.preprocess_query:
            # expand viết tắt → restore dấu → typo fix (flag). Plan build từ bản cuối.
            expanded = expand_abbreviations(query)
            expanded_query = expanded if expanded != query else None
            normalized_query = restore_diacritics(expanded)
            understood = normalized_query
            if config.ENABLE_TYPO_FIX:
                fixed = correct_typos(normalized_query)
                typo_corrected = fixed if fixed != normalized_query else None
                understood = fixed
            plan = extract_plan(understood)
            plan_dict = {key: (sorted(v) if isinstance(v, set) else v)
                         for key, v in asdict(plan).items()}
            normalized_query = understood  # UI hiện bản "đã hiểu" CUỐI CÙNG
            # Plan ở dạng người-đọc-được: concept id → nhãn có dấu
            interpreted = {
                "categories": sorted(plan.categories),
                "attributes": [concept_label(c) for c in sorted(plan.attr_concepts)],
                "excluded": [concept_label(c) for c in sorted(plan.neg_concepts)],
                "city": plan.city,
                "district": plan.district,
                "landmark": landmark_label(plan.landmark) if plan.landmark else None,
            }

        hits = []
        for row in ranked:
            poi = idx.by_id[row["poi_id"]]
            hit = SearchHit(poi=poi, score=round(row["total"] / idx.reranker.max_score, 4))
            if lat is not None and lon is not None:
                hit.distance_meters = int(haversine_km(lat, lon, poi.lat, poi.lon) * 1000)
            if explain:
                hit.explanation = {
                    "plan": plan_dict,
                    "expanded_query": expanded_query,
                    "typo_corrected": typo_corrected,
                    "normalized_query": normalized_query,
                    "interpreted": interpreted,
                    "signals": {name: round(v, 4) for name, v in row["signals"].items()},
                    # Lớp reasoning: tách ràng buộc, chấm thỏa/nới TỪNG cái cho
                    # result này — annotation-only, không đổi thứ tự ranking.
                    "constraints": annotate_constraints(plan, poi),
                }
            hits.append(hit)
        return hits
