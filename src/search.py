"""Orchestrator: SearchService — hàm duy nhất mà API và demo gọi.

Full pipeline: BM25 ∪ dense (union pool) → multi-signal rerank. Deterministic,
offline hoàn toàn (embedding từ .npy cache, 0 LLM call), không cần network/API key.
Trả kèm signal breakdown cho từng kết quả khi explain=True (explainability).

Điểm kiến trúc: MỌI retriever (BM25, dense, +rerank) implement CÙNG Protocol
`Retriever` — bảng ablation trong eval/run_eval.py swap object mà không sửa harness.
"""
from __future__ import annotations

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


class SearchService:
    """Build 1 lần lúc startup (encode/cache embedding), mỗi query sau đó sub-second."""

    def __init__(self, pois: list[POI] | None = None):
        self._pois = pois if pois is not None else load_pois()
        self._by_id = {p.id: p for p in self._pois}
        bm25 = BM25Retriever(self._pois)
        dense = DenseRetriever(self._pois)
        self._reranker = RerankRetriever(self._pois, base=bm25, dense=dense)
        # Warmup: model encode query load lazy — ép load NGAY tại startup để
        # request đầu tiên không gánh ~7s (embedding corpus thì đã có .npy cache).
        self._reranker.search("warmup", k=1)

    @property
    def n_pois(self) -> int:
        return len(self._pois)

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
        user_coord = (lat, lon) if lat is not None and lon is not None else None
        ranked = self._reranker.search_explained(query, k=max(limit, k_internal),
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
            poi = self._by_id[row["poi_id"]]
            hit = SearchHit(poi=poi, score=round(row["total"] / self._reranker.max_score, 4))
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
