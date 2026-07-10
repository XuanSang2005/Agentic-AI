"""Orchestrator: search(query, lat?, lon?, limit) → plan → retrieve → rerank → results.

Đây là hàm duy nhất mà API và eval harness gọi. Deterministic sau bước plan.
Trả kèm signal breakdown cho từng kết quả (explainability).

Điểm kiến trúc quan trọng nhất: MỌI retriever (BM25, dense, hybrid, +rerank)
implement CÙNG Protocol `Retriever` bên dưới — nhờ đó bảng ablation trong
eval/run_eval.py chỉ việc swap object, không sửa harness.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Interface chung cho mọi tầng retrieval/ranking trong bảng ablation."""

    def search(self, query: str, k: int = 10) -> list[str]:
        """Câu query thô → list poi_id đã xếp hạng (tốt nhất đứng đầu), tối đa k."""
        ...


# TODO Ngày 1 (slice sau): nối L1 plan → L2 retrieve → L3 rerank thành hàm search()
# end-to-end cho API. Slice hiện tại dừng ở interface + BM25 baseline.
