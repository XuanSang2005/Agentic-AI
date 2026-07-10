"""Semantic cache cho QueryPlan đã hiểu.

- Có NGƯỠNG khoảng cách embedding: đủ gần mới trả cache, xa quá → gọi LLM.
- Cache phải được nạp sẵn (warm) trước khi chấm eval — không vừa chấm vừa điền.
"""

# TODO Ngày 2
