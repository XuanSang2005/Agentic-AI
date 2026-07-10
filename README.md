# Tasco Maps — AI Semantic Search & Ranking (AABW 2026, Track 2)

Search-and-ranking service cho POI tiếng Việt: query understanding (LLM planner) → hybrid retrieval (dense + BM25 + hard filter, RRF) → multi-signal re-rank (7 signals, explainable).

## Chạy nhanh

```bash
make install       # venv + dependencies
make verify-data   # xác minh dataset
make api           # FastAPI tại http://localhost:8000
make eval          # Hit@1 / MRR / Recall@3 + ablation
```

## API

- `GET /v1/search?q=...&lat=...&lon=...&limit=...` (alias: `/search`, `/v1/geocode-search`) — trả PlaceResult DTO theo `docs/tasco_api.pdf`.
- `GET /health`

## Kết quả eval

<!-- TODO Ngày 2: dán bảng ablation vào đây -->

| Pipeline | Hit@1 | MRR | Recall@3 |
|---|---|---|---|
| BM25 only | 0.833 | 0.890 | 0.861 |
| + Dense (RRF) | — | — | — |
| + Hard Filter | — | — | — |
| + Multi-signal Rerank | 0.967 | 0.983 | 0.936 |
| + LLM Planner | — | — | — |

Chi tiết kiến trúc, gotchas và build order: xem `CLAUDE.md`.
