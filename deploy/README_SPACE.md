---
title: Tasco Semantic Search
emoji: 🗺️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Tasco Maps — AI Semantic Search & Ranking (AABW 2026, Track 2)

Search-and-ranking service cho POI tiếng Việt — **deterministic, self-contained, 0 LLM call**.
Model (multilingual-e5-small) và embedding cache được bake vào image lúc build;
runtime chạy với `HF_HUB_OFFLINE=1`, không network call nào.

- **Demo UI**: trang chủ Space này (`/`) — bấm preset chips, xem query plan + signal breakdown.
- **API**: `GET /v1/search?q=...&explain=true` (alias `/search`, `/v1/geocode-search`) — PlaceResult DTO theo contract `docs/tasco_api.pdf`; `GET /health`; OpenAPI tại `/docs`.
- **Kết quả**: Hit@1 1.000 / MRR 1.000 trên public set (60 câu), 0.950 trên stress set unseen-phrasing; G-above-gold = 0.

Ví dụ:

```
GET /v1/search?q=cafe%20c%C3%B3%20wifi%20g%E1%BA%A7n%20h%E1%BB%93%20g%C6%B0%C6%A1m&limit=3&explain=true
```

Source đầy đủ (eval harness, stress instrument, lexicon, Dart adapter) nằm ngay trong Space repo này.
Chi tiết kiến trúc & ablation: xem phần dưới của repo (`CLAUDE.md`, `eval/`).
