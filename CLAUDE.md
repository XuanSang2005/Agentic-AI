# AABW 2026 — Track 2: AI Semantic Search & Ranking cho Tasco Maps

Mục tiêu: **TOP 5**. Deliverable = search-and-ranking service (FastAPI) + con số eval thuyết phục.
API contract chuẩn: `docs/tasco_api.pdf`. Dataset: `data/ai_maps_track2_dataset_participants.xlsx`.

## Lệnh

```bash
make install      # venv + pip install -r requirements.txt
make api          # uvicorn src.api.main:app --port 8000
make eval         # eval/run_eval.py — Hit@1/MRR/Recall@3 + ablation
make verify-data  # eval/verify_dataset.py — kiểm lại mọi con số bên dưới
make test         # pytest tests/
```

## Sự thật về data (đã xác minh bằng `eval/verify_dataset.py` — đừng tin, hãy chạy lại)

- **POI_Dataset = 111 dòng**: **39 POI THẬT** (C:8 cafe, R:10 nhà hàng, H:5 khách sạn, A:5 tham quan, S:7 tiện ích, M:4 mall) + **72 dòng SYNTHETIC prefix G**.
- ⚠️ **72 dòng G là DISTRACTOR thuần**: 0/60 câu eval có đáp án là G (đã verify). Dấu hiệu nhận biết: địa chỉ giả "Đường Trung Tâm", attribute mâu thuẫn — **5 dòng G gắn `gần biển` ở city không có biển** (G007/G008/G009 Hà Nội, G025 TP.HCM, G062 Đà Lạt). Robustness với G bị chấm nặng.
- **Vế phải để KHỚP attribute = token thật trong cột `POI_Dataset.attributes`**: 82 token duy nhất toàn dataset, trong đó **80 xuất hiện ở POI thật**; 2 token chỉ có ở dòng G (`cửa hàng tiện lợi`, `giá dưới 2 triệu`). **KHÔNG** dùng 10 nhãn `Attribute_Taxonomy` làm vế phải — taxonomy chỉ là 10 khái niệm khoảng-cách-ngôn-ngữ-rộng-nhất.
- Data tự nhiễu synonym (`wifi` vs `wifi mạnh`; `gia đình` vs `phù hợp gia đình` vs `phù hợp trẻ em`; `giá rẻ` vs `giá hợp lý` vs `giá vừa phải`) → chuẩn hóa về **~25–30 concept** trong `lexicon/attribute_concepts.yaml`, mỗi concept nhớ danh sách token thành viên.
- **7 Ranking_Signals**: relevance, distance, rating, popularity, review, freshness, business_attributes.
- **Public_Evaluation = 60 câu**: 25 Hard / 30 Medium / 5 Easy. Category: Semantic 18 + Attribute 14 (quá nửa), Intent 8, Location-Aware 6, Discovery 6, MixedLanguage 5, Category 2, POI 1. **Semantic-Hard + Intent (~20 câu, gần như toàn Hard) là chỗ phân định top 5.**
- **METRIC**: 43/60 câu chỉ có 1 đáp án → chấm bằng **Hit@1 / MRR + Recall@3**, luôn báo cáo chia theo difficulty & category.
- **Location xuất hiện 43/60 câu** (trong `ranking_signals_to_use`) → location handling là bắt buộc, không phải nice-to-have.
- **Gazetteer bắt buộc** (landmark không có toạ độ trong data): Hồ Gươm, phố đi bộ Nguyễn Huệ, sông Hàn, chợ Bến Thành, hồ Xuân Hương, biển Mỹ Khê, Duy Tân → `lexicon/gazetteer.yaml`.
- **Bẫy P009**: "cây xăng có toilet **trên đường đi Hạ Long**" — vế Hạ Long là ngữ cảnh chuyến đi, **KHÔNG phải hard filter location** (đáp án là trạm xăng Cầu Giấy, Hà Nội).

## Kiến trúc 3 lớp

- **L1 Query Understanding** (`src/understanding/`): LLM trích ý → **Query Plan** kiểu `{category, attributes[], location{city,district,landmark}, time, price, ranking_bias, language}`. Lexicon làm **WHITELIST** + few-shot ép LLM chỉ nhả token trong 82 nhãn thật. Regex lo location/time/price. Embedding fallback cho câu lạ.
- **L2 Hybrid Retrieval** (`src/retrieval/`): dense (multilingual-e5 hoặc bge-m3) + BM25 (`rank_bm25`) + structured **HARD FILTER** (city/district/category/time/price), fuse bằng **RRF**. FAISS `IndexFlatIP` — chỉ ~111 POI nên brute-force, exact.
- **L3 Multi-signal Re-rank** (`src/ranking/`): kết hợp 7 signal, **trọng số điều kiện theo `query_category`**, tune bằng coordinate ascent trên 60 câu (leave-one-out). Optional: LLM reason-rerank top-8 cho câu Intent. Response luôn kèm **signal breakdown** (explainability).

## Kỷ luật LLM

- **CHỈ gọi LLM ở L1** (1 lần, ở input). Mọi thứ sau đó (lọc, khoảng cách, so giờ/giá, xếp hạng) chạy bằng luật — nhanh, rẻ, **DETERMINISTIC** (đề bài yêu cầu chạy phát nào ra y phát nấy).
- Semantic cache cho câu đã hiểu, nhưng **có ngưỡng khoảng cách**: đủ gần mới tin, xa quá → quăng về LLM. Tránh trả "hàng gần nhất nhưng sai". **Cache phải nạp sẵn trước khi chấm**, không vừa chấm vừa điền.
- Có chế độ deterministic **không cần API key** (planner rule-based fallback) để test/demo.

## Stack & API

- Python · FastAPI · sentence-transformers · faiss-cpu · rank_bm25 · pyvi/underthesea (tokenize + chuẩn hóa dấu) · openpyxl.
- `GET /v1/search` (+ alias `/search`, `/v1/geocode-search`) trả **PlaceResult DTO** đúng `docs/tasco_api.pdf`; có `/health`.
- Mapping PlaceResult → SearchSuggestion (cho Flutter): `id→id`, `label|name→label`, `category|type→meta`, `address→description`, `coordinates.lat/lon→coordinates`.
- Config được `baseUrl` + auth (`Authorization: Bearer` hoặc `X-API-Key`), không hardcode credentials.

## Cấu trúc repo

```
data/                 # xlsx dataset (không sửa tay)
docs/                 # tasco_api.pdf — API contract
lexicon/              # attribute_concepts.yaml (~25-30 concept ← 82 token)
                      # gazetteer.yaml (landmark → toạ độ), categories.yaml
src/
  data_loader.py      # xlsx → POI documents (giữ nguyên dấu)
  search.py           # orchestrator: plan → retrieve → rerank
  understanding/      # query_plan.py, llm_planner.py, rules.py, semantic_cache.py
  retrieval/          # dense.py, bm25.py, filters.py, fusion.py (RRF)
  ranking/            # signals.py (7 signal), reranker.py, llm_rerank.py
  api/                # main.py (FastAPI), dto.py (PlaceResult ↔ SearchSuggestion)
eval/
  verify_dataset.py   # xác minh con số dataset (ĐANG CHẠY ĐƯỢC)
  run_eval.py         # ƯU TIÊN SỐ 1 — metric + ablation
tests/
```

## Eval harness — ƯU TIÊN SỐ 1

- `eval/run_eval.py`: load 60 câu → Hit@1 / MRR / Recall@3, chia theo difficulty & category.
- Bảng **ablation**: BM25 → +Dense → +Filter → +Rerank → +LLM. Chạy offline hoàn toàn.
- Sanity check tự động: **không dòng G nào lọt top-3** của bất kỳ câu nào.

## GOTCHAS / DO-NOT

- **ĐỪNG** tokenize cột `expected_semantic_requirements` theo dấu (`24/7` bị cắt sai). Phải parse ra slot.
- **ĐỪNG** nhét mọi thứ thành attribute: `nổi tiếng` → popularity signal, `giá rẻ` → price, `quán cà phê` → category, `gần X` → location.
- **ĐỪNG** xóa cứng dòng G (bộ private có thể khác) — **down-rank bền vững** bằng location-sanity (gần biển ở Hà Nội?) + rating + nhất quán attribute.
- **ĐỪNG** hardcode `expected_ids`, **ĐỪNG** overfit 60 câu (leave-one-out, ít weight). Extractor chỉ được nhả token có thật trong data, không chế từ mới.
- **Giữ nguyên dấu tiếng Việt** trong dữ liệu lưu/trả; chỉ bỏ dấu khi so khớp (normalize trong matcher, không normalize trong storage).

## Build order (2 ngày)

**Ngày 1**: data loader + POI document → chuẩn hóa 82 token → concept + gazetteer → embeddings + BM25 + RRF + hard filter → `search()` chạy end-to-end → eval harness (**lấy baseline**).

**Ngày 2**: LLM planner + lexicon whitelist + reason-rerank → **đo lại lift** → explainability + demo chống nhiễu G + OpenAPI + SDK Dart snippet + README có bảng ablation.
