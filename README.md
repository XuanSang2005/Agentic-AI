# Tasco Maps — AI Semantic Search & Ranking (AABW 2026, Track 2)

Search-and-ranking service cho POI tiếng Việt: rule-based query understanding →
hybrid retrieval (BM25 ∪ dense union-pool) → multi-signal re-rank (explainable).
**Deterministic, offline hoàn toàn, 0 LLM call** — chạy phát nào ra y phát nấy.

## Kết quả eval (Public_Evaluation, 60 câu)

| Pipeline | Hit@1 | MRR | Recall@3 |
|---|---|---|---|
| BM25 only | 0.833 | 0.890 | 0.861 |
| + Rerank (rules, no distance) | 0.967 | 0.983 | 0.911 |
| + Multi-signal Rerank | 0.967 | 0.983 | 0.936 |
| + Dense (signal) | 1.000 | 1.000 | 0.942 |
| + LLM Planner | — | — | — |
| Stress (unseen phrasing, n=20) | 0.950 | 0.967 | — |

- Breakdown theo difficulty/category + danh sách saved/regression: `make eval` (report JSON trong `eval/reports/`).
- **Stress set** (`make stress`): 20 câu phrasing chưa từng thấy (slang "sống ảo"/"chill", không dấu
  toàn bộ, English, landmark ngoài gazetteer) mượn gold từ public set — proxy trung thực cho private
  set vì public đã bão hòa. Câu fail duy nhất: mall-bait synthetic thắng khi query không dấu.
- Chống nhiễu synthetic: **G-above-gold = 0** (không dòng synthetic nào xếp trên đáp án đúng).

## Chạy nhanh

```bash
make install       # venv + dependencies (pin sentence-transformers/torch)
make verify-data   # xác minh mọi con số về dataset
make eval          # bảng ablation + report JSON
make stress        # 20 câu unseen phrasing (private-proxy)
make api           # FastAPI tại http://localhost:8000
make openapi       # export + VERIFY docs/openapi.json khớp contract
make test          # smoke + regression (kể cả bộ câu không dấu)
```

Lần chạy đầu tải model `intfloat/multilingual-e5-small` (~450MB, 1 lần); embedding corpus
cache tại `data/cache/*.npy` — các lần sau khởi động không cần network.

### Nguồn data: xlsx (mặc định) hoặc Postgres (opt-in)

POI đọc được từ 2 nguồn tương đương — chọn qua env `DATA_SOURCE`; **mặc định `xlsx`**,
không cần làm gì thêm. Postgres là opt-in cho lộ trình production:

```bash
docker compose up -d           # Postgres 16 (port host 5433), schema tự áp từ migrations/
make db-seed                   # xlsx → bảng pois (idempotent, chạy 1 lần)
DATA_SOURCE=postgres make eval # toàn pipeline đọc từ Postgres
```

`DATABASE_URL` không đặt thì default khớp docker-compose dev
(`postgresql://tasco:tasco@localhost:5433/tasco`); Postgres khác (brew/cloud) thì
export URL tương ứng. Tính tương đương được gate bằng
`tests/test_postgres_equivalence.py`: list POI 2 nguồn phải bằng nhau **từng field,
đúng thứ tự** (thứ tự corpus quyết định hash embedding cache — bảng `pois` có cột
`row_order` giữ vị trí dòng xlsx gốc, loader `ORDER BY row_order`).
`Public_Evaluation` luôn đọc từ xlsx (eval harness là tooling offline).

## Kiến trúc 3 lớp

```
query ──▶ L0 Preprocess (deterministic, idempotent)
          │  viết tắt → đầy đủ (bv/ks/q1…, whitelist + vocab guard)
          │  restore dấu (dictionary domain, phrase-first) · typo fix BẢO THỦ
          │  (edit-dist 1, unique-candidate, bigram tie-break — TẮT được:
          │   TASCO_TYPO_FIX=0, xem src/understanding/typo_fix.py)
          L1 Query Understanding (rules, deterministic)
          │  lexicon whitelist: 82 attr token → concept; category synonyms
          │  regex: city / district (→ centroid) / landmark (gazetteer, cần cue "gần")
          │  polarity ("không quá đông" → NEG đông khách) · popularity flag
          ▼  QueryPlan {categories, attr_concepts, neg, city, district, landmark, coord}
          L2 Retrieval — union pool (tối đa recall, không hard-filter)
          │  BM25 trên norm_document (BỎ DẤU 2 phía → câu không dấu vẫn match)
          │  Dense e5-small trên document GIỮ DẤU (prefix passage:/query:)
          ▼  pool = top-25 BM25 ∪ top-25 dense
          L3 Multi-signal Rerank (weighted sum, explainable)
          │  dense .32 · category .22 · attr .20 · distance .15 · city .10
          │  bm25 .06 · name(exact) .06 · rating .03 · pop .02
          ▼  top-k + signal breakdown (?explain=true)
```

## API (contract: `docs/tasco_api.pdf`)

- `GET /v1/search` (+ alias `/search`, `/v1/geocode-search`) — params `q` (bắt buộc),
  `lat`, `lon`, `radiusMeters`, `bbox`, `category`, `limit` (mặc định 10, trần 20), `lang`.
  Response `{query, results: [PlaceResult], meta: {limit, lang}}` — giữ nguyên dấu tiếng Việt.
- `?explain=true` (extension): mỗi result kèm `explanation` = QueryPlan đã parse + điểm
  từng signal (đúng con số dùng để xếp hạng — explain và ranking chung một đường code).
- `lat/lon` = focus point: dùng làm **fallback** location khi query text không tự nói
  ("atm" + đứng ở Hồ Gươm → ATM Hoàn Kiếm; "cafe gần sông hàn" thì text thắng).
- `GET /health` — status + chế độ deterministic.
- OpenAPI: `docs/openapi.json` (`make openapi` tự verify đủ path/param/field theo PDF).
- SDK adapter Flutter/Dart: `docs/tasco_search_adapter.dart` (PlaceResult → SearchSuggestion,
  constructor nhận `baseUrl` + `bearerToken`/`apiKey`/`headerProvider`).

```bash
curl "http://localhost:8000/health"
curl -G "http://localhost:8000/v1/search" \
  --data-urlencode "q=cafe có wifi gần hồ gươm" \
  --data-urlencode "lat=21.0287" --data-urlencode "lon=105.8524" \
  --data-urlencode "limit=3" --data-urlencode "explain=true"
```

**Auth (pluggable, không hardcode):** mặc định mock mode nhận mọi request (đúng PDF).
Đặt `TASCO_BEARER_TOKEN` và/hoặc `TASCO_API_KEY` khi deploy → service yêu cầu
`Authorization: Bearer …` hoặc `X-API-Key: …`, sai/thiếu trả `401` đúng shape `ErrorResponse`.

## Notes (theo Submission Expectations của PDF)

- **Ranking logic**: 9 signal weighted-sum trên candidate pool BM25 ∪ dense — dense relevance
  (chính), category/attr match từ QueryPlan (concept-level, lexicon 82 token thật),
  distance (landmark gazetteer → toạ độ chính xác; district → match trực tiếp, centroid làm
  gradient), city, exact name/brand, BM25, rating, popularity (chỉ khi query bật cờ
  "nổi tiếng/best"). Mọi kết quả trả được breakdown từng signal (`?explain=true`).
- **Latency** (MacBook CPU, đo trên 60 câu eval): p50 **6ms**, p95 **20ms**, max **30ms**/query;
  startup 1 lần ~7s (load model + warmup; embedding corpus đọc từ `.npy` cache).
  0 LLM call, 0 network call lúc serve.
- **Fallback behavior**: BM25 và dense bù blind-spot cho nhau trong union pool — dense mù
  câu KHÔNG DẤU (model train trên text có dấu) thì BM25 bỏ-dấu-2-phía gánh; BM25 dính
  syllable-bait ("địa" khớp "địa phương") thì dense gánh. Query không hiểu được → mọi
  signal về trung tính 0.5, không crash, vẫn trả kết quả theo relevance. Landmark ngoài
  gazetteer → rơi về city/district/BM25 (đã test: "gần bờ hồ" vẫn ra đúng).
- **Data provenance**: 100% từ dataset hackathon (`data/ai_maps_track2_dataset_participants.xlsx`,
  sheet POI_Dataset 111 dòng = 39 thật + 72 synthetic prefix G). `source: "mock"` trong mọi
  response. Không dùng data ngoài; gazetteer 7 landmark là toạ độ công khai tự cung cấp
  (đã verify chéo với POI trong data, sai số <1km — xem `lexicon/gazetteer.yaml`).

## Design choices

- **Soft-gating thay hard-filter**: category/city/distance là signal có trọng số, không phải
  filter loại bỏ — vì bẫy P009 ("cây xăng có toilet **trên đường đi Hạ Long**": Hạ Long là ngữ
  cảnh chuyến đi, hard-filter location sẽ loại mất đáp án ở Cầu Giấy). Landmark chỉ được
  resolve khi có cue "gần/near" ngay trước tên. Dòng ablation "+ Hard Filter" vì thế bỏ qua
  có chủ đích.
- **0 LLM call, deterministic là chủ đích**: đề bài yêu cầu reproducible — toàn bộ pipeline
  là rules + math thuần, seed-free, chạy offline. LLM planner để dành cho slang nặng (xem
  Future work) và sẽ chỉ đứng ở L1 với semantic cache + rule fallback.
- **G-defense không nhìn nhãn**: tuyệt đối không dùng `is_synthetic` làm signal (private set
  có thể khác). Down-rank synthetic đến từ tín hiệu generalizable: city/distance consistency
  (5 dòng G "gần biển" ở city không có biển tự chết vì location), rating, name-match strict.
  Đo bằng **G-above-gold = 0/60** (harmful); G-in-top3 thô còn 67 nhưng toàn nằm DƯỚI đáp án
  (cosmetic — không ảnh hưởng kết quả trả cho user ở top).

## Future work

- **LLM planner ở L1** (structured output theo lexicon whitelist + semantic cache có ngưỡng
  + rule fallback) cho slang/câu mơ hồ nặng — kiến trúc đã chừa sẵn chỗ (`src/understanding/`).
- **Landmark-span consumption**: fix leak "cafe gần hồ xuân hương" ăn attr `gan_ho` generic
  (đã đo: fix được leak nhưng cost 1 câu public vì dense-bait — bật lại khi reranker semantic
  mạnh hơn, xem comment trong `src/understanding/rules.py`).
- **e5-base / bge-m3** làm ablation thêm — corpus 111 doc, phải ĐO chứ không mặc định model
  to hơn tốt hơn (e5-small đã đủ 1.000 public).
- Hard-filter opt-in cho use-case cần precision tuyệt đối (kèm cờ tắt để né P009-type).

## Cấu trúc repo

```
data/                 # dataset hackathon (+ data/cache/ embedding .npy, gitignore)
docs/                 # tasco_api.pdf (contract) · openapi.json · tasco_search_adapter.dart
lexicon/              # attribute_concepts.yaml · categories.yaml · gazetteer.yaml
src/
  data_loader.py      # nguồn sự thật: xlsx → POI/EvalQuery + normalize_vi
  search.py           # SearchService (orchestrator) + Protocol Retriever
  understanding/      # rules.py (QueryPlan extractor) · query_plan.py
  retrieval/          # bm25.py · dense.py
  ranking/            # signals.py (9 signal) · reranker.py (weighted rerank)
  api/                # main.py (FastAPI) · dto.py (PlaceResult) · export_openapi.py
eval/
  run_eval.py         # ablation + breakdown + G-diagnostic + saved/regression
  stress_queries.py   # 20 câu unseen phrasing (private-proxy)
  verify_dataset.py   # xác minh dataset claims
tests/                # smoke + no-accent regression
```
