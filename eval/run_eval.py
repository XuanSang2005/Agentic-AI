"""ƯU TIÊN SỐ 1 — eval harness offline.

- Load 60 câu Public_Evaluation → chạy retriever → Hit@1 / MRR / Recall@3 (+Recall@5).
- Báo cáo chia theo difficulty (Easy/Medium/Hard) & query_category.
- Nhận bất kỳ object nào implement Protocol `Retriever` (src.search) — nhờ đó
  bảng ablation (BM25 → +Dense → +Filter → +Rerank → +LLM) chỉ việc swap retriever.
- Sanity check (in cảnh báo, KHÔNG crash): đếm dòng G lọt top-3. Baseline BM25
  chắc chắn dính nhiều — ghi rõ con số để các lớp sau chứng minh kéo xuống 0.
- Ghi report JSON vào eval/reports/ + cập nhật dòng tương ứng trong bảng ablation README.
- ĐỪNG tokenize expected_semantic_requirements theo dấu (24/7 bị cắt) — giữ raw string.
- ĐỪNG hardcode expected_ids vào pipeline — chỉ harness này được đọc chúng.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime

from src import config
from src.data_loader import EvalQuery, load_eval, load_pois
from src.retrieval.bm25 import BM25Retriever
from src.search import Retriever


def _evaluate_query(retriever: Retriever, q: EvalQuery, synthetic_ids: set[str], k: int) -> dict:
    """Chạy 1 câu → các metric thành phần + chẩn đoán G-in-top3."""
    ranked = retriever.search(q.query, k=k)
    expected = set(q.expected_ids)

    rr = 0.0  # reciprocal rank của id đúng ĐẦU TIÊN; không thấy trong top-k → 0
    for rank, pid in enumerate(ranked, start=1):
        if pid in expected:
            rr = 1.0 / rank
            break

    return {
        "query_id": q.id,
        "query": q.query,
        "difficulty": q.difficulty,
        "category": q.category,
        "expected": q.expected_ids,
        "top5": ranked[:5],
        "hit1": bool(ranked) and ranked[0] in expected,
        "rr": rr,
        "recall3": len(set(ranked[:3]) & expected) / len(expected),
        "recall5": len(set(ranked[:5]) & expected) / len(expected),
        "g_in_top3": [pid for pid in ranked[:3] if pid in synthetic_ids],
    }


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "hit@1": sum(r["hit1"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "recall@3": sum(r["recall3"] for r in rows) / n,
        "recall@5": sum(r["recall5"] for r in rows) / n,
    }


def _fmt_row(label: str, agg: dict) -> str:
    return (f" {label:<28}{agg['n']:>4}"
            f"{agg['hit@1']:>9.3f}{agg['mrr']:>9.3f}"
            f"{agg['recall@3']:>9.3f}{agg['recall@5']:>9.3f}")


def _update_readme_ablation(row_label: str, agg: dict) -> bool:
    """Thay số thật vào dòng `| <row_label> | ... |` của bảng ablation trong README."""
    text = config.README_MD.read_text(encoding="utf-8")
    new_row = (f"| {row_label} | {agg['hit@1']:.3f} | {agg['mrr']:.3f} "
               f"| {agg['recall@3']:.3f} |")
    pattern = re.compile(rf"^\| {re.escape(row_label)} \|.*$", re.MULTILINE)
    new_text, n_sub = pattern.subn(new_row, text)
    if n_sub:
        config.README_MD.write_text(new_text, encoding="utf-8")
    return bool(n_sub)


def run_eval(
    retriever: Retriever,
    name: str,
    queries: list[EvalQuery] | None = None,
    synthetic_ids: set[str] | None = None,
    k: int = config.EVAL_TOP_K,
    readme_row: str | None = None,
    compare_to: dict | None = None,
) -> dict:
    """Chạy retriever trên toàn bộ eval set, in bảng, ghi report JSON, trả report dict.

    compare_to: report của lần chạy khác (thường baseline) → in thêm danh sách
    CỨU được (fail→pass) và REGRESSION (pass→fail) theo Hit@1.
    """
    if queries is None:
        queries = load_eval()
    if synthetic_ids is None:
        synthetic_ids = {p.id for p in load_pois() if p.is_synthetic}

    rows = [_evaluate_query(retriever, q, synthetic_ids, k) for q in queries]

    overall = _aggregate(rows)
    by_difficulty = {
        d: _aggregate([r for r in rows if r["difficulty"] == d])
        for d in ("Easy", "Medium", "Hard")
        if any(r["difficulty"] == d for r in rows)
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    by_category = {
        c: _aggregate(cat_rows)
        for c, cat_rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }

    # --- Sanity check G-in-top3 (cảnh báo, không crash) ---
    g_detail = {r["query_id"]: r["g_in_top3"] for r in rows if r["g_in_top3"]}
    g_total = sum(len(v) for v in g_detail.values())

    # --- In bảng ---
    width = 73
    print("=" * width)
    print(f" EVAL: {name}   (k={k}, {len(rows)} queries)")
    print("=" * width)
    print(f" {'segment':<28}{'n':>4}{'Hit@1':>9}{'MRR':>9}{'R@3':>9}{'R@5':>9}")
    print("-" * width)
    print(_fmt_row("OVERALL", overall))
    print(" --- difficulty ---")
    for d, agg in by_difficulty.items():
        print(_fmt_row(d, agg))
    print(" --- query_category ---")
    for c, agg in by_category.items():
        print(_fmt_row(c, agg))
    print("-" * width)
    n_fail = sum(1 for r in rows if not r["hit1"])
    print(f" Hit@1 fail: {n_fail}/{len(rows)} câu (danh sách trong report JSON)")
    # G-in-top3 là DIAGNOSTIC (robustness với distractor), không phải mục tiêu tối ưu trực tiếp
    if g_total:
        print(f" ⚠ G-in-top3 (diagnostic): {g_total} dòng G trong top-3 trên"
              f" {len(g_detail)}/{len(rows)} câu")
    else:
        print(" ✓ G-in-top3 (diagnostic): 0 — không dòng G nào lọt top-3")

    # --- So sánh với baseline (Hit@1 fail→pass / pass→fail) ---
    vs_baseline = None
    if compare_to is not None:
        base_pq = compare_to["per_query"]
        saved = [r for r in rows if r["hit1"] and not base_pq[r["query_id"]]["hit1"]]
        regressed = [r for r in rows if not r["hit1"] and base_pq[r["query_id"]]["hit1"]]
        vs_baseline = {
            "baseline_name": compare_to["name"],
            "saved": [r["query_id"] for r in saved],
            "regressed": [r["query_id"] for r in regressed],
        }
        print(f" vs '{compare_to['name']}':")
        print(f"   CỨU được (fail→pass): {len(saved)} câu")
        for r in saved:
            print(f"     + {r['query_id']} {r['query']!r} → top1={r['top5'][0]}")
        print(f"   REGRESSION (pass→fail): {len(regressed)} câu")
        for r in regressed:
            print(f"     - {r['query_id']} {r['query']!r} expected={r['expected']}"
                  f" top3={r['top5'][:3]}")

    # --- Report JSON ---
    report = {
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "g_in_top3": {"total": g_total, "queries_affected": len(g_detail), "detail": g_detail},
        "per_query": {r["query_id"]: {"hit1": r["hit1"], "rr": r["rr"],
                                      "recall3": r["recall3"], "top5": r["top5"]}
                      for r in rows},
        "vs_baseline": vs_baseline,
        "hit1_failures": [
            {key: r[key] for key in
             ("query_id", "query", "difficulty", "category", "expected", "top5")}
            for r in rows if not r["hit1"]
        ],
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    report_path = config.REPORTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" Report: {report_path.relative_to(config.ROOT)}")

    if readme_row:
        if _update_readme_ablation(readme_row, overall):
            print(f" README ablation: đã cập nhật dòng '{readme_row}'")
        else:
            print(f" ⚠ README ablation: KHÔNG tìm thấy dòng '{readme_row}' để cập nhật")
    print("=" * width)
    return report


def main() -> None:
    from src.ranking.reranker import RerankRetriever

    pois = load_pois()
    queries = load_eval()
    synthetic_ids = {p.id for p in pois if p.is_synthetic}
    print(f"POI: {len(pois)} ({len(pois) - len(synthetic_ids)} thật + "
          f"{len(synthetic_ids)} synthetic G) | Eval: {len(queries)} câu\n")

    # Dòng 1 bảng ablation — BM25 thuần trên câu thô: sàn lexical.
    bm25 = BM25Retriever(pois)
    base_report = run_eval(bm25, "BM25 baseline", queries, synthetic_ids,
                           readme_row="BM25 only")
    print()

    # L3 rerank (rules-based plan): category + attr concept + city + rating (+pop flag).
    rerank = RerankRetriever(pois, base=bm25)
    run_eval(rerank, "BM25 + Rerank", queries, synthetic_ids,
             readme_row="+ Multi-signal Rerank", compare_to=base_report)


if __name__ == "__main__":
    main()
