"""Private-proxy instrument: 20 câu phrasing LẠ (chưa từng thấy) map về gold có sẵn.

Public set đã bão hòa 1.000 → con số này mới là proxy trung thực cho private set.
Mỗi câu stress lấy gold từ 1 câu Public_Evaluation cùng intent (gold_from) —
KHÔNG hardcode poi_id trong file này (đọc expected_ids qua data_loader lúc chạy).

Độ lạ cố tình đa dạng: slang ("sống ảo", "chill", "thanh tịnh", "im ắng"),
KHÔNG DẤU toàn bộ, English/mixed, landmark ngoài gazetteer ("bờ hồ"),
paraphrase category ("ô tô điện", "nhà vệ sinh", "rạp chiếu phim").

Chạy FULL PIPELINE (BM25 ∪ dense + rerank) — đúng thứ sẽ chấm private.
Chạy: make stress
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from src import config
from src.data_loader import load_eval, load_pois
from src.ranking.reranker import RerankRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

# (câu stress "lạ", query_id public cho gold, ghi chú độ lạ)
STRESS_QUERIES: list[tuple[str, str, str]] = [
    ("quan cafe song ao dep o da lat",          "P023", "no-accent + slang 'sống ảo'"),
    ("cho nao chill yen tinh de lam viec",      "P001", "slang 'chill'"),
    ("quan an khuya o sai gon",                 "P028", "no-accent, bỏ 'giá rẻ'"),
    ("tim cho rut tien gan bo ho",              "P046", "landmark NGOÀI gazetteer ('bờ hồ')"),
    ("hotel gan bien co ho boi cho gia dinh",   "P047", "no-accent + EN mix"),
    ("coffee yen tinh lam viec o ha noi",       "P001", "no-accent + EN mix"),
    ("di choi cuoi tuan voi con nho o sai gon", "P022", "paraphrase 'con nhỏ'"),
    ("quan chay thanh tinh quan 3",             "P013", "slang 'thanh tịnh'"),
    ("mall co rap chieu phim o ha noi",         "P017", "paraphrase 'rạp chiếu phim'"),
    ("cho sac o to dien tai da nang",           "P010", "paraphrase 'ô tô điện'"),
    ("restaurant for date night in saigon",     "P003", "English hoàn toàn"),
    ("cafe doc sach im ang",                    "P058", "no-accent + slang 'im ắng'"),
    ("quan pho ngon o ha noi",                  "P018", "no-accent"),
    ("cay xang co nha ve sinh gan cau giay",    "P009", "paraphrase 'nhà vệ sinh'"),
    ("khach san di cong tac co phong hop",      "P015", "no-accent, bỏ landmark"),
    ("diem checkin dep chup hinh o da nang",    "P052", "no-accent"),
    ("lau am cung an nhom troi lanh da lat",    "P026", "no-accent, dồn intent"),
    ("mua sam an uong gan cho ben thanh",       "P021", "landmark thay cho district"),
    ("quiet cafe for studying in hanoi",        "P011", "English hoàn toàn"),
    ("bun cha noi tieng cho tourist",           "P019", "no-accent + EN mix"),
]


def main() -> None:
    pois = load_pois()
    gold_by_qid = {q.id: q.expected_ids for q in load_eval()}
    retriever = RerankRetriever(pois, base=BM25Retriever(pois), dense=DenseRetriever(pois))

    rows = []
    for query, gold_from, note in STRESS_QUERIES:
        expected = set(gold_by_qid[gold_from])
        ranked = retriever.search(query, k=10)
        rr = next((1.0 / rank for rank, pid in enumerate(ranked, 1) if pid in expected), 0.0)
        rows.append({"query": query, "gold_from": gold_from, "note": note,
                     "expected": sorted(expected), "top3": ranked[:3],
                     "hit1": ranked[0] in expected, "rr": rr})

    n = len(rows)
    hit1 = sum(r["hit1"] for r in rows) / n
    mrr = sum(r["rr"] for r in rows) / n

    width = 78
    print("=" * width)
    print(f" STRESS — unseen phrasing, full pipeline ({n} câu, gold mượn từ public set)")
    print("=" * width)
    for r in rows:
        mark = "✓" if r["hit1"] else "✗"
        print(f" {mark} [{r['gold_from']}] {r['query']!r}  ({r['note']})")
        if not r["hit1"]:
            print(f"      expected={r['expected']} | top3={r['top3']}")
    print("-" * width)
    print(f" Hit@1 = {hit1:.3f}   MRR = {mrr:.3f}   (public set: 1.000 — số này mới là proxy private)")
    print("=" * width)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_stress.json"
    out.write_text(json.dumps({"hit@1": hit1, "mrr": mrr, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" Report: {out.relative_to(config.ROOT)}")

    # Cập nhật dòng stress trong README (thêm mới nếu chưa có)
    readme = config.README_MD.read_text(encoding="utf-8")
    row = f"| Stress (unseen phrasing, n={n}) | {hit1:.3f} | {mrr:.3f} | — |"
    pattern = re.compile(r"^\| Stress \(unseen phrasing.*$", re.MULTILINE)
    if pattern.search(readme):
        readme = pattern.sub(row, readme)
    else:
        anchor = re.compile(r"^(\| \+ LLM Planner \|.*)$", re.MULTILINE)
        readme = anchor.sub(rf"\1\n{row}", readme)
    config.README_MD.write_text(readme, encoding="utf-8")
    print(" README: đã cập nhật dòng 'Stress (unseen phrasing)'")


if __name__ == "__main__":
    main()
