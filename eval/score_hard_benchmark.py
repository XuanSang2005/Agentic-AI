"""
Hard benchmark scorer — private-proxy instrument cho tasco-semantic-search.

133 câu KHÓ, giữ-riêng, gold suy từ field thật của 39 POI (không phải public 60 đã bão hòa).
Đo: Hit@1 / MRR / Recall@k (có Wilson 95% CI), phân tầng độ khó & biến thể,
robustness trên câu có hard-negative, VÀ satisfaction accuracy cho câu đa-ràng-buộc.

CÁCH DÙNG: điền 2 hàm ADAPTER ở dưới để nối vào pipeline thật, rồi:
    python eval/score_hard_benchmark.py
"""
import json, math, os
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
BENCH=json.load(open(os.path.join(HERE,"hard_benchmark.json"),encoding="utf-8"))

# ============================================================
# ADAPTER — nối vào pipeline thật (measure-only, KHÔNG đụng ranking/reasoning)
# ============================================================
import sys

sys.path.insert(0, os.path.dirname(HERE))  # repo root để import src.*

_STATE: dict = {}


def _pipeline():
    """Build full pipeline THẬT đúng 1 lần (BM25 ∪ dense + rerank) + map tra cứu."""
    if _STATE:
        return _STATE
    from src.data_loader import (extract_unique_attributes, extract_unique_categories,
                                 load_pois, normalize_vi)
    from src.ranking.reranker import RerankRetriever
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import (AttributeIndex, ColumnAnchorIndex, DenseRetriever,
                                     JointMetadataIndex)

    pois = load_pois()
    _STATE["by_id"] = {p.id: p for p in pois}
    unique_attrs = extract_unique_attributes(pois)
    _STATE["retriever"] = RerankRetriever(
        pois, base=BM25Retriever(pois), dense=DenseRetriever(pois),
        attr_index=AttributeIndex(unique_attrs),
        joint_index=JointMetadataIndex(extract_unique_categories(pois), unique_attrs),
        column_anchor=ColumnAnchorIndex())
    # v2 dynamic-vector-attributes: lớp reasoning chấm bằng RAW attribute string —
    # tập token thật (normalized) chỉ dùng để chọn value/label nào là token thật
    _STATE["attr_norms"] = {normalize_vi(a) for a in unique_attrs}
    _STATE["normalize"] = normalize_vi
    return _STATE


def search(query: str, limit: int = 10):
    """List poi_id đã rank từ pipeline thật. Benchmark gold dùng id THÔ ("C003") —
    retriever nội bộ trả id thô sẵn; vẫn strip "poi:" phòng hờ (DTO layer thêm prefix)."""
    st = _pipeline()
    return [pid.removeprefix("poi:") for pid in st["retriever"].search(query, k=limit)]


def predict_satisfaction(query: str, poi_id: str, constraints: list):
    """Chấm satisfaction bằng ĐÚNG lớp constraint reasoning đang serve ?explain
    (src.reasoning.constraints.score_constraint) — không tự tính lại logic.

    Map type/value của benchmark → Constraint của hệ:
      category → category; city/district → location; attribute/price → RAW
      attribute string (v2 dynamic-vector-attributes: lớp reasoning chấm bằng
      normalized string match + keyword classify, không còn concept id).
      price của benchmark dùng machine-key EN ("cheap") + label VN ("giá rẻ")
      → chọn value nếu là token thật trong data, trượt thì lấy label.
    """
    from src.reasoning.constraints import (Constraint, _classify_attr,
                                           score_constraint)
    st = _pipeline()
    poi = st["by_id"].get(poi_id.removeprefix("poi:"))
    if poi is None:
        return None
    out = {}
    for c in constraints:
        ctype, value = c["type"], c["value"]
        if ctype == "category":
            con = Constraint("category", value, value, priority=2)
        elif ctype == "city":
            con = Constraint("location", value, value, priority=2,
                             data={"city": value})
        elif ctype == "district":
            con = Constraint("location", value, value, priority=2,
                             data={"district": value, "city": None})
        else:  # attribute | price — chọn dạng là token thật trong data
            token = value
            if (st["normalize"](value) not in st["attr_norms"]
                    and st["normalize"](c.get("label", "")) in st["attr_norms"]):
                token = c["label"]
            kind = _classify_attr(token)
            con = Constraint(kind, token, token, priority=1)
        out[c["id"]] = float(score_constraint(poi, con))
    return out
# ============================================================

def wilson(k, n, z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h), min(1,c+h))

def score():
    scored=[q for q in BENCH["queries"] if q["gold"]]
    hit=mrr=r3=r5=0
    by_d=defaultdict(lambda:[0,0]); by_v=defaultdict(lambda:[0,0])
    hn=[0,0]  # hard-negative subset
    for q in scored:
        order=search(q["query"], 10) or []
        g=set(q["gold"])
        ok = bool(order) and order[0] in g
        hit+=ok; by_d[q["difficulty"]][0]+=ok; by_d[q["difficulty"]][1]+=1
        by_v[q["variant"]][0]+=ok; by_v[q["variant"]][1]+=1
        if q.get("has_hard_negative"): hn[0]+=ok; hn[1]+=1
        r3+= any(p in g for p in order[:3]); r5+= any(p in g for p in order[:5])
        for i,p in enumerate(order):
            if p in g: mrr+=1/(i+1); break
    n=len(scored); lo,hiCI=wilson(hit,n)
    print(f"\n{'='*56}\nRETRIEVAL (n={n} câu có gold)\n{'='*56}")
    print(f"  Hit@1 = {hit/n:.3f}   [95% CI {lo:.3f}–{hiCI:.3f}]")
    print(f"  MRR   = {mrr/n:.3f}    Recall@3 = {r3/n:.3f}    Recall@5 = {r5/n:.3f}")
    print("  Hit@1 theo độ khó :", {k:round(v[0]/v[1],3) for k,v in sorted(by_d.items())})
    print("  Hit@1 theo biến thể:", {k:round(v[0]/v[1],3) for k,v in sorted(by_v.items())})
    if hn[1]: print(f"  Robustness (câu có G khớp bề mặt, n={hn[1]}): Hit@1 = {hn[0]/hn[1]:.3f}")

    # ---- satisfaction accuracy (câu đa-ràng-buộc) ----
    multi=[q for q in scored if "constraints" in q]
    probe=predict_satisfaction(multi[0]["query"], multi[0]["gold"][0], multi[0]["constraints"]) if multi else None
    if probe is None:
        print(f"\nSATISFACTION: adapter chưa nối (predict_satisfaction trả None) — bỏ qua.")
        return
    exact=perc=tot=0; per_type=defaultdict(lambda:[0,0])
    for q in multi:
        for pid in q["gold"]:
            truth=q["gold_satisfaction"].get(pid,{})
            pred=predict_satisfaction(q["query"], pid, q["constraints"]) or {}
            allmatch=True
            for c in q["constraints"]:
                t=truth.get(c["id"]); p=pred.get(c["id"])
                if t is None: continue
                ok=(p is not None and abs(p-t)<1e-6)
                per_type[c["type"]][0]+=ok; per_type[c["type"]][1]+=1
                tot+=1; perc+=ok; allmatch&=ok
            exact+=allmatch
    npairs=sum(len(q["gold"]) for q in multi)
    print(f"\n{'='*56}\nSATISFACTION ACCURACY (câu đa-ràng-buộc, {npairs} cặp query×gold)\n{'='*56}")
    print(f"  Per-constraint đúng : {perc}/{tot} = {perc/max(tot,1):.3f}")
    print(f"  Hồ sơ thỏa khớp HOÀN TOÀN (exact profile): {exact}/{npairs} = {exact/max(npairs,1):.3f}")
    print("  Đúng theo loại ràng buộc:", {k:round(v[0]/v[1],3) for k,v in sorted(per_type.items())})

if __name__=="__main__":
    score()
