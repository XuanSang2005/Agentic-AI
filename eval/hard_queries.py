"""Bộ 77 câu KHÓ dày đặc — 10 nhóm thách thức, đo weakness thật của pipeline.

Khác stress_queries (20 câu mượn gold public): bộ này gold GHI TRỰC TIẾP theo
field thật của 39 POI (harness được đọc gold — pipeline thì KHÔNG bao giờ).
Đây là instrument ĐO, không phải gate CI: một số câu cực khó fail là kỳ vọng —
giá trị nằm ở việc lộ nhóm yếu, đừng tune weight để "xanh" bộ này (overfit).

Nhóm: typo đơn/đôi · không dấu · viết tắt+district · mixed/English · phủ định
· bẫy location-context (P009-style) · G-bait robustness · đa ràng buộc
· time/price · intent đời thường.

Sanity kèm theo: đếm dòng G lọt top-1/top-3 trên TOÀN BỘ 77 câu.
Chạy: make hardq
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime

from src import config
from src.data_loader import load_pois
from src.ranking.reranker import RerankRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

# (nhóm, query, gold set — hit khi top-1 thuộc set, ghi chú độ khó)
CASES: list[tuple[str, str, set[str], str]] = [
    # ============ A. TYPO (đơn/đôi/transpose) ============
    ("typo", "nha hag ngon o ha noi",              {"R001", "R003", "R004"}, "hàg→hàng, không dấu"),
    ("typo", "khach sann gan bien da nang",        {"H001", "H002"}, "sạnn→sạn"),
    ("typo", "quan cafee yen tinh lam viec",       {"C001", "C004", "C005", "C007"}, "cafee→cafe"),
    ("typo", "nhaf hangf cho gia dinh",            {"R001", "R005", "R007", "R010"}, "double-typo Telex f"),
    ("typo", "benh vein cap cuu ha noi",           {"S005"}, "vein: transpose 2-edit (ngoài luật sửa)"),
    ("typo", "pho thin lo duc hà noi",             {"R004"}, "trộn có dấu + không dấu"),
    ("typo", "khach san co ho boii da nang",       {"H001", "H002"}, "boii→bơi"),
    ("typo", "tram xagn gan cau giay",             {"S003"}, "xagn: transpose"),
    ("typo", "rap phimm o da nang",                {"M004"}, "phimm→phim"),
    ("typo", "ca phe voi vieww dep da lat",        {"C006"}, "vieww→view"),
    ("typo", "hangf quan an khuya sai gon",        {"R006"}, "hangf→hàng"),
    ("typo", "quan cahy quan 3",                   {"R007"}, "cahy: transpose 'chay'"),

    # ============ B. KHÔNG DẤU hoàn toàn ============
    ("no_accent", "quan an mo khuya gia re sai gon",        {"R006"}, ""),
    ("no_accent", "khach san co phong hop cho cong tac ha noi", {"H003"}, ""),
    ("no_accent", "benh vien quoc te tphcm",                {"S006"}, ""),
    ("no_accent", "nha thuoc mo muon quan 1",               {"S007"}, ""),
    ("no_accent", "cong vien di dao ha noi",                {"A002"}, ""),
    ("no_accent", "cho sac xe dien da nang",                {"S004"}, ""),
    ("no_accent", "quan bun cha noi tieng ha noi",          {"R003"}, ""),
    ("no_accent", "khu mua sam an uong quan 1",             {"M001"}, ""),
    ("no_accent", "dai quan sat ngam thanh pho ha noi",     {"A005"}, "'đài quan sát' hiếm"),
    ("no_accent", "quan lau am cung cho nhom o da lat",     {"R009"}, ""),

    # ============ C. VIẾT TẮT + DISTRICT ============
    ("abbrev", "bv gan day",           {"S005", "S006"}, "bv→bệnh viện"),
    ("abbrev", "nh mon y q1",          {"R002"}, "nh→nhà hàng, món Ý"),
    ("abbrev", "ks 24/7 q1",           {"H005"}, "ks + token 24/7"),
    ("abbrev", "cf co o cam q1",       {"C001"}, "cf→cà phê, ổ cắm"),
    ("abbrev", "cv cho tre em hn",     {"A002"}, "cv→công viên, hn→Hà Nội"),
    ("abbrev", "nt gan q1",            {"S007"}, "nt→nhà thuốc"),
    ("abbrev", "tttm dong khoi",       {"M001"}, "tttm→trung tâm thương mại"),
    ("abbrev", "rp nguyen du",         {"M003"}, "rp→rạp phim"),

    # ============ D. MIXED / ENGLISH ============
    ("english", "romantic dinner rooftop saigon",           {"R008", "R002"}, ""),
    ("english", "cheap homestay with garden dalat",         {"H004"}, ""),
    ("english", "hospital emergency 24/7 hanoi",            {"S005"}, ""),
    ("english", "ev charging station da nang",              {"S004"}, ""),
    ("english", "vegetarian restaurant hcmc",               {"R007"}, "hcmc: alias city EN"),
    ("english", "cinema in danang",                         {"M004"}, ""),
    ("english", "coffee with strong wifi for remote work",  {"C007", "C001", "C004", "C005"}, ""),
    ("english", "shopping mall with observation deck hanoi", {"A005"}, ""),

    # ============ E. PHỦ ĐỊNH / POLARITY ============
    ("negation", "cafe khong on ao de doc sach",            {"C004"}, "phủ định + 'sách'"),
    ("negation", "quan an khong qua dong cho gia dinh",     {"R001", "R005", "R007", "R010"}, "loại quán đông khách"),
    ("negation", "cho lam viec vang ve it nguoi",           {"C001", "C004", "C005", "C007"}, "'vắng vẻ ít người'"),
    ("negation", "quan cafe it dong de hoc bai",            {"C004", "C001", "C007"}, "'học bài' paraphrase"),
    ("negation", "nha hang khong on cho cap doi",           {"R002", "R008"}, "phủ định + hẹn hò"),

    # ============ F. BẪY LOCATION-CONTEXT (P009-style) ============
    ("loc_trap", "cay xang co toilet tren duong di ha long", {"S003"}, "P009 gốc — Hạ Long là ngữ cảnh"),
    ("loc_trap", "do xang truoc khi di sapa",                {"S003"}, "Sapa là đích chuyến đi"),
    ("loc_trap", "an pho truoc khi ra san bay noi bai",      {"R004"}, "Nội Bài là ngữ cảnh"),
    ("loc_trap", "mua thuoc truoc khi ve que o quan 1",      {"S007"}, "'về quê' không phải location"),
    ("loc_trap", "cafe hop nhom truoc khi len da lat",       {"C007", "C008"}, "CỰC KHÓ: 'lên đà lạt' sẽ dụ city=Đà Lạt"),
    ("loc_trap", "khach san gan van phong cho sep tu sai gon ra ha noi", {"H003"}, "2 city trong câu — đích là HN"),

    # ============ G. G-BAIT ROBUSTNESS (khớp bề mặt dòng G) ============
    ("g_bait", "khach san gia duoi 2 trieu",       {"H004"}, "'giá dưới 2 triệu' là token CHỈ CÓ ở dòng G"),
    ("g_bait", "quan an ngon re gan trung tam",    {"R006"}, "generic — G có cả rổ 'gần trung tâm'"),
    ("g_bait", "cafe dep gan ho o ha noi",         {"C003"}, "G007-9 gắn 'gần biển' ở HN"),
    ("g_bait", "atm gan pho di bo",                {"S001"}, ""),
    ("g_bait", "khach san co spa gan bien",        {"H002"}, ""),
    ("g_bait", "diem check in mien phi o da lat",  {"A004"}, ""),

    # ============ H. ĐA RÀNG BUỘC (4-5 constraint/câu) ============
    ("multi", "cafe yen tinh co wifi va o cam de lam viec quan 1",      {"C001"}, "5 ràng buộc"),
    ("multi", "nha hang y lang man de dat ban q1",                      {"R002"}, ""),
    ("multi", "khach san gan bien co ho boi cho gia dinh da nang",      {"H001"}, "H002 thiếu 'gia đình'"),
    ("multi", "quan an re mo khuya cho nhom o quan 1",                  {"R006"}, ""),
    ("multi", "resort co spa cho cap doi di tuan trang mat",            {"H002"}, "'tuần trăng mật' paraphrase"),
    ("multi", "cafe san vuon gia re cho nhom ban o thu duc",            {"C008"}, ""),
    ("multi", "trung tam thuong mai co rap phim va bai do xe ha noi",   {"M002"}, ""),
    ("multi", "benh vien quoc te cap cuu dich vu tot sai gon",          {"S006"}, ""),

    # ============ I. TIME / PRICE ============
    ("time_price", "atm hoat dong 24/7 gan ho guom",   {"S002"}, ""),
    ("time_price", "hieu thuoc mo muon sau 22h",       {"S007"}, "'hiệu thuốc' + mốc giờ"),
    ("time_price", "quan an sau nua dem sai gon",      {"R006"}, "18:00-03:00 qua đêm"),
    ("time_price", "cho choi mien phi o da lat",       {"A004"}, ""),
    ("time_price", "an toi muon 11h dem quan 1",       {"R006", "R008"}, "R008 mở tới 01:00"),
    ("time_price", "khach san mo cua 24 gio",          {"H001", "H002", "H003", "H005"}, ""),

    # ============ J. INTENT ĐỜI THƯỜNG (colloquial) ============
    ("intent", "dua ban gai di an ky niem 1 nam",      {"R002", "R008"}, "không từ khoá attr nào khớp thô"),
    ("intent", "cho song ao dep o da lat",             {"C006", "A004"}, "slang 'sống ảo'"),
    ("intent", "doi bung luc 2 gio sang",              {"R006"}, "suy luận giờ mở qua đêm"),
    ("intent", "cho ca nha di choi cuoi tuan ha noi",  {"A002", "R010"}, ""),
    ("intent", "meeting voi doi tac o dau ha noi",     {"H003", "C007"}, "EN 'meeting' + intent"),
    ("intent", "xem phim voi hoi ban o sai gon",       {"M003"}, ""),
    ("intent", "nguoi nha om giua dem o ha noi",       {"S005"}, "'ốm giữa đêm' → cấp cứu 24/7"),
    ("intent", "mua thuoc cam luc toi muon quan 1",    {"S007"}, ""),
]


def main() -> None:
    pois = load_pois()
    g_ids = {p.id for p in pois if p.is_synthetic}
    retriever = RerankRetriever(pois, base=BM25Retriever(pois), dense=DenseRetriever(pois))

    rows = []
    for group, query, gold, note in CASES:
        ranked = retriever.search(query, k=10)
        rr = next((1.0 / r for r, pid in enumerate(ranked, 1) if pid in gold), 0.0)
        rows.append({
            "group": group, "query": query, "gold": sorted(gold), "note": note,
            "top3": ranked[:3], "hit1": bool(ranked) and ranked[0] in gold, "rr": rr,
            "g_top1": bool(ranked) and ranked[0] in g_ids,
            "g_top3": [pid for pid in ranked[:3] if pid in g_ids],
        })

    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    width = 80
    print("=" * width)
    print(f" HARD QUERIES — {len(rows)} câu khó / {len(by_group)} nhóm, full pipeline")
    print("=" * width)
    print(f" {'nhóm':<12}{'n':>4}{'Hit@1':>9}{'MRR':>9}   câu fail")
    print("-" * width)
    for group, grs in by_group.items():
        h = sum(r["hit1"] for r in grs) / len(grs)
        m = sum(r["rr"] for r in grs) / len(grs)
        fails = [r for r in grs if not r["hit1"]]
        print(f" {group:<12}{len(grs):>4}{h:>9.3f}{m:>9.3f}   {len(fails)} fail")
        for r in fails:
            print(f"     ✗ {r['query']!r}")
            print(f"       gold={r['gold']} top3={r['top3']}"
                  + (f"  ({r['note']})" if r["note"] else ""))
    print("-" * width)
    hit1 = sum(r["hit1"] for r in rows) / len(rows)
    mrr = sum(r["rr"] for r in rows) / len(rows)
    n_g1 = sum(r["g_top1"] for r in rows)
    n_g3 = sum(1 for r in rows if r["g_top3"])
    print(f" OVERALL   Hit@1 = {hit1:.3f}   MRR = {mrr:.3f}")
    print(f" G-sanity: G chiếm top-1 ở {n_g1}/{len(rows)} câu"
          f" | G lọt top-3 ở {n_g3}/{len(rows)} câu")
    print("=" * width)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_hard_queries.json"
    out.write_text(json.dumps({"hit@1": hit1, "mrr": mrr,
                               "g_top1": n_g1, "g_top3": n_g3, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" Report: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
