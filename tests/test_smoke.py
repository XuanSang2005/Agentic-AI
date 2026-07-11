"""Smoke test: data loader đọc đúng dataset, BM25 baseline trả kết quả deterministic.

Chạy được ở deterministic mode, không cần API key.
(/health của FastAPI sẽ thêm ở slice API — chưa có trong slice baseline.)
"""
from src.data_loader import load_eval, load_pois, normalize_vi
from src.retrieval.bm25 import BM25Retriever
from src.search import Retriever


def test_load_pois_counts():
    pois = load_pois()
    assert len(pois) == 111
    synth = [p for p in pois if p.is_synthetic]
    assert len(synth) == 72                      # dòng G giữ nguyên, không xóa cứng
    assert all(p.id.startswith("G") for p in synth)
    assert all(p.document and p.norm_document for p in pois)


def test_poi_fields_parsed():
    poi = next(p for p in load_pois() if p.id == "C001")
    assert poi.name == "The Workshop Coffee"
    assert "yên tĩnh" in poi.attributes          # attributes tách ";" và giữ dấu
    assert poi.city == "TP.HCM"
    assert poi.lat > 0 and poi.lon > 0
    assert "yen tinh" in poi.norm_document       # norm_document đã bỏ dấu


def test_load_eval_counts():
    queries = load_eval()
    assert len(queries) == 60
    assert all(q.expected_ids for q in queries)
    single = sum(1 for q in queries if len(q.expected_ids) == 1)
    assert single == 43                          # 43/60 câu chỉ có 1 đáp án


def test_normalize_vi():
    assert normalize_vi("Quán Cà Phê Yên Tĩnh ĐẸP") == "quan ca phe yen tinh dep"
    assert normalize_vi("Đường") == "duong"      # đ/Đ → d
    assert normalize_vi("24/7") == "24/7"        # không phá token dạng số


def test_rules_extract_plan():
    from src.understanding.rules import extract_plan

    # Concept-expansion: "hẹn hò" phải ra concept lang_man (không match token thô)
    plan = extract_plan("nơi phù hợp để hẹn hò ở quận 1")
    assert "lang_man" in plan.attr_concepts

    # Polarity: "không quá đông" → cần yên tĩnh + PHỦ ĐỊNH đông khách
    plan = extract_plan("cafe không quá đông để họp nhóm")
    assert "yen_tinh" in plan.attr_concepts
    assert "dong_khach" in plan.neg_concepts
    # Span consumption: "họp nhóm" → phong_hop, KHÔNG fire concept nhom
    assert "phong_hop" in plan.attr_concepts
    assert "nhom" not in plan.attr_concepts

    # Popularity là cờ, không phải attribute
    plan = extract_plan("quán phở nổi tiếng hà nội")
    assert plan.want_pop and plan.city == "Hà Nội"
    assert "Nhà hàng" in plan.categories

    # Bẫy P009: "trên đường đi hạ long" là ngữ cảnh — không city, không landmark filter
    plan = extract_plan("cây xăng có toilet trên đường đi hạ long")
    assert plan.city is None
    assert "Trạm xăng" in plan.categories
    assert "toilet" in plan.attr_concepts


def test_name_match_strict():
    from src.ranking.signals import name_match
    from src.understanding.rules import extract_plan

    pois = {p.id: p for p in load_pois()}
    # Khớp đúng tên đầy đủ → 1.0
    assert name_match(extract_plan("đường tới The Workshop Coffee quận 1"), pois["C001"]) == 1.0
    # Token trùng lẻ tẻ (city/district trong tên) KHÔNG được ăn điểm — phải strict
    assert name_match(extract_plan("mua sắm ăn uống trung tâm quận 1"), pois["G036"]) == 0.0


def test_restore_diacritics():
    """Restore CHỈ cho nhánh dense: câu không dấu → có dấu (phrase-level thắng
    nhập nhằng từ đơn); câu có dấu và English phải GIỮ NGUYÊN tuyệt đối.
    """
    from src.understanding.diacritics import restore_diacritics as r

    # Phrase-level: "cho" giới từ vs "chợ" — gazetteer/concept phrase phải thắng
    assert r("mua sam an uong gan cho ben thanh") == "mua sắm ăn uống gần chợ bến thành"
    assert r("quan cafe yen tinh lam viec gan ho guom") == "quán cafe yên tĩnh làm việc gần hồ gươm"
    # Câu đã có dấu → no-op (eval public toàn câu có dấu — bất biến 60/60)
    assert r("cafe có wifi gần hồ gươm") == "cafe có wifi gần hồ gươm"
    # English → không được phá (guard tỉ lệ token phục hồi được)
    assert r("quiet coffee shop to work near hoan kiem") == "quiet coffee shop to work near hoan kiem"
    assert r("restaurant for date night in saigon") == "restaurant for date night in saigon"
    # Số/ký hiệu giữ nguyên
    assert "24/7" in r("atm rut tien 24/7 gan pho di bo")


def test_abbreviation_expansion():
    """Viết tắt expand whole-word ở ĐẦU pipeline; từ thật không được biến dạng."""
    from src.understanding.abbreviations import expand_abbreviations as ex

    assert ex("bv gần đây") == "bệnh viện gần đây"
    assert ex("cf yên tĩnh q1") == "cà phê yên tĩnh quận 1"
    assert ex("BV GẦN ĐÂY").lower().startswith("bệnh viện")  # case-insensitive
    # Bẫy: token trùng từ thật / không phải viết tắt → giữ nguyên tuyệt đối
    assert ex("bun bo") == "bun bo"
    assert ex("quán bún bò") == "quán bún bò"
    assert ex("cafe q1a") == "cafe q1a"  # q1a không phải qN


def test_abbreviation_retrieval():
    """Câu viết tắt phải ra đúng loại POI (path v2 không dense cho nhanh)."""
    from src.ranking.reranker import RerankRetriever, WEIGHTS_WITH_DISTANCE
    from src.retrieval.bm25 import BM25Retriever

    pois = load_pois()
    by_id = {p.id: p for p in pois}
    r = RerankRetriever(pois, base=BM25Retriever(pois), weights=WEIGHTS_WITH_DISTANCE)

    assert by_id[r.search("bv gần đây", k=3)[0]].category == "Bệnh viện"
    assert r.search("ks gần biển đà nẵng", k=3)[0] in {"H001", "H002"}
    assert r.search("cf yên tĩnh q1", k=3)[0] == "C001"


def test_no_accent_regression():
    """Câu KHÔNG DẤU phải ra đúng gold — dense mù chỗ này, BM25 (bỏ dấu 2 phía)
    + rules (match trên norm) + district phải gánh. Chạy path v2 (không dense) cho nhanh.
    """
    from src.ranking.reranker import RerankRetriever, WEIGHTS_WITH_DISTANCE
    from src.retrieval.bm25 import BM25Retriever

    pois = load_pois()
    r = RerankRetriever(pois, base=BM25Retriever(pois), weights=WEIGHTS_WITH_DISTANCE)

    cases = [
        ("quan ca phe yen tinh de lam viec", {"C001", "C004", "C007"}),
        ("cafe co wifi gan ho guom", {"C003", "C004"}),
        ("benh vien cap cuu 24/7 ha noi", {"S005"}),
        ("tram sac xe dien o da nang", {"S004"}),
        ("khach san gan bien da nang co ho boi", {"H001", "H002"}),
    ]
    for query, gold in cases:
        top1 = r.search(query, k=5)[0]
        assert top1 in gold, f"no-accent fail: {query!r} → {top1}, muốn {gold}"


def test_bm25_retriever_smoke():
    pois = load_pois()
    retriever = BM25Retriever(pois)
    assert isinstance(retriever, Retriever)      # đúng Protocol để ablation swap được

    top = retriever.search("quán cà phê yên tĩnh để làm việc", k=10)
    assert len(top) == 10
    valid_ids = {p.id for p in pois}
    assert set(top) <= valid_ids
    assert top == retriever.search("quán cà phê yên tĩnh để làm việc", k=10)  # deterministic

    # Bỏ dấu 2 phía: câu KHÔNG DẤU phải ra cùng kết quả với câu có dấu
    assert retriever.search("quan ca phe yen tinh de lam viec", k=10) == top
