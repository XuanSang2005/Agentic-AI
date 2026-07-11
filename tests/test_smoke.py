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
    from src.data_loader import extract_unique_attributes
    from src.retrieval.dense import AttributeIndex
    from src.understanding.rules import extract_plan

    pois = load_pois()
    attr_index = AttributeIndex(extract_unique_attributes(pois))

    # Subtractive parsing: "yên tĩnh" is extracted as remaining text after
    # category/city/district are consumed, then matched via radius search
    plan = extract_plan("quán cà phê yên tĩnh để làm việc", attr_index=attr_index)
    assert any(normalize_vi(a) == "yen tinh" for a in plan.attr_concepts)

    # Polarity: "không quá đông" → cần yên tĩnh + PHỦ ĐỊNH đông khách
    plan = extract_plan("cafe không quá đông", attr_index=attr_index)
    assert any(normalize_vi(a) == "yen tinh" for a in plan.attr_concepts)
    assert any(normalize_vi(a) == "dong khach" for a in plan.neg_concepts)

    # Popularity là cờ, không phải attribute
    plan = extract_plan("quán phở nổi tiếng hà nội", attr_index=attr_index)
    assert plan.want_pop and plan.city == "Hà Nội"
    assert "Nhà hàng" in plan.categories

    # Bẫy P009: "trên đường đi hạ long" là ngữ cảnh — không city, không landmark filter
    plan = extract_plan("cây xăng có toilet trên đường đi hạ long", attr_index=attr_index)
    assert plan.city is None
    assert "Trạm xăng" in plan.categories


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


def test_typo_correction():
    """Typo fix bảo thủ: sửa đúng typo thật, TUYỆT ĐỐI không đụng từ đúng."""
    from src.understanding.typo_fix import correct_typos as fix

    # Sửa đúng (có dấu, unique candidate / bigram tie-break)
    assert fix("nhà hàg gia đình") == "nhà hàng gia đình"
    assert fix("khách sạnn gần biển") == "khách sạn gần biển"
    # BẪY — phải giữ nguyên tuyệt đối:
    assert fix("bún bò") == "bún bò"                    # từ thật trong vocab
    assert fix("quán bún bò") == "quán bún bò"
    assert fix("vinmec ở đâu") == "vinmec ở đâu"        # tên riêng hợp lệ trong data
    assert fix("cf ok") == "cf ok"                      # token < 3 ký tự
    assert fix("hag gan ho") == "hag gan ho"            # ≥2 ứng viên (hang/hai) → bỏ
    assert fix("nhà hàn quốc") == "nhà hàn quốc"        # "hàn" là từ thật (sông Hàn)
    assert fix("cafe không quá đông") == "cafe không quá đông"  # function word "quá"


def test_double_typo_strip_trailing_junk():
    """Double-typo "ký tự rác cuối" quy về single-typo — không nới edit distance."""
    from src.ranking.reranker import preprocess_query as pq
    from src.understanding.typo_fix import correct_typos as fix

    # Sửa đúng (qua chain đầy đủ expand→restore→typo)
    assert pq("nhaf hangf") == "nhà hàng"
    assert pq("khach sanj") == "khách sạn"
    assert pq("cf") == "cà phê"  # abbreviation vẫn chạy trước, không bị typo đụng
    # BẪY — giữ nguyên tuyệt đối:
    assert fix("jazz") == "jazz"                      # strip xong không đi tiếp được
    assert fix("bbq") == "bbq"                        # 'q' không phải ký tự rác
    assert fix("grabfood") == "grabfood"              # brand, không kết thúc rác
    assert fix("view đẹp") == "view đẹp"              # 'view' kết thúc w nhưng KHỚP vocab
    assert fix("nghe nhạc jazz") == "nghe nhạc jazz"
    assert fix("asdfw") == "asdfw"                    # strip rồi vẫn bí → nguyên
    # Bẫy cũ vẫn phải sống sau khi thêm strip:
    assert fix("bún bò") == "bún bò"
    assert fix("vinmec ở đâu") == "vinmec ở đâu"
    assert fix("nhà hàn quốc") == "nhà hàn quốc"


def test_typo_flag_off(monkeypatch):
    """Tắt được bằng 1 flag — preprocess không đụng gì khi ENABLE_TYPO_FIX=False."""
    from src import config
    from src.ranking.reranker import preprocess_query

    monkeypatch.setattr(config, "ENABLE_TYPO_FIX", False)
    assert preprocess_query("nhà hàg gia đình") == "nhà hàg gia đình"
    monkeypatch.setattr(config, "ENABLE_TYPO_FIX", True)
    assert preprocess_query("nhà hàg gia đình") == "nhà hàng gia đình"


def test_typo_retrieval():
    """Câu có typo phải ra đúng loại POI (path v2 không dense cho nhanh)."""
    from src.ranking.reranker import RerankRetriever, WEIGHTS_WITH_DISTANCE
    from src.retrieval.bm25 import BM25Retriever

    pois = load_pois()
    by_id = {p.id: p for p in pois}
    r = RerankRetriever(pois, base=BM25Retriever(pois), weights=WEIGHTS_WITH_DISTANCE)
    assert by_id[r.search("nhà hàg cho gia đình", k=3)[0]].category == "Nhà hàng"
    assert by_id[r.search("khách sạnn gần biển đà nẵng", k=3)[0]].category == "Khách sạn"


def test_constraint_reasoning():
    """Lớp reasoning: tách ràng buộc có kiểu, chấm thỏa/nới trên field data thật."""
    from src.data_loader import extract_unique_attributes
    from src.ranking.reranker import preprocess_query
    from src.reasoning.constraints import annotate, parse_constraints, score_constraint
    from src.retrieval.dense import AttributeIndex
    from src.understanding.rules import extract_plan

    all_pois = load_pois()
    pois = {p.id: p for p in all_pois}
    attr_index = AttributeIndex(extract_unique_attributes(all_pois))

    # Câu nhiều ràng buộc: category + location + attributes
    plan = extract_plan(preprocess_query(
        "quán cà phê yên tĩnh có chỗ đậu xe ở quận 1"), attr_index=attr_index)
    cons = parse_constraints(plan)
    types = {c.type for c in cons}
    assert "category" in types
    assert "location" in types
    assert all(c.priority == 2 for c in cons if c.type in ("category", "location"))
    assert all(c.priority == 1 for c in cons if c.type in ("attribute", "time", "price"))

    # location (district Quận 1): đúng quận → 1.0; khác city → 0
    loc_con = next(c for c in cons if c.type == "location")
    assert score_constraint(pois["C001"], loc_con) == 1.0
    assert score_constraint(pois["C004"], loc_con) == 0.0

    # Negation: "không quá đông" → thỏa khi KHÔNG có đông khách
    plan_neg = extract_plan(preprocess_query("cafe không quá đông"), attr_index=attr_index)
    neg_cons = parse_constraints(plan_neg)
    neg = next((c for c in neg_cons if c.negated), None)
    if neg is not None:
        assert score_constraint(pois["C001"], neg) == 1.0  # không đông khách
        assert score_constraint(pois["C002"], neg) == 0.0  # đông khách

    # Bẫy P009: "trên đường đi hạ long" KHÔNG được thành ràng buộc location
    plan_trap = extract_plan(preprocess_query("cây xăng có toilet trên đường đi hạ long"),
                             attr_index=attr_index)
    assert not any(c.type == "location" for c in parse_constraints(plan_trap))


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


def test_poi_status_display_field():
    """status (badge verify, policy A): NGUỒN XLSX không có cột → default
    'active' (test ép nhánh xlsx — nguồn postgres mang status thật sau
    verify/backfill); DTO chuyển tiếp nguyên vẹn, KHÔNG rò vào document/ranking."""
    from src.api.dto import to_place_result
    from src.data_loader import _pois_from_xlsx
    from src.search import SearchHit

    poi = _pois_from_xlsx()[0]
    assert poi.status == "active"
    assert "active" not in poi.document      # status không được rò vào corpus
    assert to_place_result(SearchHit(poi=poi, score=0.5)).status == "active"


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


def test_superlative_sorting_intent():
    from src.understanding.rules import extract_plan
    plan1 = extract_plan("quán ăn rẻ nhất")
    assert plan1.sort_by == "price"
    assert plan1.sort_order == "asc"
    assert "re nhat" not in plan1.clean_query
    assert "quán ăn" in plan1.clean_query

    plan2 = extract_plan("khách sạn đánh giá cao nhất")
    assert plan2.sort_by == "rating"
    assert plan2.sort_order == "desc"
    assert "cao nhat" not in plan2.clean_query

    plan3 = extract_plan("quán cafe giá thấp nhất")
    assert plan3.sort_by == "price"
    assert plan3.sort_order == "asc"


def test_currently_open_intent():
    from src.understanding.rules import extract_plan
    plan = extract_plan("quán cafe đang mở cửa")
    assert plan.current_time_open is True
    assert "dang mo cua" not in plan.clean_query


def test_is_open_at_helper():
    from src.ranking.signals import is_open_at
    # 24/7
    assert is_open_at("24/7", 500) is True
    # Normal hours (08:00 - 22:00 -> 480 to 1320)
    assert is_open_at("08:00-22:00", 600) is True
    assert is_open_at("08:00-22:00", 400) is False
    # Midnight wrapping (18:00 - 02:00 -> 1080 to 120)
    assert is_open_at("18:00-02:00", 1100) is True
    assert is_open_at("18:00-02:00", 60) is True
    assert is_open_at("18:00-02:00", 600) is False


def test_reranker_sorting_override():
    from src.search import SearchService
    service = SearchService()

    # "giá rẻ nhất" -> price_level should be ascending, 0/missing price levels at the end
    hits = service.search("quán cafe giá rẻ nhất", limit=5)
    assert len(hits) > 0
    # verify they are ordered by price_level asc (excluding 0)
    price_levels = [h.poi.price_level for h in hits if h.poi.price_level > 0]
    assert price_levels == sorted(price_levels)

    # "đánh giá cao nhất" -> rating should be descending
    hits2 = service.search("quán ăn đánh giá cao nhất", limit=5)
    assert len(hits2) > 0
    ratings = [h.poi.rating for h in hits2]
    assert ratings == sorted(ratings, reverse=True)
