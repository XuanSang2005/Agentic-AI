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
