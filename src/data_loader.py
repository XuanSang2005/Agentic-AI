"""Load POI_Dataset + Public_Evaluation từ data/*.xlsx thành POI documents.

Nguồn sự thật duy nhất về data — mọi lớp trên (BM25, dense, filter, rerank, eval)
đọc qua module này, không tự mở xlsx.

- GIỮ NGUYÊN dấu tiếng Việt trong document lưu trữ; chỉ bỏ dấu ở tầng matcher
  (normalize_vi là util chuẩn hóa dùng chung cho MỌI bước so khớp).
- Text field để BM25/dense dùng chung được dựng sẵn Ở ĐÂY (document / norm_document)
  — tránh mỗi retriever tự ghép một kiểu rồi lệch nhau.
- ĐỪNG xóa dòng G — chỉ đánh dấu is_synthetic để eval chẩn đoán và L3 down-rank.
  TUYỆT ĐỐI không dùng is_synthetic để lọc trong retriever.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import openpyxl

from src import config


def normalize_vi(text: str) -> str:
    """Chuẩn hóa tiếng Việt để SO KHỚP: lowercase + bỏ dấu (NFD, bỏ combining marks, đ→d).

    Chỉ dùng khi match — bản gốc có dấu luôn được giữ để hiển thị/embed.
    Nhờ bỏ dấu cả 2 phía (query lẫn document), câu không dấu vẫn match được.
    """
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # NFD không tách được đ/Đ — thay tay sau khi đã lowercase
    return text.replace("đ", "d")


@dataclass
class POI:
    id: str
    name: str
    brand: str
    category: str
    sub_category: str
    city: str
    district: str
    address: str
    lat: float
    lon: float
    rating: float
    review_count: int
    popularity: float
    price_level: int
    opening_hours: str
    attributes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""
    is_synthetic: bool = False  # id bắt đầu bằng "G" — CHỈ để chẩn đoán trong eval
    # Text field dựng sẵn cho retrieval — BM25 và dense sau này DÙNG CHUNG:
    document: str = ""       # giữ nguyên dấu — để hiển thị / embed
    norm_document: str = ""  # normalize_vi(document) — để BM25 match


@dataclass
class EvalQuery:
    id: str
    query: str
    category: str      # query_category (Semantic Search, Intent Search, ...)
    difficulty: str    # Easy / Medium / Hard
    expected_ids: list[str] = field(default_factory=list)
    # Giữ RAW string — đừng tokenize theo dấu ("24/7" bị cắt sai); parse ra slot để sau.
    expected_requirements: str = ""
    ranking_signals: list[str] = field(default_factory=list)


def _rows_of(ws) -> list[dict]:
    """Sheet → list[dict] theo header dòng 1, bỏ dòng trống."""
    rows = ws.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(rows)]
    out = []
    for row in rows:
        if all(v is None for v in row):
            continue
        out.append(dict(zip(header, row)))
    return out


def _text(value) -> str:
    """Cell → str đã strip; None → chuỗi rỗng (giữ nguyên dấu)."""
    return "" if value is None else str(value).strip()


def _split_list(value, sep: str = ";") -> list[str]:
    """Cell "a;b;c" → ["a", "b", "c"] (strip từng phần, bỏ phần rỗng)."""
    return [p.strip() for p in _text(value).split(sep) if p.strip()]


@lru_cache(maxsize=1)
def _load_workbook_rows() -> dict[str, list[dict]]:
    """Đọc cả 5 sheet một lần, cache lại (xlsx chỉ ~34KB)."""
    wb = openpyxl.load_workbook(config.DATA_XLSX, read_only=True, data_only=True)
    return {name: _rows_of(wb[name]) for name in wb.sheetnames}


def load_pois() -> list[POI]:
    """POI_Dataset → list[POI], giữ nguyên thứ tự dòng, KHÔNG lọc dòng G."""
    pois = []
    for r in _load_workbook_rows()[config.SHEET_POI]:
        poi = POI(
            id=_text(r["poi_id"]),
            name=_text(r["poi_name"]),
            brand=_text(r["brand"]),
            category=_text(r["category"]),
            sub_category=_text(r["sub_category"]),
            city=_text(r["city"]),
            district=_text(r["district"]),
            address=_text(r["address"]),
            lat=float(r["latitude"] or 0.0),
            lon=float(r["longitude"] or 0.0),
            rating=float(r["rating"] or 0.0),
            review_count=int(r["review_count"] or 0),
            popularity=float(r["popularity_score"] or 0.0),
            price_level=int(r["price_level"] or 0),
            opening_hours=_text(r["opening_hours"]),
            attributes=_split_list(r["attributes"]),
            tags=_split_list(r["tags"]),
            description=_text(r["description"]),
        )
        poi.is_synthetic = poi.id.startswith("G")
        # Document ghép các field ngữ nghĩa (không address/brand — để location/brand
        # xử lý có chủ đích ở L1/L2, không nhiễu lexical).
        poi.document = " ".join(part for part in [
            poi.name,
            poi.category,
            poi.sub_category,
            poi.district,
            poi.city,
            " ".join(poi.attributes),
            " ".join(poi.tags),
            poi.description,
        ] if part)
        poi.norm_document = normalize_vi(poi.document)
        pois.append(poi)
    return pois


def load_eval() -> list[EvalQuery]:
    """Public_Evaluation → list[EvalQuery]. expected_ids tách ";", signals tách ","."""
    return [
        EvalQuery(
            id=_text(r["query_id"]),
            query=_text(r["input_query"]),
            category=_text(r["query_category"]),
            difficulty=_text(r["difficulty"]),
            expected_ids=_split_list(r["expected_top_poi_ids"]),
            expected_requirements=_text(r["expected_semantic_requirements"]),
            ranking_signals=_split_list(r["ranking_signals_to_use"], sep=","),
        )
        for r in _load_workbook_rows()[config.SHEET_EVAL]
    ]
