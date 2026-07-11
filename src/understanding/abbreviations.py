"""Abbreviation expansion: viết tắt → dạng đầy đủ CÓ DẤU — bước ĐẦU TIÊN của query.

Khác diacritic restore (chỉ nhánh dense): token viết tắt vô nghĩa với CẢ BA
nhánh (BM25/dense/rules) nên expand TRƯỚC mọi thứ, áp cho query-side ONLY
(POI corpus không đụng). Bản expand ra chữ có dấu → các lớp sau
(normalize/diacritic-restore) chạy tiếp bình thường.

Nguyên tắc:
- WHITELIST curated (config/abbreviations.yaml — non-dev sửa được), match
  whole-word (\\b), case-insensitive. Không substring.
- Guard: token trùng MỘT TỪ THẬT trong POI DATA (name/category/description…,
  đã bỏ dấu) → không expand — chống kiểu "bo"↔"bún bò" nếu sau này thêm seed ẩu.
  Guard cố ý chỉ nhìn DATA, không nhìn lexicon surface (surface chứa chính các
  dạng viết tắt như "tttm" — nhìn cả lexicon sẽ tự chặn nhầm seed hợp lệ).
  Chạy lúc BUILD map: seed va vocab bị loại hẳn.
- Deterministic, offline, zero-dep.
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from src import config
from src.data_loader import load_pois, normalize_vi

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@lru_cache(maxsize=1)
def _data_vocab() -> frozenset[str]:
    """Từ đơn (bỏ dấu) xuất hiện trong POI data thật — nguồn guard."""
    words: set[str] = set()
    for p in load_pois():
        for field in (p.name, p.category, p.sub_category, p.district, p.city,
                      p.address, p.description, " ".join(p.attributes), " ".join(p.tags)):
            words.update(normalize_vi(w) for w in _WORD_RE.findall(field or ""))
    return frozenset(words)

@lru_cache(maxsize=1)
def _seed() -> dict[str, str]:
    """viết tắt → dạng đầy đủ, từ config/abbreviations.yaml (giữ thứ tự khai báo)."""
    raw = yaml.safe_load(config.ABBREVIATIONS_YAML.read_text(encoding="utf-8"))
    return {str(abbr): str(full) for abbr, full in raw.items()}

# q1/q12 → quận 1/quận 12 (KHÔNG expand "q" trần)
_Q_DISTRICT = re.compile(r"(?<![a-zA-Z0-9đĐ])[qQ](\d{1,2})(?![a-zA-Z0-9])")


@lru_cache(maxsize=1)
def _rules() -> list[tuple[re.Pattern, str]]:
    """Compile seed → [(boundary_pattern, full_form)], loại seed va vocab data thật."""
    vocab = _data_vocab()
    rules = []
    for abbr, full in _seed().items():
        if normalize_vi(abbr) in vocab:
            continue  # trùng từ thật trong data (vd "hcm" từ "TP.HCM") — bỏ để an toàn
        pat = re.compile(rf"(?<![\wđĐ]){re.escape(abbr)}(?![\wđĐ])", re.IGNORECASE)
        rules.append((pat, full))
    return rules


@lru_cache(maxsize=512)
def expand_abbreviations(text: str) -> str:
    """Expand viết tắt whole-word; text không có viết tắt → trả nguyên (idempotent)."""
    out = _Q_DISTRICT.sub(r"quận \1", text)
    for pat, full in _rules():
        out = pat.sub(full, out)
    return out
