"""Domain-specific diacritic restoration — CHỈ dùng cho nhánh DENSE.

Bối cảnh: e5 train trên text CÓ DẤU nên câu không dấu làm dense lệch; BM25 đã
bỏ-dấu-2-phía nên tự khỏe. Vì vậy restore chỉ được wire vào query-side của
DenseRetriever — BM25/rules vẫn nhận query gốc.

Cách làm (đã loại pyvi.ViUtils.add_accents — thử thật: "bên Thanh", "Cây xáng",
phá cả English "tô Work"): dictionary build từ CHÍNH data + lexicon của project
(POI name/category/district/description, 82 attr token + surface, gazetteer,
city). Domain hẹp (địa điểm/đặc điểm) nên map phủ phần lớn query thật:
  1. PHRASE trước (greedy longest-match 4→2 gram): "gan cho"→"gần chợ",
     "ben thanh"→"bến thành" — né nhập nhằng từ đơn ("cho" giới từ vs "chợ").
  2. WORD sau: dạng có dấu PHỔ BIẾN NHẤT trong corpus domain.
  3. Guard: câu đã có dấu → trả nguyên; câu trông như English (tỉ lệ token
     phục-hồi-được < 50%) → trả nguyên. Không đụng số/"24/7".

Deterministic, offline, không LLM/API — giữ đúng điểm mạnh pitch.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

import yaml

from src import config
from src.data_loader import load_pois, normalize_vi

_MAX_GRAM = 4
_MIN_COVERAGE = 0.5  # dưới ngưỡng này coi là câu ngoại ngữ → không restore
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # token chữ, bỏ số/punct


def _has_diacritics(text: str) -> bool:
    if "đ" in text.lower():
        return True
    return any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", text))


def _accented_phrases() -> list[str]:
    """Mọi cụm CÓ DẤU trong domain: lexicon + gazetteer + field POI + description."""
    # Alias city nằm trong regex của rules.py (không có trong YAML/data) — khai báo tay,
    # không thì "sai gon" bị vote nhầm thành "sài gọn" (từ surface "nhanh gọn")
    phrases: list[str] = ["sài gòn", "hồ chí minh", "hà nội", "đà nẵng", "đà lạt"]

    cats = yaml.safe_load(config.CATEGORIES_YAML.read_text(encoding="utf-8"))
    for entry in cats.values():
        phrases.append(str(entry["canonical"]))
        phrases.extend(str(s) for s in entry["synonyms"])

    concepts = yaml.safe_load(config.ATTRIBUTE_CONCEPTS_YAML.read_text(encoding="utf-8"))
    for entry in concepts.values():
        phrases.extend(str(t) for t in entry.get("tokens", []))
        phrases.extend(str(s) for s in entry.get("surface", []) or [])

    gaz = yaml.safe_load(config.GAZETTEER_YAML.read_text(encoding="utf-8"))
    for entry in gaz.values():
        phrases.extend(str(n) for n in entry["names"])
        phrases.append(str(entry["city"]))

    for p in load_pois():
        phrases.extend([p.name, p.category, p.sub_category, p.district, p.city])
        phrases.extend(p.attributes)
        # description cho tần suất từ đơn tự nhiên ("có", "gần", "phù hợp"…)
        phrases.extend(_WORD_RE.findall(p.description))

    return [p.lower() for p in phrases if p]


@lru_cache(maxsize=1)
def _maps() -> tuple[dict[str, str], dict[str, str]]:
    """(phrase_map, word_map): key = normalize_vi, value = dạng có dấu phổ biến nhất."""
    phrase_votes: dict[str, Counter] = {}
    word_votes: dict[str, Counter] = {}

    def vote(book: dict[str, Counter], key: str, value: str) -> None:
        book.setdefault(key, Counter())[value] += 1

    for phrase in _accented_phrases():
        norm = normalize_vi(phrase)
        words = phrase.split()
        if len(words) >= 2 and len(norm.split()) == len(words):
            vote(phrase_votes, norm, phrase)
        for w in words:
            if _WORD_RE.fullmatch(w):
                vote(word_votes, normalize_vi(w), w)

    pick = lambda votes: {k: c.most_common(1)[0][0] for k, c in votes.items()}
    return pick(phrase_votes), pick(word_votes)


@lru_cache(maxsize=512)
def restore_diacritics(text: str) -> str:
    """Phục hồi dấu cho câu KHÔNG DẤU; câu có dấu / trông như English → trả nguyên."""
    if not text.strip() or _has_diacritics(text):
        return text

    phrase_map, word_map = _maps()
    tokens = normalize_vi(text).split()
    out: list[str] = []
    covered = 0  # token được map phục hồi RA DẠNG KHÁC (bằng chứng tiếng Việt)
    n_alpha = sum(1 for t in tokens if _WORD_RE.fullmatch(t)) or 1

    i = 0
    while i < len(tokens):
        matched = False
        for n in range(min(_MAX_GRAM, len(tokens) - i), 1, -1):
            gram = " ".join(tokens[i:i + n])
            acc = phrase_map.get(gram)
            if acc is not None:
                out.append(acc)
                covered += sum(1 for a, b in zip(acc.split(), tokens[i:i + n]) if a != b)
                i += n
                matched = True
                break
        if matched:
            continue
        tok = tokens[i]
        acc = word_map.get(tok, tok)
        if acc != tok:
            covered += 1
        out.append(acc)
        i += 1

    if covered / n_alpha < _MIN_COVERAGE:
        return text  # ít bằng chứng tiếng Việt (câu English/mã) — không đụng
    return " ".join(out)
