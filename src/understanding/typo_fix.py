"""Typo correction BẢO THỦ — sửa lỗi gõ thật ("nhà hàg"→"nhà hàng"), query-side ONLY.

TRIẾT LÝ: thà BỎ SÓT còn hơn SỬA NHẦM. Mọi luật đều nghiêng về giữ nguyên:
  a. Token khớp vocab (so trong không-gian BỎ DẤU) → để YÊN tuyệt đối.
  b. Chỉ sửa edit distance = 1 (Levenshtein tự cài, zero-dep, deterministic).
  c. Chỉ sửa khi có ĐÚNG MỘT ứng viên distance-1 (0 hoặc ≥2 → giữ nguyên).
  d. Token < 3 ký tự → không đụng (quá dễ va chạm).
  e. Token chứa số → không đụng. Token đã qua abbreviation/diacritic thì
     đương nhiên khớp vocab → luật (a) tự bảo vệ.

Hai tập từ vựng TÁCH BIỆT có chủ đích:
  - KNOWN (để-yên): rộng nhất có thể — thêm cả description/address words và
    function-words tiếng Việt thường gặp. Càng rộng càng ÍT token bị xét → an toàn.
  - TARGETS (đích sửa): đúng spec — POI name/attribute/tag + lexicon +
    landmark/district/city. Hẹp để không sửa về phía từ vô nghĩa.

So khớp HAI TẦNG theo bản chất token:
  - Token CÓ DẤU ("hàg") → so edit-distance trong không-gian CÓ DẤU: phân biệt
    cao ("hàg"→"hàng" duy nhất; "hải" cách 2) → sửa được typo thật.
  - Token KHÔNG DẤU ("hag") → so trong không-gian bỏ dấu: nhiều hàng xóm
    ("hang"/"hai") → thường bị luật (c) chặn — chấp nhận bỏ sót, đúng triết lý.
Tắt bằng 1 dòng: config.ENABLE_TYPO_FIX (env TASCO_TYPO_FIX=0).
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from src import config
from src.data_loader import load_pois, normalize_vi

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_HAS_DIGIT = re.compile(r"\d")

# Function-words tiếng Việt hay gặp trong query mà data/lexicon không chứa đủ —
# nằm trong KNOWN để tuyệt đối không bị coi là typo ("cuối tuần" ≠ "cuối quận").
_STOPWORDS = """
va và la là cua của cho voi với o ở tai tại den đến từ tu ve về đi di đâu dau nao nào
gì gi khi nếu neu thì thi mà ma hay hoặc hoac cũng cung đã da đang se sẽ có co không khong
một mot hai ba bốn bon năm nam sáu sau bảy bay tám tam chín chin mười muoi
này nay đó do kia ấy ay tôi toi bạn ban mình minh chúng chung
buổi buoi sáng trưa trua chiều chieu tối toi đêm dem ngày ngay tuần tuan tháng thang cuối cuoi sớm som
được duoc cần can muốn muon tìm tim kiếm kiem chỗ nơi noi quanh gần đây day
người nguoi khách khach hàng nhất nhat rất rat hơi hoi khá kha
quá qua vừa vua lắm luôn luon nữa nua thêm them đừng dung nên nen phải phai
cả ca mỗi moi từng tung vẫn van đều deu chỉ chi thật that siêu sieu cực cuc hết het còn
""".split()


# Operator and unit keywords used in numeric and attribute constraint parsing
_GRAMMAR_TERMS = """
duoi dưới truoc trước kem nho hon nhỏ hơn it hon ít hơn duoi muc dưới mức thap hon thấp hơn
tren trên sau hon hơn lon hon lớn hơn nhieu hon nhiều hơn tren muc trên mức cao hon cao hơn
tam tầm khoang khoảng dung đúng bang bằng trieu triệu tr star sao gio giờ g h k am pm
sang sáng chieu chiều toi tối dem đêm danh gia đánh giá rating gia giá tien tiền mo cua mở cửa dong cua đóng cửa
""".split()


def _words(*texts: str) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(normalize_vi(w) for w in _WORD_RE.findall(t or ""))
    return out


@lru_cache(maxsize=1)
def _vocabs() -> tuple[frozenset[str], dict[str, str], frozenset[str], frozenset[tuple[str, str]]]:
    """(KNOWN norm-keys, TARGETS {norm_key→có dấu}, TARGETS_ACCENTED, TARGET_BIGRAMS).

    TARGET_BIGRAMS: cặp từ liền kề (normalize) trong các cụm target thật
    ("nhà hàng", "khách sạn"…) — làm bằng chứng ngữ cảnh khi tie-break.
    """
    targets: dict[str, str] = {}
    targets_accented: set[str] = set()
    bigrams: set[tuple[str, str]] = set()

    def add_target(phrase: str) -> None:
        words = _WORD_RE.findall(phrase or "")
        for w in words:
            targets.setdefault(normalize_vi(w), w.lower())
            targets_accented.add(w.lower())
        keys = [normalize_vi(w) for w in words]
        bigrams.update(zip(keys, keys[1:]))

    cats = yaml.safe_load(config.CATEGORIES_YAML.read_text(encoding="utf-8"))
    for entry in cats.values():
        add_target(str(entry["canonical"]))
        for s in entry["synonyms"]:
            add_target(str(s))
    concepts = yaml.safe_load(config.ATTRIBUTE_CONCEPTS_YAML.read_text(encoding="utf-8"))
    for entry in concepts.values():
        for t in list(entry.get("tokens", [])) + list(entry.get("surface", []) or []):
            add_target(str(t))
    gaz = yaml.safe_load(config.GAZETTEER_YAML.read_text(encoding="utf-8"))
    for entry in gaz.values():
        for n in entry["names"]:
            add_target(str(n))
        add_target(str(entry["city"]))

    known: set[str] = set(_STOPWORDS) | {normalize_vi(w) for w in _STOPWORDS}
    known |= set(_GRAMMAR_TERMS) | {normalize_vi(w) for w in _GRAMMAR_TERMS}
    for p in load_pois():
        for field in (p.name, p.category, p.sub_category, p.district, p.city):
            add_target(field)
        for tok in p.attributes + p.tags:
            add_target(tok)
        # description/address CHỈ vào KNOWN (mở rộng vùng để-yên, không làm đích sửa)
        known |= _words(p.description, p.address)

    known |= set(targets)
    return frozenset(known), targets, frozenset(targets_accented), frozenset(bigrams)


def _edit1(a: str, b: str) -> bool:
    """Levenshtein distance == 1 (một-pass, deterministic)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or a == b:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    # a ngắn hơn hoặc bằng b; lb - la ∈ {0, 1}
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
            continue
        diff += 1
        if diff > 1:
            return False
        if la == lb:
            i += 1; j += 1  # substitution
        else:
            j += 1          # insertion vào b
    return True  # ký tự thừa cuối (nếu có) là edit thứ diff+…≤1


def _has_diacritics(word: str) -> bool:
    return normalize_vi(word) != word.lower()


# Ký tự rác cuối âm tiết: chữ KHÔNG kết thúc âm tiết thuần Việt (pattern gõ
# Telex thiếu IME — f/j là phím dấu huyền/nặng). Cố ý KHÔNG gồm s/r/x
# (kết thúc từ tiếng Anh quá phổ biến: hotels, bar, box...).
_TRAILING_JUNK = frozenset("fjzw")


def _resolve(core: str, prev_k: str | None, next_k: str | None) -> str | None:
    """Luật single-typo hiện có: two-tier + unique-candidate + bigram tie-break.

    Trả dạng có dấu nếu resolve được DUY NHẤT, else None. Giả định caller đã
    kiểm tra core ∉ KNOWN và các điều kiện len/digit.
    """
    known, targets, targets_accented, bigrams = _vocabs()
    key = normalize_vi(core)
    if _has_diacritics(core):
        # Không-gian CÓ DẤU: phân biệt cao — typo có dấu sửa được an toàn
        candidates = {t for t in targets_accented if _edit1(core.lower(), t)}
    else:
        # Không-gian bỏ dấu: nhiều hàng xóm — thường bị luật (c) chặn, chấp nhận
        candidates = {targets[k] for k in targets if _edit1(key, k)}
    if len(candidates) >= 2:
        # Tie-break bằng NGỮ CẢNH: chỉ nhận ứng viên tạo bigram target THẬT
        # với từ liền kề ("nhà hàg": "nhà hàng" ∈ bigram, "nhà hàn" ∉ → chọn hàng).
        # Vẫn phải DUY NHẤT sau lọc — không đoán mò.
        candidates = {c for c in candidates
                      if (prev_k, normalize_vi(c)) in bigrams
                      or (normalize_vi(c), next_k) in bigrams}
    return next(iter(candidates)) if len(candidates) == 1 else None


def strip_trailing_junk(core: str, prev_k: str | None, next_k: str | None) -> str | None:
    """Double-typo "ký tự rác cuối token" ("hangf") → QUY VỀ single-typo.

    KHÔNG nới edit distance — bỏ TỐI ĐA 1 ký tự rác {f,j,z,w} cuối token, rồi
    phần còn lại phải "đi tiếp được" bằng đúng luật cũ:
      - khớp vocab trực tiếp (lấy dạng có dấu từ targets — "hang"→"hàng"), hoặc
      - resolve single-typo hợp lệ (unique/two-tier/bigram).
    Không thỏa → trả None (giữ nguyên token). Caller đã đảm bảo core ∉ KNOWN —
    "jazz"/"view"/brand khớp vocab không bao giờ tới đây.
    """
    if len(core) < 4 or core[-1].lower() not in _TRAILING_JUNK:
        return None  # stripped phải còn ≥3 ký tự — giữ tinh thần luật (d)
    _, targets, _, _ = _vocabs()
    stripped = core[:-1]
    skey = normalize_vi(stripped)
    if skey in _vocabs()[0]:            # khớp vocab trực tiếp
        return targets.get(skey, stripped)  # ngoài targets (từ description) → giữ dạng gõ
    return _resolve(stripped, prev_k, next_k)  # hoặc thành ứng viên single-typo hợp lệ


@lru_cache(maxsize=512)
def correct_typos(text: str) -> str:
    """Sửa typo theo luật bảo thủ; không có gì đáng sửa → trả nguyên (idempotent)."""
    known, _, _, _ = _vocabs()
    tokens = text.split()
    keys = [normalize_vi(t.strip(".,;:!?()\"'")) for t in tokens]
    out: list[str] = []
    for i, token in enumerate(tokens):
        core = token.strip(".,;:!?()\"'")
        key = keys[i]
        if (len(core) < 3 or _HAS_DIGIT.search(core) or not core.isalpha()
                or key in known):
            out.append(token)
            continue
        prev_k = keys[i - 1] if i > 0 else None
        next_k = keys[i + 1] if i + 1 < len(keys) else None
        # 1) luật single-typo nguyên bản trước — hành vi cũ giữ NGUYÊN 100%;
        # 2) thất bại mới thử strip ký tự rác cuối (double-typo → single-typo).
        fixed = _resolve(core, prev_k, next_k)
        if fixed is None:
            fixed = strip_trailing_junk(core, prev_k, next_k)
        out.append(token.replace(core, fixed) if fixed else token)
    return " ".join(out)
