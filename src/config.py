"""Điểm truy cập cấu hình TRUNG TÂM — giá trị load từ config/settings.yaml đúng 1 lần.

Tách bạch rõ:
- config/settings.yaml : giá trị KHÔNG nhạy cảm (path, model, ngưỡng, trọng số, flag default).
- ENV                  : secret (API key/token) + override feature flag — xem .env.example.

Không hardcode credentials — API key/token CHỈ đọc từ biến môi trường. Có cờ
DETERMINISTIC_MODE (không cần API key): planner rơi về rule-based cho test/demo.

Mọi module đọc qua đây (config.DATA_XLSX, config.settings()...), KHÔNG tự
yaml.load settings.yaml ở chỗ khác.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_YAML = ROOT / "config" / "settings.yaml"


# --- Schema typed cho settings.yaml (frozen — config là read-only) ---

@dataclass(frozen=True)
class PathsCfg:
    data_xlsx: Path
    lexicon_dir: Path
    attribute_concepts: Path
    gazetteer: Path
    categories: Path
    abbreviations: Path
    city_aliases: Path
    embedding_cache_dir: Path
    reports_dir: Path
    readme_md: Path


@dataclass(frozen=True)
class SheetsCfg:
    poi: str
    eval: str
    taxonomy: str
    signals: str
    readme: str


@dataclass(frozen=True)
class SearchCfg:
    default_limit: int
    k_internal: int
    api_max_limit: int


@dataclass(frozen=True)
class RetrievalCfg:
    n_candidates: int
    pool_k: int


@dataclass(frozen=True)
class DiacriticsCfg:
    max_gram: int
    min_coverage: float


@dataclass(frozen=True)
class TypoCfg:
    min_token_len: int
    junk_strip_min_len: int
    trailing_junk_chars: str


@dataclass(frozen=True)
class UnderstandingCfg:
    semantic_cache_threshold: float
    landmark_near_cue_window: int
    diacritics: DiacriticsCfg
    typo: TypoCfg


@dataclass(frozen=True)
class ConstraintsCfg:
    satisfied_threshold: float
    time_concepts: frozenset[str]
    price_concepts: frozenset[str]
    price_max_level: dict[str, int]
    late_close_full_minutes: int
    late_close_partial_minutes: int
    near_km_full: float
    near_km_partial: float


@dataclass(frozen=True)
class IngestionCfg:
    max_batch_size: int


@dataclass(frozen=True)
class VerifyCfg:
    aws_region: str
    match_score_threshold: float
    max_distance_m: float
    place_types: tuple[str, ...]
    max_workers: int
    throttle_retry_delays: tuple[float, ...]


@dataclass(frozen=True)
class Settings:
    paths: PathsCfg
    sheets: SheetsCfg
    embedding_model: str
    eval_top_k: int
    search: SearchCfg
    retrieval: RetrievalCfg
    rerank_weights: dict[str, dict[str, float]]  # GIỮ thứ tự key như trong yaml
    understanding: UnderstandingCfg
    constraints: ConstraintsCfg
    ingestion: IngestionCfg
    verify: VerifyCfg
    features: dict[str, bool]


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Đọc settings.yaml MỘT LẦN cho cả process — mọi nơi khác gọi hàm này."""
    raw = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))
    und, cons = raw["understanding"], raw["constraints"]
    return Settings(
        paths=PathsCfg(**{k: ROOT / str(v) for k, v in raw["paths"].items()}),
        sheets=SheetsCfg(**{k: str(v) for k, v in raw["sheets"].items()}),
        embedding_model=str(raw["embedding"]["model"]),
        eval_top_k=int(raw["eval"]["top_k"]),
        search=SearchCfg(**{k: int(v) for k, v in raw["search"].items()}),
        retrieval=RetrievalCfg(**{k: int(v) for k, v in raw["retrieval"].items()}),
        rerank_weights={profile: {sig: float(w) for sig, w in weights.items()}
                        for profile, weights in raw["rerank_weights"].items()},
        understanding=UnderstandingCfg(
            semantic_cache_threshold=float(und["semantic_cache_threshold"]),
            landmark_near_cue_window=int(und["landmark_near_cue_window"]),
            diacritics=DiacriticsCfg(
                max_gram=int(und["diacritics"]["max_gram"]),
                min_coverage=float(und["diacritics"]["min_coverage"]),
            ),
            typo=TypoCfg(
                min_token_len=int(und["typo"]["min_token_len"]),
                junk_strip_min_len=int(und["typo"]["junk_strip_min_len"]),
                trailing_junk_chars=str(und["typo"]["trailing_junk_chars"]),
            ),
        ),
        constraints=ConstraintsCfg(
            satisfied_threshold=float(cons["satisfied_threshold"]),
            time_concepts=frozenset(str(c) for c in cons["time_concepts"]),
            price_concepts=frozenset(str(c) for c in cons["price_concepts"]),
            price_max_level={str(k): int(v) for k, v in cons["price_max_level"].items()},
            late_close_full_minutes=int(cons["late_close_full_minutes"]),
            late_close_partial_minutes=int(cons["late_close_partial_minutes"]),
            near_km_full=float(cons["near_km_full"]),
            near_km_partial=float(cons["near_km_partial"]),
        ),
        ingestion=IngestionCfg(
            max_batch_size=int(raw["ingestion"]["max_batch_size"]),
        ),
        verify=VerifyCfg(
            aws_region=str(raw["verify"]["aws_region"]),
            match_score_threshold=float(raw["verify"]["match_score_threshold"]),
            max_distance_m=float(raw["verify"]["max_distance_m"]),
            place_types=tuple(str(t) for t in raw["verify"]["place_types"]),
            max_workers=int(raw["verify"]["max_workers"]),
            throttle_retry_delays=tuple(float(d) for d in raw["verify"]["throttle_retry_delays"]),
        ),
        features={str(k): bool(v) for k, v in raw["features"].items()},
    )


_S = settings()

# --- Data & lexicon ---
DATA_XLSX = _S.paths.data_xlsx
LEXICON_DIR = _S.paths.lexicon_dir
ATTRIBUTE_CONCEPTS_YAML = _S.paths.attribute_concepts
GAZETTEER_YAML = _S.paths.gazetteer
CATEGORIES_YAML = _S.paths.categories
ABBREVIATIONS_YAML = _S.paths.abbreviations
CITY_ALIASES_YAML = _S.paths.city_aliases

# Tên sheet trong xlsx (đã verify bằng eval/verify_dataset.py)
SHEET_POI = _S.sheets.poi
SHEET_EVAL = _S.sheets.eval
SHEET_TAXONOMY = _S.sheets.taxonomy
SHEET_SIGNALS = _S.sheets.signals
SHEET_README = _S.sheets.readme

# --- Eval ---
REPORTS_DIR = _S.paths.reports_dir
README_MD = _S.paths.readme_md
EVAL_TOP_K = _S.eval_top_k  # số kết quả retriever trả cho eval (MRR tính trong top-k)

# --- L2 dense ---
EMBEDDING_MODEL = _S.embedding_model
EMBEDDING_CACHE_DIR = _S.paths.embedding_cache_dir  # .npy cache — xoá là tự build lại

# --- L1 LLM planner (chưa dùng ở slice BM25) ---
# Ngưỡng cosine cho semantic cache: đủ gần mới tin, xa hơn → quăng về LLM.
SEMANTIC_CACHE_THRESHOLD = _S.understanding.semantic_cache_threshold


# --- Secrets & flags: CHỈ từ env (default an toàn cho dev — xem .env.example) ---

def _env_flag(name: str, default: bool) -> bool:
    """Flag bool: default từ settings.yaml, env "0"/"1" override."""
    return os.environ.get(name, "1" if default else "0") == "1"


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# DETERMINISTIC_MODE=1 (mặc định): không gọi LLM, planner rule-based — test/demo không cần key.
DETERMINISTIC_MODE = _env_flag("DETERMINISTIC_MODE", _S.features["deterministic_mode"])
# Typo correction bảo thủ, query-side. TẮT = export TASCO_TYPO_FIX=0.
# Xem luật trong src/understanding/typo_fix.py.
ENABLE_TYPO_FIX = _env_flag("TASCO_TYPO_FIX", _S.features["typo_fix"])


# Nguồn POI data: "xlsx" (mặc định — offline, không phá gì) | "postgres" (opt-in).
DATA_SOURCE = os.environ.get("DATA_SOURCE", "xlsx")


def database_url() -> str:
    """Connection string Postgres — secret, CHỈ từ env. Default khớp docker-compose
    dev (port 5433, user/pass tasco) — production PHẢI đặt DATABASE_URL thật."""
    return os.environ.get("DATABASE_URL", "postgresql://tasco:tasco@localhost:5433/tasco")


def aws_location_api_key() -> str:
    """API key AWS Location (Phase 4b) — secret, CHỈ từ env. TODO(production):
    chuyển sang IAM SigV4 — API key lộ qua URL/log/proxy."""
    return os.environ.get("AWS_LOCATION_API_KEY", "")


def aws_region() -> str:
    """Region AWS: env AWS_DEFAULT_REGION override, default từ settings.yaml."""
    return os.environ.get("AWS_DEFAULT_REGION") or settings().verify.aws_region


def bearer_token() -> str:
    """Secret auth service — đọc mỗi lần gọi (test set env sau import vẫn ăn)."""
    return os.environ.get("TASCO_BEARER_TOKEN", "")


def service_api_key() -> str:
    return os.environ.get("TASCO_API_KEY", "")
