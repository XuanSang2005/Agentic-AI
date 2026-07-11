from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
PipelineMode = Literal["rules_only", "rules_plus_hf", "rules_plus_llm", "full_hybrid"]


def _load_dotenv() -> None:
    """Load the project .env without adding a core runtime dependency."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _boolean(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).casefold() in {"1", "true", "yes", "on"}


def _confidence(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = ROOT / "src" / "tasco_query" / "raw_data"
    lexicon_dir: Path = ROOT / "src" / "tasco_query" / "lexicon_data"
    max_query_length: int = 512
    max_variants: int = 8
    optimizations_enabled: bool = True
    response_cache_enabled: bool = True
    response_cache_size: int = 256
    policy_version: str = "optimization-v1"
    trace_max_variants: int = 16
    trace_max_evidence: int = 64
    trace_max_candidates: int = 16
    mode: PipelineMode = "rules_only"
    hf_enabled: bool = False
    hf_model_id: str = "yammdd/vietnamese-error-correction"
    hf_revision: str = "4c5a5531e1a0cc2967ac3bd255cf2799ee10d8c1"
    hf_device: str = "cpu"
    hf_timeout_seconds: float = 8.0
    hf_batch_size: int = 4
    hf_max_input_tokens: int = 256
    hf_max_output_tokens: int = 256
    hf_local_files_only: bool = False
    llm_enabled: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-5-mini"
    llm_timeout_seconds: float = 8.0
    llm_max_output_tokens: int = 2000
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    semantic_implication_min_confidence: float = 0.5
    semantic_llm_acceptance_threshold: float = 0.8
    social_gate_confidence_threshold: float = 0.7
    social_gate_exclude_objective_only: bool = True
    social_gate_exclude_exact_queries: bool = True


def get_settings() -> Settings:
    _load_dotenv()
    raw_dir = os.getenv("TASCO_QUERY_DATA_DIR")
    data_dir = Path(raw_dir) if raw_dir else ROOT / "src" / "tasco_query" / "raw_data"
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    raw_lexicon_dir = os.getenv("TASCO_QUERY_LEXICON_DIR")
    lexicon_dir = Path(raw_lexicon_dir) if raw_lexicon_dir else ROOT / "src" / "tasco_query" / "lexicon_data"
    if not lexicon_dir.is_absolute():
        lexicon_dir = ROOT / lexicon_dir
    raw_mode = os.getenv("TASCO_QUERY_MODE", "rules_only")
    modes = {"rules_only", "rules_plus_hf", "rules_plus_llm", "full_hybrid"}
    if raw_mode not in modes:
        raise ValueError(f"unsupported TASCO_QUERY_MODE: {raw_mode}")
    return Settings(
        data_dir=data_dir,
        lexicon_dir=lexicon_dir,
        max_query_length=int(os.getenv("TASCO_QUERY_MAX_LENGTH", "512")),
        max_variants=int(os.getenv("TASCO_QUERY_MAX_VARIANTS", "8")),
        optimizations_enabled=_boolean("TASCO_QUERY_OPTIMIZATIONS_ENABLED", True),
        response_cache_enabled=_boolean("TASCO_QUERY_RESPONSE_CACHE_ENABLED", True),
        response_cache_size=max(1, int(os.getenv("TASCO_QUERY_RESPONSE_CACHE_SIZE", "256"))),
        policy_version=os.getenv("TASCO_QUERY_POLICY_VERSION", "optimization-v1"),
        trace_max_variants=int(os.getenv("TASCO_QUERY_TRACE_MAX_VARIANTS", "16")),
        trace_max_evidence=int(os.getenv("TASCO_QUERY_TRACE_MAX_EVIDENCE", "64")),
        trace_max_candidates=int(os.getenv("TASCO_QUERY_TRACE_MAX_CANDIDATES", "16")),
        mode=raw_mode,  # type: ignore[arg-type]
        hf_enabled=_boolean("TASCO_QUERY_HF_ENABLED"),
        hf_model_id=os.getenv("TASCO_QUERY_HF_MODEL_ID", "yammdd/vietnamese-error-correction"),
        hf_revision=os.getenv(
            "TASCO_QUERY_HF_REVISION", "4c5a5531e1a0cc2967ac3bd255cf2799ee10d8c1"
        ),
        hf_device=os.getenv("TASCO_QUERY_HF_DEVICE", "cpu"),
        hf_timeout_seconds=float(os.getenv("TASCO_QUERY_HF_TIMEOUT_SECONDS", "8")),
        hf_batch_size=int(os.getenv("TASCO_QUERY_HF_BATCH_SIZE", "4")),
        hf_max_input_tokens=int(os.getenv("TASCO_QUERY_HF_MAX_INPUT_TOKENS", "256")),
        hf_max_output_tokens=int(os.getenv("TASCO_QUERY_HF_MAX_OUTPUT_TOKENS", "256")),
        hf_local_files_only=_boolean("TASCO_QUERY_HF_LOCAL_FILES_ONLY"),
        llm_enabled=_boolean("TASCO_QUERY_LLM_ENABLED"),
        llm_provider=os.getenv("TASCO_QUERY_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        llm_timeout_seconds=float(os.getenv("TASCO_QUERY_LLM_TIMEOUT_SECONDS", "8")),
        llm_max_output_tokens=int(os.getenv("TASCO_QUERY_LLM_MAX_OUTPUT_TOKENS", "2000")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        semantic_implication_min_confidence=_confidence(
            "TASCO_QUERY_SEMANTIC_IMPLICATION_MIN_CONFIDENCE", 0.5
        ),
        semantic_llm_acceptance_threshold=_confidence(
            "TASCO_QUERY_SEMANTIC_LLM_ACCEPTANCE_THRESHOLD", 0.8
        ),
        social_gate_confidence_threshold=_confidence(
            "TASCO_QUERY_SOCIAL_GATE_CONFIDENCE_THRESHOLD", 0.7
        ),
        social_gate_exclude_objective_only=_boolean(
            "TASCO_QUERY_SOCIAL_GATE_EXCLUDE_OBJECTIVE_ONLY", True
        ),
        social_gate_exclude_exact_queries=_boolean(
            "TASCO_QUERY_SOCIAL_GATE_EXCLUDE_EXACT_QUERIES", True
        ),
    )
