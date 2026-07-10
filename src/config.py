"""Cấu hình tập trung: đường dẫn data/lexicon, tên model embedding, ngưỡng semantic cache.

Không hardcode credentials — đọc API key từ biến môi trường. Có cờ DETERMINISTIC_MODE
(không cần API key): planner rơi về rule-based cho test/demo.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Data & lexicon ---
DATA_XLSX = ROOT / "data" / "ai_maps_track2_dataset_participants.xlsx"
LEXICON_DIR = ROOT / "lexicon"
ATTRIBUTE_CONCEPTS_YAML = LEXICON_DIR / "attribute_concepts.yaml"
GAZETTEER_YAML = LEXICON_DIR / "gazetteer.yaml"
CATEGORIES_YAML = LEXICON_DIR / "categories.yaml"

# Tên sheet trong xlsx (đã verify bằng eval/verify_dataset.py)
SHEET_POI = "POI_Dataset"
SHEET_EVAL = "Public_Evaluation"
SHEET_TAXONOMY = "Attribute_Taxonomy"
SHEET_SIGNALS = "Ranking_Signals"
SHEET_README = "README"

# --- Eval ---
REPORTS_DIR = ROOT / "eval" / "reports"
README_MD = ROOT / "README.md"
EVAL_TOP_K = 10  # số kết quả retriever trả cho eval (MRR tính trong top-k)

# --- L2 dense ---
# e5-small đã đo retrieval-only 0.950 trên corpus 111 doc. TODO thử e5-base/bge-m3
# như ablation NHƯNG phải đo lại — corpus nhỏ, không mặc định model to hơn là tốt hơn.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_CACHE_DIR = ROOT / "data" / "cache"  # .npy cache — gitignore, xoá là tự build lại

# --- L1 LLM planner (chưa dùng ở slice BM25) ---
# Ngưỡng cosine cho semantic cache: đủ gần mới tin, xa hơn → quăng về LLM.
SEMANTIC_CACHE_THRESHOLD = 0.92
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# DETERMINISTIC_MODE=1 (mặc định): không gọi LLM, planner rule-based — test/demo không cần key.
DETERMINISTIC_MODE = os.environ.get("DETERMINISTIC_MODE", "1") == "1"
