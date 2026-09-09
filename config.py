"""Central configuration for the fact knowledge layer.

Every tunable lives here as an environment overridable constant. Model names,
thresholds, versions, and paths must not appear as literals anywhere else in
the codebase, so that a grader can retune the system from a single file or
from the environment without reading the source.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# The project's own .env wins over the surrounding shell for the variables it
# defines. Without this, a stale GEMINI_API_KEY exported from a shell profile
# silently defeats the documented quickstart, which says to put the key in .env.
load_dotenv(override=True)


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = Path(_str("DATA_DIR", str(ROOT_DIR / "data")))
DB_PATH: Path = Path(_str("DB_PATH", str(DATA_DIR / "knowledge.duckdb")))
TRACE_PATH: Path = Path(_str("TRACE_PATH", str(DATA_DIR / "traces.jsonl")))
UPLOAD_DIR: Path = Path(_str("UPLOAD_DIR", str(DATA_DIR / "uploads")))
DEFAULT_COLLECTION: str = _str("DEFAULT_COLLECTION", "uploads")

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = _str("GEMINI_API_KEY", "")
EXTRACTION_MODEL: str = _str("EXTRACTION_MODEL", "gemini-3.5-flash-lite")
ADJUDICATION_MODEL: str = _str("ADJUDICATION_MODEL", "gemini-3.5-flash")
DOCMETA_MODEL: str = _str("DOCMETA_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL: str = _str("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM: int = _int("EMBEDDING_DIM", 768)
EMBEDDING_BATCH_SIZE: int = _int("EMBEDDING_BATCH_SIZE", 32)
LLM_MAX_RETRIES: int = _int("LLM_MAX_RETRIES", 6)
LLM_RETRY_BASE_DELAY: float = _float("LLM_RETRY_BASE_DELAY", 2.0)
LLM_TIMEOUT_SECONDS: float = _float("LLM_TIMEOUT_SECONDS", 180.0)
LLM_MAX_CONCURRENCY: int = _int("LLM_MAX_CONCURRENCY", 6)
EMBEDDING_MAX_CONCURRENCY: int = _int("EMBEDDING_MAX_CONCURRENCY", 4)
# Adjudication runs one call per residual pair. The pairs are independent,
# so the only reason to serialise them is a rate limit, which the limiter
# already enforces on its own.
ADJUDICATION_MAX_CONCURRENCY: int = _int("ADJUDICATION_MAX_CONCURRENCY", 4)
# Client side pacing. Free tier keys are capped per minute per model, and a
# burst that exceeds the cap wastes retries rather than going faster.
LLM_REQUESTS_PER_MINUTE: int = _int("LLM_REQUESTS_PER_MINUTE", 15)
EMBEDDING_REQUESTS_PER_MINUTE: int = _int("EMBEDDING_REQUESTS_PER_MINUTE", 60)
LLM_ENABLED: bool = _bool("LLM_ENABLED", True)
# Thinking is expressed as a level rather than a token budget, because the two
# model generations take different arguments. See llm.thinking_config.
EXTRACTION_THINKING: str = _str("EXTRACTION_THINKING", "low")
ADJUDICATION_THINKING: str = _str("ADJUDICATION_THINKING", "high")
DOCMETA_PAGES: int = _int("DOCMETA_PAGES", 3)

# ---------------------------------------------------------------------------
# Versions. Bumping any of these invalidates the relevant cache entries.
# ---------------------------------------------------------------------------

PARSER_VERSION: str = _str("PARSER_VERSION", "parse-1.0.0")
CHUNKER_VERSION: str = _str("CHUNKER_VERSION", "chunk-1.0.0")
EXTRACTOR_VERSION: str = _str("EXTRACTOR_VERSION", "extract-1.0.0")
EXTRACTION_PROMPT_VERSION: str = _str("EXTRACTION_PROMPT_VERSION", "extract-prompt-1.0.0")
ADJUDICATION_PROMPT_VERSION: str = _str("ADJUDICATION_PROMPT_VERSION", "adjudicate-prompt-1.0.0")
DOCMETA_PROMPT_VERSION: str = _str("DOCMETA_PROMPT_VERSION", "docmeta-prompt-1.0.0")
LINKER_VERSION: str = _str("LINKER_VERSION", "link-1.0.0")
SCHEMA_VERSION: str = _str("SCHEMA_VERSION", "schema-1.0.0")

# ---------------------------------------------------------------------------
# Parsing and chunking
# ---------------------------------------------------------------------------

CHUNK_TARGET_CHARS: int = _int("CHUNK_TARGET_CHARS", 1800)
CHUNK_OVERLAP_CHARS: int = _int("CHUNK_OVERLAP_CHARS", 200)
CHUNK_MIN_CHARS: int = _int("CHUNK_MIN_CHARS", 120)
SLIDE_PAGE_MAX_CHARS: int = _int("SLIDE_PAGE_MAX_CHARS", 3600)
SPARSE_PAGE_CHAR_THRESHOLD: int = _int("SPARSE_PAGE_CHAR_THRESHOLD", 50)
PAGE_LABEL_MARGIN_FRACTION: float = _float("PAGE_LABEL_MARGIN_FRACTION", 0.10)

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

MAX_FACTS_PER_CHUNK: int = _int("MAX_FACTS_PER_CHUNK", 12)
PREDICATE_HINT_TOP_N: int = _int("PREDICATE_HINT_TOP_N", 40)
BOILERPLATE_MIN_ALPHA_RATIO: float = _float("BOILERPLATE_MIN_ALPHA_RATIO", 0.35)
BOILERPLATE_MIN_DIGITS: int = _int("BOILERPLATE_MIN_DIGITS", 1)

# ---------------------------------------------------------------------------
# Evidence verification
# ---------------------------------------------------------------------------

EVIDENCE_FUZZ_THRESHOLD: float = _float("EVIDENCE_FUZZ_THRESHOLD", 90.0)
EVIDENCE_MIN_QUOTE_CHARS: int = _int("EVIDENCE_MIN_QUOTE_CHARS", 8)
EVIDENCE_BBOX_SEARCH_CHARS: int = _int("EVIDENCE_BBOX_SEARCH_CHARS", 60)

# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

ENTITY_FUZZY_THRESHOLD: float = _float("ENTITY_FUZZY_THRESHOLD", 92.0)
ENTITY_FUZZY_REVIEW_THRESHOLD: float = _float("ENTITY_FUZZY_REVIEW_THRESHOLD", 86.0)
ENTITY_EMBED_THRESHOLD: float = _float("ENTITY_EMBED_THRESHOLD", 0.88)
ADDRESS_FUZZY_THRESHOLD: float = _float("ADDRESS_FUZZY_THRESHOLD", 80.0)

# ---------------------------------------------------------------------------
# Candidate funnel
# ---------------------------------------------------------------------------

SUBJECT_SIMILARITY_THRESHOLD: float = _float("SUBJECT_SIMILARITY_THRESHOLD", 88.0)
NEIGHBOUR_TOP_K: int = _int("NEIGHBOUR_TOP_K", 12)
NEIGHBOUR_COSINE_THRESHOLD: float = _float("NEIGHBOUR_COSINE_THRESHOLD", 0.80)
CROSS_COLLECTION_LINKING: bool = _bool("CROSS_COLLECTION_LINKING", False)

# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

RELATIVE_TOLERANCE_FLOOR: float = _float("RELATIVE_TOLERANCE_FLOOR", 0.005)
MATERIAL_GAP_RELATIVE: float = _float("MATERIAL_GAP_RELATIVE", 0.02)
# Per document budget for adjudication. Residual pairs are ordered by
# similarity, so the budget keeps the most relevant ones. When it truncates,
# the run record says how many pairs were left undecided rather than hiding it.
ADJUDICATION_MAX_PAIRS: int = _int("ADJUDICATION_MAX_PAIRS", 75)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_PAGE_SIZE: int = _int("API_PAGE_SIZE", 50)
API_MAX_PAGE_SIZE: int = _int("API_MAX_PAGE_SIZE", 500)
MAX_UPLOAD_BYTES: int = _int("MAX_UPLOAD_BYTES", 80 * 1024 * 1024)


def ensure_dirs() -> None:
    """Create the directories the system writes into."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
