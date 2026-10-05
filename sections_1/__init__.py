"""
NYAYaRAG Notebook 1 Modular Curriculum: Sections Package
--------------------------------------------------------
This package contains 17 modular sections dividing the data preparation
and retrieval indexing pipeline of NYAYaRAG into standalone exercise and
solution scripts.
"""

from .common import (
    PROJECT_DIR,
    RAW_DATA_DIR,
    PDF_DIR,
    METADATA_DIR,
    METADATA_FILE,
    PROCESSED_DIR,
    TEXT_DIR,
    CHUNKS_FILE,
    CLEAN_CORPUS_FILE,
    PROCESSED_METADATA_FILE,
    MODELS_DIR,
    TFIDF_DIR,
    BM25_DIR,
    E5_DIR,
    BGE_DIR,
    INLEGALBERT_DIR,
    RESULTS_DIR,
    SEED,
    set_seed,
    ensure_directories,
)

__all__ = [
    "PROJECT_DIR",
    "RAW_DATA_DIR",
    "PDF_DIR",
    "METADATA_DIR",
    "METADATA_FILE",
    "PROCESSED_DIR",
    "TEXT_DIR",
    "CHUNKS_FILE",
    "CLEAN_CORPUS_FILE",
    "PROCESSED_METADATA_FILE",
    "MODELS_DIR",
    "TFIDF_DIR",
    "BM25_DIR",
    "E5_DIR",
    "BGE_DIR",
    "INLEGALBERT_DIR",
    "RESULTS_DIR",
    "SEED",
    "set_seed",
    "ensure_directories",
]
