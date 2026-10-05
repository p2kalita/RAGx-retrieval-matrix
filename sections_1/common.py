"""
NYAYaRAG Sections Common Utilities and Path Configurations
==========================================================
Provides centralized, portable directory resolution and shared utility
functions across all 17 exercise and solution modules.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Portable resolution of NyayaRAG Project Root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

# Core Project Directories
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PDF_DIR = RAW_DATA_DIR / "pdf" / "year=2024" / "english"
METADATA_DIR = RAW_DATA_DIR / "metadata" / "year=2024"
METADATA_FILE = METADATA_DIR / "metadata_2024.parquet"

PROCESSED_DIR = DATA_DIR / "processed"
TEXT_DIR = PROCESSED_DIR / "text" / "year=2024" / "english"
EXTRACTION_LOG_FILE = PROCESSED_DIR / "extraction_log.csv"
CLEAN_CORPUS_FILE = PROCESSED_DIR / "clean_corpus.parquet"
PROCESSED_METADATA_FILE = PROCESSED_DIR / "metadata_2024_processed.parquet"
CHUNKS_FILE = PROCESSED_DIR / "chunks_384_40.parquet"

MODELS_DIR = PROJECT_DIR / "models"
TFIDF_DIR = MODELS_DIR / "tfidf"
BM25_DIR = MODELS_DIR / "bm25"
E5_DIR = MODELS_DIR / "e5"
BGE_DIR = MODELS_DIR / "bge"
INLEGALBERT_DIR = MODELS_DIR / "inlegalbert"

RESULTS_DIR = PROJECT_DIR / "results"
INDEX_METADATA_FILE = MODELS_DIR / "retrieval_index_metadata.json"

# Global Seed
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Set random seed across python standard library and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directories() -> Dict[str, Path]:
    """
    Ensure all necessary workspace directories exist.
    Returns a dictionary of all directory paths.
    """
    dirs = {
        "project": PROJECT_DIR,
        "raw_data": RAW_DATA_DIR,
        "pdf": PDF_DIR,
        "metadata": METADATA_DIR,
        "processed": PROCESSED_DIR,
        "text": TEXT_DIR,
        "models": MODELS_DIR,
        "tfidf": TFIDF_DIR,
        "bm25": BM25_DIR,
        "e5": E5_DIR,
        "bge": BGE_DIR,
        "inlegalbert": INLEGALBERT_DIR,
        "results": RESULTS_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def print_header(title: str, width: int = 70) -> None:
    """Prints a standardized section title header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def load_raw_metadata(metadata_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw 2024 Supreme Court metadata parquet file."""
    path = metadata_path or METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_parquet(path)


def load_processed_corpus() -> pd.DataFrame:
    """
    Load the processed corpus parquet file. Falls back to processed metadata
    or raw metadata if necessary.
    """
    if CLEAN_CORPUS_FILE.exists():
        return pd.read_parquet(CLEAN_CORPUS_FILE)
    elif PROCESSED_METADATA_FILE.exists():
        return pd.read_parquet(PROCESSED_METADATA_FILE)
    else:
        raise FileNotFoundError(
            f"Processed corpus not found at {CLEAN_CORPUS_FILE} or {PROCESSED_METADATA_FILE}"
        )


def load_chunks() -> pd.DataFrame:
    """Load the selected chunk corpus (chunk_384_40)."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")
    return pd.read_parquet(CHUNKS_FILE)
