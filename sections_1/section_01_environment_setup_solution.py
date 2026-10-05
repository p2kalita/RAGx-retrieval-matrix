"""
Section 01: Environment Setup and Project Configuration — SOLUTION
==================================================================

This module establishes the core directories, logging, and reproducibility
parameters for the NYAYaRAG data processing and indexing pipeline, replicating
Notebook Cells 0 to 6.
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    PROJECT_DIR,
    SEED,
    ensure_directories,
    print_header,
)


def set_reproducibility_seed(seed: int = SEED) -> None:
    """
    Set seeds across standard library random, numpy, and hash seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global random seed set to: {seed}")


def initialize_workspace_directories(base_dir: Path = PROJECT_DIR) -> Dict[str, Path]:
    """
    Construct, create, and verify all required directories.
    """
    dirs = {
        "project": base_dir,
        "raw_pdf": base_dir / "data" / "raw" / "pdf" / "year=2024" / "english",
        "raw_metadata": base_dir / "data" / "raw" / "metadata" / "year=2024",
        "processed": base_dir / "data" / "processed",
        "processed_text": base_dir / "data" / "processed" / "text" / "year=2024" / "english",
        "models": base_dir / "models",
        "tfidf": base_dir / "models" / "tfidf",
        "bm25": base_dir / "models" / "bm25",
        "e5": base_dir / "models" / "e5",
        "bge": base_dir / "models" / "bge",
        "inlegalbert": base_dir / "models" / "inlegalbert",
        "results": base_dir / "results",
    }
    for name, p in dirs.items():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def get_environment_info() -> Dict[str, str]:
    """
    Collect versions of core runtime libraries.
    """
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for mod_name in ["pymupdf", "sklearn", "sentence_transformers", "chromadb", "rank_bm25"]:
        try:
            m = __import__(mod_name)
            info[mod_name] = getattr(m, "__version__", "available")
        except ImportError:
            info[mod_name] = "not installed"
    return info


def run_section_01() -> Dict[str, Any]:
    """Execute complete Section 01 pipeline."""
    print_header("Section 1: Environment Setup and Project Configuration")
    set_reproducibility_seed(SEED)

    dirs = initialize_workspace_directories(PROJECT_DIR)
    print("\nVerified Project Directory Layout:")
    for key, path in dirs.items():
        exists_mark = "OK" if path.exists() else "MISSING"
        print(f"  - {key:<16}: [{exists_mark}] {path}")

    env_info = get_environment_info()
    print("\nEnvironment & Dependency Versions:")
    for lib, ver in env_info.items():
        print(f"  - {lib:<22}: {ver}")

    print("\n Section 1 initialization completed successfully.")
    return {"directories": dirs, "environment": env_info}


if __name__ == "__main__":
    run_section_01()
