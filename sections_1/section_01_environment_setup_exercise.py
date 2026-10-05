"""
Section 01: Environment Setup and Project Configuration — EXERCISE
==================================================================

Overview:
---------
In this section, you will set up the runtime environment and foundational directory
hierarchy for the NYAYaRAG pipeline. Reproducibility across runs requires fixing
random seeds and ensuring that all directory artifacts exist before pipeline execution.

Tasks:
------
1. Implement `set_reproducibility_seed(seed: int)` to set seeds for `random`, `numpy`,
   and python hash seed.
2. Implement `initialize_workspace_directories(base_dir: Path)` to verify or create
   the directory layout:
     - data/raw/pdf/year=2024/english
     - data/raw/metadata/year=2024
     - data/processed/text/year=2024/english
     - models/tfidf
     - models/bm25
     - models/e5
     - models/bge
     - models/inlegalbert
     - results
3. Implement `get_environment_info()` to report versions of core packages (numpy, pandas).

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import PROJECT_DIR, SEED, print_header


def set_reproducibility_seed(seed: int = SEED) -> None:
    """
    Set seeds for python's built-in `random` module, `numpy`, and the PYTHONHASHSEED
    environment variable to guarantee deterministic behavior.

    Args:
        seed: The integer seed value (default: 42).
    """
    # TODO: Implement seed configuration for random, numpy, and PYTHONHASHSEED
    raise NotImplementedError("Implement set_reproducibility_seed")


def initialize_workspace_directories(base_dir: Path = PROJECT_DIR) -> Dict[str, Path]:
    """
    Construct and verify the presence of all required NYAYaRAG project directories.

    Args:
        base_dir: The root path of the project.

    Returns:
        Dict[str, Path]: Mapping of directory identifiers to their Path objects.
    """
    # TODO: Define and create the required directory paths:
    # 'raw_pdf', 'raw_metadata', 'processed_text', 'tfidf', 'bm25', 'e5', 'bge', 'inlegalbert', 'results'
    # Use path.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Implement initialize_workspace_directories")


def get_environment_info() -> Dict[str, str]:
    """
    Collect versions of the core runtime libraries.

    Returns:
        Dict[str, str]: Mapping of package name to version string.
    """
    # TODO: Return a dictionary with 'python', 'numpy', and 'pandas' versions
    raise NotImplementedError("Implement get_environment_info")


if __name__ == "__main__":
    print_header("Section 01: Environment Setup — Exercise Validation")
    try:
        set_reproducibility_seed(42)
        print(" [TODO] set_reproducibility_seed executed.")

        dirs = initialize_workspace_directories(PROJECT_DIR)
        print(f" [TODO] Directories initialized: {len(dirs)} paths.")
        assert "processed_text" in dirs, "processed_text must be in directories dict"
        assert dirs["processed_text"].exists(), "processed_text path must exist"

        env = get_environment_info()
        print(f" [TODO] Environment info: {env}")
        assert "python" in env and "pandas" in env, "Environment info must include python and pandas"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
