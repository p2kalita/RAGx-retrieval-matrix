"""
Section 06: Exploratory Data Analysis — Corpus Length and Outliers — EXERCISE
============================================================================

Overview:
---------
Before chunking and indexing, we must understand the statistical length distribution
of the legal corpus. Extreme variance between short procedural orders and massive
constitution bench judgments informs chunking window sizes and retrieval strategies.

Tasks:
------
1. Implement `compute_corpus_length_summary(df: pd.DataFrame) -> Dict[str, float]`
   to calculate total documents, total words, mean, median, standard deviation, min, and max.
2. Implement `detect_length_outliers(df: pd.DataFrame) -> Dict[str, Any]` using the
   Interquartile Range (IQR) rule:
     - Q1 = 25th percentile, Q3 = 75th percentile
     - IQR = Q3 - Q1
     - Upper threshold = Q3 + 1.5 * IQR
     - Lower threshold = max(0, Q1 - 1.5 * IQR)
3. Implement `identify_extreme_cases(df: pd.DataFrame, n: int = 5) -> Dict[str, pd.DataFrame]`
   returning the n shortest and n longest judgments.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import print_header, load_processed_corpus


def compute_corpus_length_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute statistical distribution metrics of judgment word lengths.

    Args:
        df: DataFrame containing a 'clean_text' or 'clean_words' column.

    Returns:
        Dict[str, float]: Dictionary with keys:
            - 'total_judgments'
            - 'total_words'
            - 'mean_words'
            - 'median_words'
            - 'std_words'
            - 'min_words'
            - 'max_words'
    """
    # TODO: Calculate word counts for clean text if 'clean_words' is not present
    # Compute descriptive statistics using pandas or numpy
    raise NotImplementedError("Implement compute_corpus_length_summary")


def detect_length_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify judgments with extreme lengths using the 1.5 * IQR rule.

    Args:
        df: DataFrame containing word counts.

    Returns:
        Dict[str, Any]:
            - 'q1': float
            - 'q3': float
            - 'iqr': float
            - 'upper_bound': float
            - 'lower_bound': float
            - 'outlier_count': int
            - 'outlier_df': pd.DataFrame
    """
    # TODO: Compute Q1, Q3, IQR, upper_bound, and lower_bound
    # Filter df where word_count > upper_bound or word_count < lower_bound
    raise NotImplementedError("Implement detect_length_outliers")


def identify_extreme_cases(df: pd.DataFrame, n: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Extract the top n longest and top n shortest judgments.

    Args:
        df: DataFrame containing 'title', 'citation', and word counts.
        n: Number of cases to return per extreme.

    Returns:
        Dict[str, pd.DataFrame]: Dict with keys 'longest' and 'shortest'.
    """
    # TODO: Sort by word count ascending/descending and return top n records
    raise NotImplementedError("Implement identify_extreme_cases")


if __name__ == "__main__":
    print_header("Section 06: EDA Length and Outliers — Exercise Validation")
    try:
        corpus_df = load_processed_corpus()
        print(f"Loaded corpus: {len(corpus_df)} records.")

        summary = compute_corpus_length_summary(corpus_df)
        print(f" [TODO] Computed length summary: mean={summary['mean_words']:.1f}, median={summary['median_words']:.1f}")
        assert summary["total_judgments"] == 782, "Expected 782 judgments"
        assert summary["total_words"] > 5_000_000, "Expected > 5 million words"

        outliers = detect_length_outliers(corpus_df)
        print(f" [TODO] Detected {outliers['outlier_count']} length outliers (Upper: {outliers['upper_bound']:.1f})")

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
