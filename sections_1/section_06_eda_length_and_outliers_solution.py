"""
Section 06: Exploratory Data Analysis — Corpus Length and Outliers — SOLUTION
============================================================================

This module conducts statistical distribution analysis on judgment lengths,
calculates IQR thresholds to detect length outliers, and identifies extreme
cases in the 2024 Supreme Court dataset (Notebook Cells 59 to 74).
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    RESULTS_DIR,
    print_header,
    load_processed_corpus,
)


def compute_corpus_length_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute statistical distribution metrics of judgment word lengths.
    """
    if "clean_words" in df.columns:
        words = df["clean_words"]
    else:
        words = df["clean_text"].astype(str).str.split().str.len()

    return {
        "total_judgments": int(len(df)),
        "total_words": int(words.sum()),
        "mean_words": float(words.mean()),
        "median_words": float(words.median()),
        "std_words": float(words.std()),
        "min_words": int(words.min()),
        "max_words": int(words.max()),
    }


def detect_length_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify judgments with extreme lengths using the 1.5 * IQR rule.
    """
    df = df.copy()
    if "clean_words" not in df.columns:
        df["clean_words"] = df["clean_text"].astype(str).str.split().str.len()

    q1 = float(df["clean_words"].quantile(0.25))
    q3 = float(df["clean_words"].quantile(0.75))
    iqr = q3 - q1

    upper_bound = q3 + 1.5 * iqr
    lower_bound = max(0.0, q1 - 1.5 * iqr)

    outliers_df = df[(df["clean_words"] > upper_bound) | (df["clean_words"] < lower_bound)]

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "outlier_count": len(outliers_df),
        "outlier_df": outliers_df,
    }


def identify_extreme_cases(df: pd.DataFrame, n: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Extract the top n longest and top n shortest judgments.
    """
    df = df.copy()
    if "clean_words" not in df.columns:
        df["clean_words"] = df["clean_text"].astype(str).str.split().str.len()

    cols = [c for c in ["title", "citation", "case_id", "decision_date", "clean_words"] if c in df.columns]

    longest = df.sort_values("clean_words", ascending=False)[cols].head(n)
    shortest = df.sort_values("clean_words", ascending=True)[cols].head(n)

    return {"longest": longest, "shortest": shortest}


def plot_length_distributions(df: pd.DataFrame, output_dir: Path = RESULTS_DIR) -> Path:
    """
    Generate and save judgment length histogram and boxplot visualization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "eda_length_distribution.png"

    words = df["clean_words"] if "clean_words" in df.columns else df["clean_text"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Histogram
    axes[0].hist(words, bins=40, edgecolor="black", color="#1f77b4", alpha=0.7)
    axes[0].axvline(words.mean(), color="red", linestyle="--", label=f"Mean: {words.mean():,.0f}")
    axes[0].axvline(words.median(), color="green", linestyle=":", label=f"Median: {words.median():,.0f}")
    axes[0].set_title("Judgment Length Distribution (Words)")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Boxplot
    axes[1].boxplot(words, vert=False, patch_artist=True, boxprops=dict(facecolor="#aec7e8"))
    axes[1].set_title("Judgment Length Boxplot (Outliers)")
    axes[1].set_xlabel("Word Count")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def run_section_06(save_plot: bool = True) -> Dict[str, Any]:
    """Execute complete Section 06 pipeline."""
    print_header("Section 6: EDA — Corpus Length and Outliers")

    corpus_df = load_processed_corpus()
    stats = compute_corpus_length_summary(corpus_df)

    print("Corpus-wide Word Length Summary:")
    print(f"  - Total Judgments : {stats['total_judgments']:,}")
    print(f"  - Total Words     : {stats['total_words']:,}")
    print(f"  - Mean Words/Doc  : {stats['mean_words']:,.2f}")
    print(f"  - Median Words/Doc: {stats['median_words']:,.2f}")
    print(f"  - Std Dev Words   : {stats['std_words']:,.2f}")
    print(f"  - Min Words       : {stats['min_words']:,}")
    print(f"  - Max Words       : {stats['max_words']:,}")

    outliers = detect_length_outliers(corpus_df)
    print("\nOutlier Detection (1.5 * IQR Rule):")
    print(f"  - 25th Percentile (Q1) : {outliers['q1']:,.1f} words")
    print(f"  - 75th Percentile (Q3) : {outliers['q3']:,.1f} words")
    print(f"  - IQR                  : {outliers['iqr']:,.1f} words")
    print(f"  - Upper Outlier Bound  : {outliers['upper_bound']:,.1f} words")
    print(f"  - Detected Outliers    : {outliers['outlier_count']} judgments")

    extremes = identify_extreme_cases(corpus_df, n=3)
    print("\nTop 3 Longest Judgments:")
    print(extremes["longest"].to_string(index=False))

    print("\nTop 3 Shortest Judgments:")
    print(extremes["shortest"].to_string(index=False))

    if save_plot:
        fig_path = plot_length_distributions(corpus_df)
        print(f"\nSaved visualization to: {fig_path}")

    print("\n Section 6 EDA length and outlier analysis completed successfully.")
    return {"stats": stats, "outliers": outliers, "extremes": extremes}


if __name__ == "__main__":
    run_section_06()
