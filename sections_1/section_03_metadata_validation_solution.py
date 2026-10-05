"""
Section 03: Metadata Quality Assessment and Deduplication — SOLUTION
====================================================================

This module performs comprehensive metadata quality assessment, checks for
missing values and blank strings, investigates row and citation uniqueness,
and standardizes temporal fields (Notebook Cells 16 to 29).
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    METADATA_FILE,
    print_header,
    load_raw_metadata,
)


def assess_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assess metadata completeness by counting both null values and whitespace-only strings.
    """
    blank_count = df.apply(
        lambda col: col.astype(str).str.strip().eq("").sum()
    )
    null_count = df.isna().sum()
    total_missing = blank_count + null_count
    missing_pct = (total_missing / len(df) * 100).round(2)

    quality_df = pd.DataFrame({
        "Null Count": null_count,
        "Blank Count": blank_count,
        "Total Missing": total_missing,
        "Missing %": missing_pct,
    })
    return quality_df


def check_duplicates(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for duplicate rows, duplicate citations, and duplicate case IDs.
    """
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_citations": int(df["citation"].duplicated().sum()),
        "duplicate_case_ids": int(df["case_id"].duplicated().sum()),
        "duplicate_cnr": int(df["cnr"].duplicated().sum()),
    }


def standardize_decision_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'decision_date' column to datetime and report range.
    """
    df = df.copy()
    df["decision_date"] = pd.to_datetime(df["decision_date"], format="%d-%m-%Y", errors="coerce")
    null_dates = df["decision_date"].isna().sum()
    if null_dates > 0:
        # Fallback to general parsing if format differs
        df["decision_date"] = pd.to_datetime(df["decision_date"], dayfirst=True, errors="coerce")
        null_dates = df["decision_date"].isna().sum()
    if null_dates > 0:
        raise ValueError(f"Found {null_dates} unparseable decision dates!")
    return df


def run_section_03() -> Dict[str, Any]:
    """Execute complete Section 03 pipeline."""
    print_header("Section 3: Metadata Quality Assessment")

    raw_df = load_raw_metadata(METADATA_FILE)
    print(f"Total metadata records : {len(raw_df):,}")
    print(f"Total attributes/cols  : {len(raw_df.columns)}")

    print("\n1. Duplicate Analysis:")
    dup_stats = check_duplicates(raw_df)
    for k, v in dup_stats.items():
        print(f"  - {k:<22}: {v}")

    print("\n2. Missing Value Analysis (Top 8 attributes with missing values):")
    quality_df = assess_missing_values(raw_df)
    print(quality_df.sort_values("Total Missing", ascending=False).head(8))

    print("\n3. Temporal Standardization:")
    clean_df = standardize_decision_dates(raw_df)
    min_date = clean_df["decision_date"].min()
    max_date = clean_df["decision_date"].max()
    print(f"  - Date Range           : {min_date.date()} to {max_date.date()}")
    print(f"  - Verified 100% 2024   : {(clean_df['decision_date'].dt.year == 2024).all()}")

    print("\n4. Court Distribution:")
    print(clean_df["court"].value_counts().to_string())

    print("\n Section 3 metadata validation completed successfully.")
    return {"cleaned_metadata_df": clean_df, "quality_df": quality_df, "dup_stats": dup_stats}


if __name__ == "__main__":
    run_section_03()
