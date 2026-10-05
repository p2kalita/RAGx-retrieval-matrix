"""
Section 09: Editorial Front Matter Detection and Removal — SOLUTION
==================================================================

This module isolates judicial opinions from publisher-generated front matter
(headnotes, SCR volume headers, advocate lists) by scanning for official
Supreme Court judgment markers (Notebook Cells 139 to 158).
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    print_header,
    load_processed_corpus,
)

PRIMARY_MARKER = "Judgment / Order of the Supreme Court"
FALLBACK_MARKERS = [
    "J U D G M E N T",
    "JUDGMENT",
    "O R D E R",
    "ORDER",
]


def remove_editorial_front_matter(text: str, marker: str = PRIMARY_MARKER) -> str:
    """
    Remove publisher-generated front matter and retain only the judicial reasoning.
    """
    text_str = str(text)

    # 1. Search for primary marker
    pos = text_str.find(marker)
    if pos != -1:
        return text_str[pos:].strip()

    # 2. Check fallback markers if primary not present
    for fb in FALLBACK_MARKERS:
        fb_pos = text_str.find(fb)
        if 0 < fb_pos < 10000:  # Only if within reasonable front matter range
            return text_str[fb_pos:].strip()

    return text_str.strip()


def apply_front_matter_removal(df: pd.DataFrame, marker: str = PRIMARY_MARKER) -> pd.DataFrame:
    """
    Apply front matter removal across the corpus and track statistics.
    """
    df = df.copy()
    df["judgment_text"] = df["clean_text"].apply(lambda t: remove_editorial_front_matter(t, marker))

    orig_len = df["clean_text"].astype(str).str.len()
    clean_len = df["judgment_text"].astype(str).str.len()
    df["front_matter_chars_removed"] = orig_len - clean_len

    return df


def validate_front_matter_removal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that no texts are empty and count how many judgments had front matter removed.
    """
    total = len(df)
    missing = int(df["judgment_text"].isna().sum())
    empty = int((df["judgment_text"].astype(str).str.strip() == "").sum())

    modified_mask = df["front_matter_chars_removed"] > 0
    num_modified = int(modified_mask.sum())
    avg_removed = float(df.loc[modified_mask, "front_matter_chars_removed"].mean()) if num_modified else 0.0

    return {
        "total_judgments": total,
        "missing_count": missing,
        "empty_count": empty,
        "judgments_with_front_matter_removed": num_modified,
        "pct_judgments_modified": round((num_modified / total) * 100, 2),
        "avg_chars_removed_when_present": round(avg_removed, 1),
        "is_valid": bool(missing == 0 and empty == 0),
    }


def run_section_09() -> Dict[str, Any]:
    """Execute complete Section 09 pipeline."""
    print_header("Section 9: Editorial Front Matter Detection & Removal")

    corpus_df = load_processed_corpus()
    print(f"Loaded {len(corpus_df):,} judgments. Applying front matter detector...")

    cleaned_df = apply_front_matter_removal(corpus_df)
    val = validate_front_matter_removal(cleaned_df)

    print("\nFront Matter Removal Validation:")
    print(f"  - Total Judgments Processed       : {val['total_judgments']:,}")
    print(f"  - Missing Judgments               : {val['missing_count']}")
    print(f"  - Empty Judgments                 : {val['empty_count']}")
    print(f"  - Judgments with Front Matter Cut : {val['judgments_with_front_matter_removed']} ({val['pct_judgments_modified']}%)")
    print(f"  - Avg Chars Removed per Modified  : {val['avg_chars_removed_when_present']:,.1f}")
    print(f"  - Validation Status               : {'PASSED (Zero Data Loss)' if val['is_valid'] else 'FAILED'}")

    sample_orig = len(corpus_df.iloc[0]["clean_text"])
    sample_cut = len(cleaned_df.iloc[0]["judgment_text"])
    print(f"\nSample Document Inspection (Judgment 0):")
    print(f"  - Cleaned text length     : {sample_orig:,} chars")
    print(f"  - Judicial reasoning only : {sample_cut:,} chars")
    print(f"  - Front matter removed    : {sample_orig - sample_cut:,} chars")

    assert val["is_valid"], "Validation failed: empty texts created during front matter removal"
    print("\n Section 9 front matter removal completed successfully.")
    return {"cleaned_df": cleaned_df, "validation": val}


if __name__ == "__main__":
    run_section_09()
