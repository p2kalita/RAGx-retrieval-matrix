"""
Section 11: Chunk Quality Validation & Context Preservation — SOLUTION
======================================================================

This module executes formal validation tests on the selected `chunk_384_40`
corpus, auditing empty chunks, text duplication, consecutive chunk continuity,
and the preservation of statutory citations (Notebook Cells 172 to 187).
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    print_header,
    load_chunks,
)


def validate_empty_chunks(chunks_df: pd.DataFrame) -> int:
    """
    Count any empty or whitespace-only chunks in the dataset.
    """
    empty_count = (
        chunks_df["chunk_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    return int(empty_count)


def validate_duplicate_chunks(chunks_df: pd.DataFrame) -> int:
    """
    Count exact duplicate chunk_text entries across the corpus.
    """
    dup_count = chunks_df["chunk_text"].duplicated().sum()
    return int(dup_count)


def validate_consecutive_overlap(chunks_df: pd.DataFrame, judgment_id: str = None) -> Dict[str, Any]:
    """
    Verify consecutive chunk overlap on a judgment.
    """
    if judgment_id is None:
        judgment_id = chunks_df["judgment_id"].iloc[0]

    case_chunks = (
        chunks_df[chunks_df["judgment_id"] == judgment_id]
        .sort_values("local_chunk_id")
        .reset_index(drop=True)
    )

    if len(case_chunks) < 2:
        return {
            "judgment_id": judgment_id,
            "has_consecutive_chunks": False,
            "overlap_words": 0,
            "is_valid": True,
        }

    c1_words = case_chunks.iloc[0]["chunk_text"].split()
    c2_words = case_chunks.iloc[1]["chunk_text"].split()

    # Find longest common sequence between end of c1 and start of c2
    overlap_count = 0
    max_check = min(len(c1_words), len(c2_words), 100)
    for n in range(max_check, 0, -1):
        if c1_words[-n:] == c2_words[:n]:
            overlap_count = n
            break

    return {
        "judgment_id": judgment_id,
        "has_consecutive_chunks": True,
        "chunk1_word_count": len(c1_words),
        "chunk2_word_count": len(c2_words),
        "exact_overlap_words": overlap_count,
        "is_valid": bool(overlap_count > 0),
    }


def audit_legal_context_preservation(chunks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify the density of legal terminology, sections, and citations across chunks.
    """
    total = len(chunks_df)
    legal_pattern = re.compile(
        r"\b(section|article|act|order|court|judgment|appellant|respondent|crpc|ipc|cpc)\b",
        re.IGNORECASE,
    )

    matches = chunks_df["chunk_text"].apply(lambda t: bool(legal_pattern.search(str(t))))
    containing_legal = int(matches.sum())
    pct = round((containing_legal / total) * 100, 2)

    return {
        "total_chunks": total,
        "chunks_with_legal_markers": containing_legal,
        "pct_legal_context_density": pct,
        "is_valid": bool(pct > 95.0),
    }


def run_section_11() -> Dict[str, Any]:
    """Execute complete Section 11 pipeline."""
    print_header("Section 11: Chunk Quality Validation & Context Preservation")

    chunks_df = load_chunks()
    total_chunks = len(chunks_df)
    print(f"Auditing selected corpus: {total_chunks:,} chunks...")

    # 1. Empty Chunks
    empty_count = validate_empty_chunks(chunks_df)
    print("\n1. Empty Chunk Audit:")
    print(f"  - Empty Chunks Count   : {empty_count}")
    print(f"  - Empty Chunks Status  : {'PASSED (0 empty)' if empty_count == 0 else 'FAILED'}")

    # 2. Duplicate Chunks
    dup_count = validate_duplicate_chunks(chunks_df)
    print("\n2. Duplicate Chunk Audit:")
    print(f"  - Duplicate Chunks Count: {dup_count}")
    print(f"  - Duplicate Status      : {'PASSED (0 duplicates)' if dup_count == 0 else 'FAILED'}")

    # 3. Overlap Audit
    overlap_res = validate_consecutive_overlap(chunks_df)
    print("\n3. Consecutive Chunk Overlap Audit:")
    print(f"  - Audited Judgment ID   : {overlap_res['judgment_id']}")
    print(f"  - Chunk 1 Words         : {overlap_res['chunk1_word_count']}")
    print(f"  - Chunk 2 Words         : {overlap_res['chunk2_word_count']}")
    print(f"  - Overlap Words         : {overlap_res['exact_overlap_words']} words")
    print(f"  - Continuity Status     : {'PASSED (Valid boundary overlap)' if overlap_res['is_valid'] else 'WARNING'}")

    # 4. Legal Context Preservation
    context_res = audit_legal_context_preservation(chunks_df)
    print("\n4. Legal Context Preservation:")
    print(f"  - Chunks with Legal Terms: {context_res['chunks_with_legal_markers']:,} / {context_res['total_chunks']:,}")
    print(f"  - Legal Context Density : {context_res['pct_legal_context_density']}%")
    print(f"  - Context Status        : {'PASSED (>95% legal terms preserved)' if context_res['is_valid'] else 'FAILED'}")

    all_passed = (empty_count == 0) and (dup_count == 0) and context_res["is_valid"]
    print(f"\nOverall Section 11 Chunk Validation: {'ALL TESTS PASSED' if all_passed else 'FAILED'}")

    assert all_passed, "Validation tests failed on chunk corpus"
    return {
        "empty_count": empty_count,
        "dup_count": dup_count,
        "overlap_res": overlap_res,
        "context_res": context_res,
    }


if __name__ == "__main__":
    run_section_11()
