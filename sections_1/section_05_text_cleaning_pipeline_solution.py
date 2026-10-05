"""
Section 05: Text Cleaning and Preprocessing Pipeline — SOLUTION
==============================================================

This module loads the extracted judgment text files, standardizes formatting
artifacts while preserving legal citations and paragraph breaks, and validates
the clean corpus (Notebook Cells 44 to 58).
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    TEXT_DIR,
    CLEAN_CORPUS_FILE,
    print_header,
    load_raw_metadata,
)


def clean_legal_text(text: str) -> str:
    """
    Standardize extracted legal text while preserving legal meaning,
    paragraph boundaries, citations, statutory references, and punctuation.
    """
    if not isinstance(text, str):
        return ""

    # 1. Standardise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove tabs
    text = text.replace("\t", " ")

    # 3. Remove leading and trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 4. Preserve paragraph boundaries temporarily
    text = re.sub(r"\n{2,}", " <PARAGRAPH_BREAK> ", text)

    # 5. Join words split by a hyphen at the end of a line
    # Example: "in-\nlaw" becomes "in-law"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", text)

    # 6. Replace remaining single line breaks with spaces
    text = re.sub(r"\n", " ", text)

    # 7. Restore paragraph boundaries
    text = text.replace("<PARAGRAPH_BREAK>", "\n\n")

    # 8. Collapse repeated spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # 9. Remove spaces around paragraph breaks
    text = re.sub(r" *\n\n *", "\n\n", text)

    return text.strip()


def build_and_clean_corpus(
    metadata_df: pd.DataFrame,
    text_dir: Path = TEXT_DIR,
    max_docs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load raw text files and apply clean_legal_text across all documents.
    """
    records = []
    target_df = metadata_df if max_docs is None else metadata_df.head(max_docs)

    for _, row in target_df.iterrows():
        doc_path = row["path"]
        text_path = text_dir / f"{doc_path}.txt"

        try:
            raw_text = text_path.read_text(encoding="utf-8")
        except Exception:
            raw_text = ""

        records.append({
            "title": row.get("title", ""),
            "citation": row.get("citation", ""),
            "case_id": row.get("case_id", ""),
            "decision_date": row.get("decision_date", ""),
            "court": row.get("court", "Supreme Court of India"),
            "path": doc_path,
            "text_path": str(text_path),
            "raw_text": raw_text,
        })

    corpus_df = pd.DataFrame(records)
    print(f"Loaded raw text for {len(corpus_df):,} judgments. Applying cleaning...")
    corpus_df["clean_text"] = corpus_df["raw_text"].apply(clean_legal_text)
    return corpus_df


def compute_cleaning_metrics(corpus_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comparative word and character reduction metrics.
    """
    raw_chars = corpus_df["raw_text"].str.len()
    clean_chars = corpus_df["clean_text"].str.len()
    raw_words = corpus_df["raw_text"].str.split().str.len()
    clean_words = corpus_df["clean_text"].str.split().str.len()

    char_diff = raw_chars - clean_chars
    word_diff = raw_words - clean_words

    return {
        "total_judgments": len(corpus_df),
        "avg_raw_chars": float(raw_chars.mean()),
        "avg_clean_chars": float(clean_chars.mean()),
        "avg_raw_words": float(raw_words.mean()),
        "avg_clean_words": float(clean_words.mean()),
        "avg_char_reduction": float(char_diff.mean()),
        "avg_word_reduction": float(word_diff.mean()),
        "pct_char_reduction": float((char_diff.sum() / raw_chars.sum()) * 100),
        "pct_word_reduction": float((word_diff.sum() / raw_words.sum()) * 100),
    }


def run_section_05(force: bool = False) -> Dict[str, Any]:
    """Execute complete Section 05 pipeline."""
    print_header("Section 5: Text Cleaning and Corpus Preparation")

    if not force and CLEAN_CORPUS_FILE.exists():
        print(f"Loading existing cleaned corpus from: {CLEAN_CORPUS_FILE}")
        corpus_df = pd.read_parquet(CLEAN_CORPUS_FILE)
    else:
        metadata_df = load_raw_metadata()
        corpus_df = build_and_clean_corpus(metadata_df, TEXT_DIR)
        corpus_df.to_parquet(CLEAN_CORPUS_FILE, index=False)
        print(f"Saved cleaned corpus to: {CLEAN_CORPUS_FILE}")

    metrics = compute_cleaning_metrics(corpus_df)

    print("\nCleaning Transformation Metrics:")
    print(f"  - Judgments Processed      : {metrics['total_judgments']:,}")
    print(f"  - Avg Raw Words/Doc        : {metrics['avg_raw_words']:,.1f}")
    print(f"  - Avg Clean Words/Doc      : {metrics['avg_clean_words']:,.1f}")
    print(f"  - Avg Word Reduction       : {metrics['avg_word_reduction']:,.1f} ({metrics['pct_word_reduction']:.2f}%)")
    print(f"  - Avg Character Reduction  : {metrics['avg_char_reduction']:,.1f} ({metrics['pct_char_reduction']:.2f}%)")

    sample_preview = corpus_df.iloc[0]["clean_text"][:250].replace("\n", " ")
    print(f"\nCleaned Text Sample Preview:\n  \"{sample_preview}...\"")

    print("\n Section 5 cleaning pipeline completed successfully.")
    return {"corpus_df": corpus_df, "metrics": metrics}


if __name__ == "__main__":
    run_section_05()
