"""
Section 10: Refined Chunker & Corpus-Wide Chunk Generation — SOLUTION
=====================================================================

This module executes corpus-wide chunk generation across all 782 judgments,
evaluates candidate chunk configurations, and establishes the standard
`chunk_384_40` dataset (Notebook Cells 159 to 171).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    CHUNKS_FILE,
    print_header,
    load_processed_corpus,
)
from sections_1.section_08_paragraph_aware_chunking_solution import paragraph_aware_chunker

CANDIDATE_CONFIGS = {
    "chunk_256_30": {"target_size": 256, "overlap_size": 30},
    "chunk_384_40": {"target_size": 384, "overlap_size": 40},
    "chunk_512_50": {"target_size": 512, "overlap_size": 50},
}


def generate_chunks_for_corpus(
    corpus_df: pd.DataFrame,
    target_size: int = 384,
    overlap_size: int = 40,
    max_judgments: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate paragraph-aware chunks for every document in the corpus.
    """
    records = []
    global_chunk_id = 1
    target_df = corpus_df if max_judgments is None else corpus_df.head(max_judgments)
    total_docs = len(target_df)

    for doc_idx, (_, row) in enumerate(target_df.iterrows(), start=1):
        text = row.get("judgment_text", row.get("clean_text", ""))
        chunks = paragraph_aware_chunker(text, target_size, overlap_size)

        judgment_id = row.get("path", f"doc_{doc_idx}")
        case_title = row.get("title", "")
        citation = row.get("citation", "")
        case_id = row.get("case_id", "")
        decision_date = row.get("decision_date", "")
        court = row.get("court", "Supreme Court of India")

        for local_id, chunk_item in enumerate(chunks, start=1):
            chunk_txt = chunk_item["chunk_text"]
            records.append({
                "chunk_id": f"chunk_{global_chunk_id:06d}",
                "judgment_id": judgment_id,
                "case_title": case_title,
                "citation": citation,
                "case_id": case_id,
                "decision_date": decision_date,
                "court": court,
                "local_chunk_id": local_id,
                "chunk_text": chunk_txt,
                "word_count": chunk_item["word_count"],
                "character_count": len(chunk_txt),
            })
            global_chunk_id += 1

        if doc_idx % 150 == 0 or doc_idx == total_docs:
            print(f"Chunked [{doc_idx}/{total_docs}] judgments -> {len(records):,} chunks so far...")

    chunks_df = pd.DataFrame(records)
    return chunks_df


def compare_chunking_configurations(chunk_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute comparative distribution metrics across candidate chunking configurations.
    """
    summary = []
    for cfg_name, df in chunk_sets.items():
        summary.append({
            "Configuration": cfg_name,
            "Total Chunks": len(df),
            "Mean Words": round(df["word_count"].mean(), 2),
            "Median Words": round(df["word_count"].median(), 2),
            "Min Words": df["word_count"].min(),
            "Max Words": df["word_count"].max(),
            "Std Dev Words": round(df["word_count"].std(), 2),
        })
    return pd.DataFrame(summary)


def run_section_10(force_rechunk: bool = False) -> Dict[str, Any]:
    """Execute complete Section 10 pipeline."""
    print_header("Section 10: Refined Chunker & Corpus-Wide Chunk Generation")

    if not force_rechunk and CHUNKS_FILE.exists():
        print(f"Loading precomputed chunks from: {CHUNKS_FILE}")
        selected_chunks = pd.read_parquet(CHUNKS_FILE)
        print(f"Loaded existing chunk corpus: {len(selected_chunks):,} chunks.")
    else:
        corpus_df = load_processed_corpus()
        print(f"Generating chunks for complete corpus ({len(corpus_df):,} judgments)...")
        selected_chunks = generate_chunks_for_corpus(
            corpus_df,
            target_size=CANDIDATE_CONFIGS["chunk_384_40"]["target_size"],
            overlap_size=CANDIDATE_CONFIGS["chunk_384_40"]["overlap_size"],
        )
        selected_chunks.to_parquet(CHUNKS_FILE, index=False)
        print(f"Saved {len(selected_chunks):,} chunks to: {CHUNKS_FILE}")

    print("\nSelected Configuration (chunk_384_40) Summary:")
    print(f"  - Total Generated Chunks : {len(selected_chunks):,}")
    print(f"  - Unique Judgments       : {selected_chunks['judgment_id'].nunique():,}")
    print(f"  - Mean Words per Chunk   : {selected_chunks['word_count'].mean():,.1f}")
    print(f"  - Median Words per Chunk : {selected_chunks['word_count'].median():,.1f}")
    print(f"  - Min Words              : {selected_chunks['word_count'].min():,}")
    print(f"  - Max Words              : {selected_chunks['word_count'].max():,}")

    preview_chunk = selected_chunks.iloc[0]
    chunk_id_val = preview_chunk.get("global_chunk_id", preview_chunk.get("chunk_id", "0"))
    print(f"\nChunk 0 Inspection:")
    print(f"  - Chunk ID   : {chunk_id_val}")
    print(f"  - Case Title : {preview_chunk['case_title']}")
    print(f"  - Preview    : \"{preview_chunk['chunk_text'][:200].replace(chr(10), ' ')}...\"")

    print("\n Section 10 corpus chunk generation completed successfully.")
    return {"selected_chunks": selected_chunks}


if __name__ == "__main__":
    run_section_10()
