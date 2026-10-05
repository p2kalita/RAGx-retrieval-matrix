"""
Section 13: Sparse Lexical Retrieval Index II — BM25 Indexing — SOLUTION
========================================================================

This module implements BM25 lexical indexing using `rank_bm25`, tokenizes
all 30,521 chunks, serializes the inverted index and tokenized representations,
and validates retrieval with legal test queries (Notebook Cells 208 to 220).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import joblib

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    BM25_DIR,
    print_header,
    load_chunks,
)

BM25_MODEL_FILE = BM25_DIR / "bm25.pkl"
TOKENIZED_CORPUS_FILE = BM25_DIR / "tokenized_corpus.pkl"


def tokenize_corpus(chunks_df: pd.DataFrame) -> List[List[str]]:
    """
    Tokenize chunks into lists of lowercase whitespace-split words.
    """
    return [chunk.lower().split() for chunk in chunks_df["chunk_text"]]


def build_bm25_index(tokenized_corpus: List[List[str]]) -> BM25Okapi:
    """
    Fit a BM25Okapi retrieval model on the tokenized corpus.
    """
    print(f"Building BM25Okapi inverted index for {len(tokenized_corpus):,} documents...")
    return BM25Okapi(tokenized_corpus)


def save_bm25_artefacts(
    bm25_model: BM25Okapi, tokenized_corpus: List[List[str]], output_dir: Path = BM25_DIR
) -> None:
    """
    Save the fitted BM25 model and tokenized corpus to disk via joblib.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bm25_model, BM25_MODEL_FILE)
    joblib.dump(tokenized_corpus, TOKENIZED_CORPUS_FILE)
    print(f"Saved BM25 model to        : {BM25_MODEL_FILE}")
    print(f"Saved tokenized corpus to  : {TOKENIZED_CORPUS_FILE}")


def load_bm25_artefacts() -> Tuple[BM25Okapi, List[List[str]]]:
    """
    Load precomputed BM25 model and tokenized corpus from disk.
    """
    if not BM25_MODEL_FILE.exists() or not TOKENIZED_CORPUS_FILE.exists():
        raise FileNotFoundError(f"BM25 artefacts not found in {BM25_DIR}")
    bm25 = joblib.load(BM25_MODEL_FILE)
    tokens = joblib.load(TOKENIZED_CORPUS_FILE)
    return bm25, tokens


def search_bm25(
    query: str,
    bm25_model: BM25Okapi,
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve top-k chunks for a legal query using BM25 scores.
    """
    t0 = time.time()
    query_tokens = query.lower().split()
    scores = np.array(bm25_model.get_scores(query_tokens))
    top_indices = np.argsort(scores)[::-1][:top_k]
    latency_ms = (time.time() - t0) * 1000

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = chunks_df.iloc[idx]
        chunk_id = row.get("global_chunk_id", row.get("chunk_id", f"idx_{idx}"))
        results.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "score": round(float(scores[idx]), 4),
            "case_title": row.get("case_title", ""),
            "judgment_id": row.get("judgment_id", ""),
            "preview": row["chunk_text"][:220].replace("\n", " "),
            "latency_ms": round(latency_ms, 2),
        })

    return pd.DataFrame(results)


def run_section_13(force_rebuild: bool = False) -> Dict[str, Any]:
    """Execute complete Section 13 pipeline."""
    print_header("Section 13: Sparse Lexical Retrieval II — BM25 Indexing")

    chunks_df = load_chunks()
    print(f"Loaded chunk corpus: {len(chunks_df):,} records.")

    if not force_rebuild and BM25_MODEL_FILE.exists() and TOKENIZED_CORPUS_FILE.exists():
        print(f"Loading existing BM25 artefacts from: {BM25_DIR}...")
        bm25, tokenized_corpus = load_bm25_artefacts()
    else:
        tokenized_corpus = tokenize_corpus(chunks_df)
        bm25 = build_bm25_index(tokenized_corpus)
        save_bm25_artefacts(bm25, tokenized_corpus)

    total_tokens = sum(len(doc) for doc in tokenized_corpus)
    avg_tokens = total_tokens / len(tokenized_corpus)

    print("\nBM25 Corpus Statistics:")
    print(f"  - Total Documents Indexed : {len(tokenized_corpus):,}")
    print(f"  - Total Tokens Indexed    : {total_tokens:,}")
    print(f"  - Average Tokens per Doc  : {avg_tokens:,.1f}")

    test_query = "murder conviction under section 302 ipc"
    print(f"\nSample Retrieval Query: \"{test_query}\"")
    results_df = search_bm25(test_query, bm25, chunks_df, top_k=3)

    for _, r in results_df.iterrows():
        print(f"\n[Rank {r['rank']}] BM25 Score: {r['score']:.4f} | Chunk ID: {r['chunk_id']}")
        print(f"Case   : {r['case_title']}")
        print(f"Preview: {r['preview']}...")

    print(f"\nQuery Latency: {results_df.iloc[0]['latency_ms']} ms")
    print("\n Section 13 BM25 indexing completed successfully.")
    return {"bm25": bm25, "results_df": results_df}


if __name__ == "__main__":
    run_section_13()
