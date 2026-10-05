"""
Section 12: Sparse Lexical Retrieval Index I — TF-IDF Vectorizer — SOLUTION
===========================================================================

This module builds a sparse TF-IDF retrieval index across all 30,521 chunks,
persists the vectorizer and CSR matrix, and validates lexical retrieval on
sample legal benchmark queries (Notebook Cells 188 to 207).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse
import joblib

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    TFIDF_DIR,
    print_header,
    load_chunks,
)

VECTORIZER_FILE = TFIDF_DIR / "tfidf_vectorizer.pkl"
MATRIX_FILE = TFIDF_DIR / "tfidf_matrix.npz"


def build_tfidf_index(
    chunks_df: pd.DataFrame,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
) -> Tuple[TfidfVectorizer, Any]:
    """
    Construct and fit a TF-IDF vectorizer across the chunk corpus.
    """
    print(f"Building TF-IDF Index (ngram_range={ngram_range}, min_df={min_df})...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(chunks_df["chunk_text"])
    return vectorizer, matrix


def save_tfidf_artefacts(
    vectorizer: TfidfVectorizer, matrix: Any, output_dir: Path = TFIDF_DIR
) -> None:
    """
    Persist the vectorizer and TF-IDF matrix to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    sparse.save_npz(MATRIX_FILE, matrix)
    print(f"Saved vectorizer to: {VECTORIZER_FILE}")
    print(f"Saved matrix to    : {MATRIX_FILE}")


def load_tfidf_artefacts() -> Tuple[TfidfVectorizer, Any]:
    """
    Load precomputed TF-IDF artefacts from disk.
    """
    if not VECTORIZER_FILE.exists() or not MATRIX_FILE.exists():
        raise FileNotFoundError(f"TF-IDF artefacts not found in {TFIDF_DIR}")
    vectorizer = joblib.load(VECTORIZER_FILE)
    matrix = sparse.load_npz(MATRIX_FILE)
    return vectorizer, matrix


def search_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve top-k chunks for a legal query using TF-IDF cosine similarity.
    """
    t0 = time.time()
    query_vec = vectorizer.transform([query])
    scores = linear_kernel(query_vec, matrix).flatten()
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


def run_section_12(force_rebuild: bool = False) -> Dict[str, Any]:
    """Execute complete Section 12 pipeline."""
    print_header("Section 12: Sparse Lexical Retrieval I — TF-IDF Vectorizer")

    chunks_df = load_chunks()
    print(f"Loaded chunks corpus: {len(chunks_df):,} records.")

    if not force_rebuild and VECTORIZER_FILE.exists() and MATRIX_FILE.exists():
        print(f"Loading existing TF-IDF artefacts from {TFIDF_DIR}...")
        vectorizer, matrix = load_tfidf_artefacts()
    else:
        vectorizer, matrix = build_tfidf_index(chunks_df)
        save_tfidf_artefacts(vectorizer, matrix)

    vocab_size = len(vectorizer.get_feature_names_out())
    matrix_shape = matrix.shape
    density = (matrix.nnz / (matrix_shape[0] * matrix_shape[1])) * 100

    print("\nTF-IDF Index Characteristics:")
    print(f"  - Vocabulary Features : {vocab_size:,}")
    print(f"  - Matrix Shape        : {matrix_shape[0]:,} chunks x {matrix_shape[1]:,} terms")
    print(f"  - Non-Zero Entries    : {matrix.nnz:,}")
    print(f"  - Sparsity / Density  : {density:.3f}% non-zero")

    test_query = "murder conviction under section 302 ipc"
    print(f"\nSample Retrieval Query: \"{test_query}\"")
    results_df = search_tfidf(test_query, vectorizer, matrix, chunks_df, top_k=3)

    for _, r in results_df.iterrows():
        print(f"\n[Rank {r['rank']}] Score: {r['score']:.4f} | Chunk ID: {r['chunk_id']}")
        print(f"Case   : {r['case_title']}")
        print(f"Preview: {r['preview']}...")

    print(f"\nQuery Latency: {results_df.iloc[0]['latency_ms']} ms")
    print("\n Section 12 TF-IDF indexing completed successfully.")
    return {"vectorizer": vectorizer, "matrix": matrix, "results_df": results_df}


if __name__ == "__main__":
    run_section_12()
