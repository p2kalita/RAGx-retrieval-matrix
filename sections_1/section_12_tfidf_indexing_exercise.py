"""
Section 12: Sparse Lexical Retrieval Index I — TF-IDF Vectorizer — EXERCISE
===========================================================================

Overview:
---------
TF-IDF (Term Frequency - Inverse Document Frequency) serves as a fundamental lexical
retrieval baseline. In legal search, TF-IDF effectively captures statutory section numbers
(e.g., "Section 302", "Section 438"), party names, and legal terminology without requiring
GPU infrastructure.

Tasks:
------
1. Implement `build_tfidf_index(chunks_df: pd.DataFrame, ngram_range: tuple, min_df: int)`
   configuring and fitting a `TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=ngram_range, min_df=min_df)`.
2. Implement `save_tfidf_artefacts(vectorizer, matrix, output_dir: Path)`
   saving `tfidf_vectorizer.pkl` (via joblib) and `tfidf_matrix.npz` (via scipy.sparse).
3. Implement `search_tfidf(query: str, vectorizer, matrix, chunks_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame`
   transforming the query, computing cosine similarity via `linear_kernel`, and returning
   the top-k matching chunks with similarity scores.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse
import joblib

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import TFIDF_DIR, print_header, load_chunks


def build_tfidf_index(
    chunks_df: pd.DataFrame,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
) -> Tuple[TfidfVectorizer, Any]:
    """
    Construct and fit a TF-IDF vectorizer across the chunk corpus.

    Args:
        chunks_df: DataFrame containing 'chunk_text'.
        ngram_range: Unigram and bigram bounds (default: (1, 2)).
        min_df: Minimum document frequency threshold.

    Returns:
        Tuple[TfidfVectorizer, scipy.sparse.csr_matrix]: Fitted vectorizer and sparse matrix.
    """
    # TODO: Initialize TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=ngram_range, min_df=min_df)
    # Fit and transform chunks_df['chunk_text']
    # Return (vectorizer, tfidf_matrix)
    raise NotImplementedError("Implement build_tfidf_index")


def save_tfidf_artefacts(vectorizer: TfidfVectorizer, matrix: Any, output_dir: Path = TFIDF_DIR) -> None:
    """
    Persist the vectorizer and TF-IDF matrix to disk.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        matrix: Scipy sparse matrix.
        output_dir: Target directory.
    """
    # TODO: Ensure output_dir exists. Save vectorizer with joblib.dump and matrix with sparse.save_npz
    raise NotImplementedError("Implement save_tfidf_artefacts")


def search_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve top-k chunks for a legal query using TF-IDF cosine similarity.

    Args:
        query: Query string.
        vectorizer: Fitted TfidfVectorizer.
        matrix: Document TF-IDF matrix.
        chunks_df: Chunks dataframe.
        top_k: Number of results.

    Returns:
        pd.DataFrame: Top-k retrieved chunks with 'rank', 'score', and chunk details.
    """
    # TODO: Transform query: query_vec = vectorizer.transform([query])
    # Compute similarity: sims = linear_kernel(query_vec, matrix).flatten()
    # Rank descending, slice top_k indices, and format result DataFrame
    raise NotImplementedError("Implement search_tfidf")


if __name__ == "__main__":
    print_header("Section 12: TF-IDF Indexing — Exercise Validation")
    try:
        sample_chunks = pd.DataFrame({
            "chunk_id": ["c1", "c2", "c3"],
            "case_title": ["Case A", "Case B", "Case C"],
            "judgment_id": ["j1", "j2", "j3"],
            "chunk_text": [
                "Bail application under section 438 of the Code of Criminal Procedure.",
                "Conviction under section 302 of the Indian Penal Code for murder.",
                "Dishonour of cheque under section 138 of Negotiable Instruments Act.",
            ],
        })

        vec, mat = build_tfidf_index(sample_chunks, ngram_range=(1, 1), min_df=1)
        print(f" [TODO] Built TF-IDF with vocab size: {len(vec.vocabulary_)}")
        assert len(vec.vocabulary_) > 10, "Vocabulary should be populated"

        res = search_tfidf("murder section 302", vec, mat, sample_chunks, top_k=1)
        print(f" [TODO] Top search result for 'murder section 302': {res.iloc[0]['case_title']} (score: {res.iloc[0]['score']:.4f})")
        assert res.iloc[0]["chunk_id"] == "c2", "Case B should rank first for section 302 murder"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
