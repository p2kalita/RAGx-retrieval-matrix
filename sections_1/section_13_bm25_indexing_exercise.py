"""
Section 13: Sparse Lexical Retrieval Index II — BM25 Indexing — EXERCISE
========================================================================

Overview:
---------
BM25 (Best Matching 25) is an industry-standard probabilistic lexical ranking function
that incorporates term frequency saturation and document length normalization.
Compared to standard TF-IDF, BM25 prevents lengthy legal opinions from dominating results
simply by repeating statutory terms.

Tasks:
------
1. Implement `tokenize_corpus(chunks_df: pd.DataFrame) -> List[List[str]]`
   converting each chunk text into lowercased tokens.
2. Implement `build_bm25_index(tokenized_corpus: List[List[str]]) -> BM25Okapi`
   initializing and fitting the BM25 model using `rank_bm25.BM25Okapi`.
3. Implement `save_bm25_artefacts(bm25_model, tokenized_corpus, output_dir: Path)`
   saving `bm25.pkl` and `tokenized_corpus.pkl`.
4. Implement `search_bm25(query: str, bm25_model, chunks_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame`
   ranking chunks by BM25 score for the tokenized query.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from rank_bm25 import BM25Okapi
import joblib

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import BM25_DIR, print_header, load_chunks


def tokenize_corpus(chunks_df: pd.DataFrame) -> List[List[str]]:
    """
    Tokenize chunks into lists of lowercase whitespace-split words.

    Args:
        chunks_df: DataFrame containing 'chunk_text'.

    Returns:
        List[List[str]]: Tokenized corpus.
    """
    # TODO: [chunk.lower().split() for chunk in chunks_df['chunk_text']]
    raise NotImplementedError("Implement tokenize_corpus")


def build_bm25_index(tokenized_corpus: List[List[str]]) -> BM25Okapi:
    """
    Fit a BM25Okapi retrieval model on the tokenized corpus.

    Args:
        tokenized_corpus: List of token lists.

    Returns:
        BM25Okapi: Fitted BM25 model.
    """
    # TODO: Return BM25Okapi(tokenized_corpus)
    raise NotImplementedError("Implement build_bm25_index")


def save_bm25_artefacts(bm25_model: BM25Okapi, tokenized_corpus: List[List[str]], output_dir: Path = BM25_DIR) -> None:
    """
    Save the fitted BM25 model and tokenized corpus to disk via joblib.

    Args:
        bm25_model: Fitted model.
        tokenized_corpus: Tokenized corpus.
        output_dir: Destination folder.
    """
    # TODO: Save bm25.pkl and tokenized_corpus.pkl in output_dir using joblib.dump
    raise NotImplementedError("Implement save_bm25_artefacts")


def search_bm25(
    query: str,
    bm25_model: BM25Okapi,
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve top-k chunks for a legal query using BM25 scores.

    Args:
        query: Query string.
        bm25_model: Fitted BM25Okapi instance.
        chunks_df: Chunks dataframe.
        top_k: Number of chunks to retrieve.

    Returns:
        pd.DataFrame: Top-k retrieved chunks with scores and metadata.
    """
    # TODO: Tokenize query: query_tokens = query.lower().split()
    # Compute scores: scores = bm25_model.get_scores(query_tokens)
    # Sort indices descending, take top_k, and construct results DataFrame
    raise NotImplementedError("Implement search_bm25")


if __name__ == "__main__":
    print_header("Section 13: BM25 Indexing — Exercise Validation")
    try:
        sample_corpus = [
            ["bail", "under", "section", "438", "crpc"],
            ["murder", "conviction", "under", "section", "302", "ipc"],
            ["cheque", "dishonour", "section", "138", "ni", "act"],
        ]
        bm25 = build_bm25_index(sample_corpus)
        scores = bm25.get_scores(["murder", "section", "302"])
        print(f" [TODO] BM25 scores for 'murder section 302': {scores}")
        assert scores[1] > scores[0] and scores[1] > scores[2], "Document 1 should score highest"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
