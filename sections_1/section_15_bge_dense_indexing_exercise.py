"""
Section 15: Dense Semantic Retrieval II — BGE Embeddings & ChromaDB — EXERCISE
==============================================================================

Overview:
---------
BAAI's `BAAI/bge-base-en-v1.5` is one of the highest-performing open-source dense
retrieval models on MTEB benchmarks. Unlike E5, standard BGE indexing operates
on raw passages without hardcoded prefixes, while query encoding can optionally
leverage domain retrieval prompts.

Tasks:
------
1. Implement `format_bge_passages(chunks_df: pd.DataFrame) -> List[str]`
   extracting the raw chunk texts as strings.
2. Implement `create_or_get_bge_chromadb_collection(chroma_dir: Path, collection_name: str)`
   initializing persistent ChromaDB or falling back gracefully to in-memory mode.
3. Implement `search_bge(query: str, bge_model, collection, embeddings, chunks_df, top_k: int = 5) -> pd.DataFrame`
   encoding the query and retrieving top-k nearest neighbors by cosine similarity.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import BGE_DIR, print_header, load_chunks


def format_bge_passages(chunks_df: pd.DataFrame) -> List[str]:
    """
    Extract chunk texts for BGE embedding generation.

    Args:
        chunks_df: DataFrame with 'chunk_text'.

    Returns:
        List[str]: List of clean chunk strings.
    """
    # TODO: Return chunks_df['chunk_text'].astype(str).tolist()
    raise NotImplementedError("Implement format_bge_passages")


def create_or_get_bge_chromadb_collection(
    chroma_dir: Path = BGE_DIR / "chromadb",
    collection_name: str = "nyayarag_bge",
):
    """
    Initialize persistent ChromaDB client or fall back to in-memory mode.

    Args:
        chroma_dir: Destination folder for vector database.
        collection_name: Name of ChromaDB collection.

    Returns:
        Tuple[chromadb.Collection, str]: Collection instance and client mode ('persistent' or 'in_memory').
    """
    # TODO: Initialize ChromaDB collection with cosine distance metric
    raise NotImplementedError("Implement create_or_get_bge_chromadb_collection")


def search_bge(
    query: str,
    bge_model,
    collection,
    embeddings: np.ndarray,
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Search the BGE vector index for a legal query.

    Args:
        query: Legal query string.
        bge_model: Loaded SentenceTransformer model.
        collection: ChromaDB collection.
        embeddings: Precomputed normalized embeddings array.
        chunks_df: Chunks dataframe.
        top_k: Number of results.

    Returns:
        pd.DataFrame: Top-k matching chunks with similarity scores.
    """
    # TODO: Encode query, query ChromaDB or compute cosine similarity with embeddings, return DataFrame
    raise NotImplementedError("Implement search_bge")


if __name__ == "__main__":
    print_header("Section 15: BGE Dense Retrieval — Exercise Validation")
    try:
        sample_df = pd.DataFrame({"chunk_text": ["Section 302 IPC murder conviction.", "Anticipatory bail petition."]})
        passages = format_bge_passages(sample_df)
        print(f" [TODO] Extracted {len(passages)} BGE passages.")
        assert len(passages) == 2, "Expected 2 passages"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
