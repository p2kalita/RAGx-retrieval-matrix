"""
Section 14: Dense Semantic Retrieval I — E5 Embeddings & ChromaDB — EXERCISE
============================================================================

Overview:
---------
Dense semantic retrieval maps queries and documents into continuous embedding space,
allowing retrieval based on conceptual meaning rather than exact keyword overlap.
In this section, you will use `intfloat/e5-base-v2` (a 768-dimensional transformer)
and ChromaDB to build a persistent semantic vector database.

Crucial E5 Architecture Rule:
-----------------------------
E5 was trained with asymmetric task prefixes:
  - Document passages MUST be prefixed with "passage: "
  - Search queries MUST be prefixed with "query: "
Failing to include these prefixes causes significant performance degradation.

Tasks:
------
1. Implement `format_e5_passages(chunks_df: pd.DataFrame) -> List[str]`
   prefixing every chunk text with "passage: ".
2. Implement `create_or_get_e5_chromadb_collection(chroma_dir: Path, collection_name: str)`
   initializing a persistent ChromaDB client with cosine distance space.
3. Implement `search_e5(query: str, model, collection, top_k: int = 5) -> pd.DataFrame`
   encoding the query with the "query: " prefix, querying ChromaDB, and formatting results.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import chromadb
from chromadb.config import Settings

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import E5_DIR, print_header, load_chunks


def format_e5_passages(chunks_df: pd.DataFrame) -> List[str]:
    """
    Format chunk texts with the required E5 passage prefix.

    Args:
        chunks_df: DataFrame with 'chunk_text'.

    Returns:
        List[str]: List of "passage: {text}" strings.
    """
    # TODO: Return [f"passage: {t}" for t in chunks_df['chunk_text']]
    raise NotImplementedError("Implement format_e5_passages")


def create_or_get_e5_chromadb_collection(
    chroma_dir: Path = E5_DIR / "chromadb",
    collection_name: str = "nyayarag_e5",
):
    """
    Initialize persistent ChromaDB client and retrieve or create the collection.

    Args:
        chroma_dir: Directory where ChromaDB files are stored.
        collection_name: Name of the collection.

    Returns:
        chromadb.Collection: Persistent collection configured with cosine distance.
    """
    # TODO: Initialize chromadb.PersistentClient(path=str(chroma_dir))
    # Return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    raise NotImplementedError("Implement create_or_get_e5_chromadb_collection")


def search_e5(
    query: str,
    e5_model,
    collection,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Search the E5 vector index using a legal query.

    Args:
        query: Raw search query string.
        e5_model: Loaded SentenceTransformer model.
        collection: ChromaDB collection.
        top_k: Number of nearest neighbors to retrieve.

    Returns:
        pd.DataFrame: Retrieved chunks with distance and metadata.
    """
    # TODO: Encode f"query: {query}" with normalize_embeddings=True
    # Query collection.query(query_embeddings=[query_vec.tolist()], n_results=top_k)
    # Return formatted DataFrame
    raise NotImplementedError("Implement search_e5")


if __name__ == "__main__":
    print_header("Section 14: E5 Dense Retrieval — Exercise Validation")
    try:
        sample_df = pd.DataFrame({"chunk_text": ["Section 302 murder conviction.", "Bail petition under 438."]})
        passages = format_e5_passages(sample_df)
        print(f" [TODO] Formatted sample passages: {passages}")
        assert passages[0].startswith("passage: "), "Passage prefix required for E5"

        coll = create_or_get_e5_chromadb_collection(E5_DIR / "chromadb")
        print(f" [TODO] Connected to ChromaDB collection: {coll.name}")
        assert coll.name == "nyayarag_e5", "Expected nyayarag_e5 collection"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
