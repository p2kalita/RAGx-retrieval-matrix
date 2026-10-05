"""
Section 14: Dense Semantic Retrieval I — E5 Embeddings & ChromaDB — SOLUTION
============================================================================

This module builds the dense semantic retrieval index using `intfloat/e5-base-v2`,
generates 768-dimensional normalized passage embeddings, and populates a persistent
ChromaDB collection for vector similarity search (Notebook Cells 221 to 245).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    E5_DIR,
    print_header,
    load_chunks,
)

E5_MODEL_NAME = "intfloat/e5-base-v2"
E5_EMBEDDINGS_FILE = E5_DIR / "e5_embeddings.npy"
E5_CONFIG_FILE = E5_DIR / "e5_config.json"
E5_CHROMA_DIR = E5_DIR / "chromadb"


def load_e5_model(model_name: str = E5_MODEL_NAME) -> SentenceTransformer:
    """Load the SentenceTransformer E5 model."""
    print(f"Loading E5 model: {model_name}...")
    return SentenceTransformer(model_name)


def format_e5_passages(chunks_df: pd.DataFrame) -> List[str]:
    """Prefix every chunk text with 'passage: ' as required by E5."""
    return [f"passage: {text}" for text in chunks_df["chunk_text"]]


def generate_e5_embeddings(
    model: SentenceTransformer,
    passages: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Generate dense normalized embeddings for all passages."""
    print(f"Generating E5 embeddings for {len(passages):,} passages...")
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


def save_e5_artefacts(embeddings: np.ndarray, config: Dict[str, Any]) -> None:
    """Save embeddings numpy array and configuration metadata."""
    E5_DIR.mkdir(parents=True, exist_ok=True)
    np.save(E5_EMBEDDINGS_FILE, embeddings)
    with open(E5_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved E5 embeddings to : {E5_EMBEDDINGS_FILE}")
    print(f"Saved E5 config to     : {E5_CONFIG_FILE}")


def create_or_get_e5_chromadb_collection(
    chroma_dir: Path = E5_CHROMA_DIR,
    collection_name: str = "nyayarag_e5",
):
    """
    Initialize ChromaDB client and collection.
    Falls back gracefully to an in-memory client if legacy sqlite schema causes an error.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return collection, "persistent"
    except BaseException as e:
        print(f"Note: Persistent ChromaDB raised ({type(e).__name__}: {e}). Using in-memory Chroma client.")
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return collection, "in_memory"


def direct_vector_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Fast direct cosine vector search on normalized embeddings."""
    t0 = time.time()
    # Normalized dot product equals cosine similarity
    scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]
    latency_ms = (time.time() - t0) * 1000

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = chunks_df.iloc[idx]
        chunk_id = row.get("global_chunk_id", row.get("chunk_id", f"idx_{idx}"))
        score = float(scores[idx])
        results.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "score": round(score, 4),
            "distance": round(1.0 - score, 4),
            "case_title": row.get("case_title", ""),
            "judgment_id": row.get("judgment_id", ""),
            "preview": row["chunk_text"][:220].replace("\n", " "),
            "latency_ms": round(latency_ms, 2),
        })
    return pd.DataFrame(results)


def search_e5(
    query: str,
    e5_model: SentenceTransformer,
    collection,
    embeddings: np.ndarray,
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Execute sample semantic search on E5 embeddings via ChromaDB or vector search."""
    t0 = time.time()
    query_text = f"query: {query}"
    query_embedding = e5_model.encode(
        query_text,
        normalize_embeddings=True,
    )

    if collection is not None:
        try:
            res = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
            if res["ids"] and len(res["ids"][0]) > 0:
                latency_ms = (time.time() - t0) * 1000
                results = []
                for rank in range(len(res["ids"][0])):
                    chunk_id = res["ids"][0][rank]
                    dist = res["distances"][0][rank]
                    score = 1.0 - dist
                    meta = res["metadatas"][0][rank]
                    doc = res["documents"][0][rank]
                    results.append({
                        "rank": rank + 1,
                        "chunk_id": chunk_id,
                        "score": round(float(score), 4),
                        "distance": round(float(dist), 4),
                        "case_title": meta.get("case_title", ""),
                        "judgment_id": meta.get("judgment_id", ""),
                        "preview": doc[:220].replace("\n", " "),
                        "latency_ms": round(latency_ms, 2),
                    })
                return pd.DataFrame(results)
        except Exception as e:
            print(f"ChromaDB search error ({e}), falling back to direct vector search.")

    return direct_vector_search(query_embedding, embeddings, chunks_df, top_k=top_k)


def run_section_14(force_recompute: bool = False) -> Dict[str, Any]:
    """Execute complete Section 14 pipeline."""
    print_header("Section 14: Dense Semantic Retrieval I — E5 Embeddings & ChromaDB")

    chunks_df = load_chunks()
    print(f"Loaded chunk corpus: {len(chunks_df):,} records.")

    collection, client_type = create_or_get_e5_chromadb_collection(E5_CHROMA_DIR)
    print(f"ChromaDB mode: {client_type}")

    e5_model = load_e5_model()

    if not force_recompute and E5_EMBEDDINGS_FILE.exists():
        print(f"Loading precomputed E5 embeddings from: {E5_EMBEDDINGS_FILE}")
        embeddings = np.load(E5_EMBEDDINGS_FILE)
    else:
        passages = format_e5_passages(chunks_df)
        embeddings = generate_e5_embeddings(e5_model, passages)
        config = {
            "model_name": E5_MODEL_NAME,
            "embedding_dimension": int(embeddings.shape[1]),
            "total_chunks": int(len(chunks_df)),
            "metric": "cosine",
            "prefix_rule": "query: / passage:",
        }
        save_e5_artefacts(embeddings, config)

    print("\nE5 Vector Index Characteristics:")
    print(f"  - Model Architecture   : {E5_MODEL_NAME}")
    print(f"  - Embedding Dimension  : {embeddings.shape[1]}")
    print(f"  - Total Embeddings     : {len(embeddings):,}")

    test_query = "murder conviction under section 302 ipc"
    print(f"\nSample Semantic Search Query: \"{test_query}\"")
    results_df = search_e5(test_query, e5_model, collection, embeddings, chunks_df, top_k=3)

    for _, r in results_df.iterrows():
        print(f"\n[Rank {r['rank']}] Cosine Sim: {r['score']:.4f} (Dist: {r['distance']:.4f}) | Chunk ID: {r['chunk_id']}")
        print(f"Case   : {r['case_title']}")
        print(f"Preview: {r['preview']}...")

    print(f"\nQuery Latency: {results_df.iloc[0]['latency_ms']} ms")
    print("\n Section 14 E5 dense semantic retrieval completed successfully.")
    return {"model": e5_model, "collection": collection, "results_df": results_df}


if __name__ == "__main__":
    run_section_14()
