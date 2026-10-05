"""
Section 16: Legal Domain Semantic Retrieval III — InLegalBERT — SOLUTION
========================================================================

This module implements domain-specific dense semantic retrieval using
`law-ai/InLegalBERT`, an Indian legal pretrained transformer model, generating
domain-adapted dense representations for statutory and judicial retrieval
(Notebook Cells 264 to 281).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    INLEGALBERT_DIR,
    print_header,
    load_chunks,
)

INLEGALBERT_MODEL_NAME = "law-ai/InLegalBERT"
INLEGALBERT_EMBEDDINGS_FILE = INLEGALBERT_DIR / "inlegalbert_embeddings.npy"
INLEGALBERT_CONFIG_FILE = INLEGALBERT_DIR / "inlegalbert_config.json"
INLEGALBERT_CHROMA_DIR = INLEGALBERT_DIR / "chromadb"


def load_inlegalbert_model(model_name: str = INLEGALBERT_MODEL_NAME) -> SentenceTransformer:
    """Load the domain-specific InLegalBERT SentenceTransformer model."""
    print(f"Loading InLegalBERT model: {model_name}...")
    return SentenceTransformer(model_name)


def format_inlegalbert_passages(chunks_df: pd.DataFrame) -> List[str]:
    """Extract raw chunk texts for InLegalBERT encoding."""
    return chunks_df["chunk_text"].astype(str).tolist()


def generate_inlegalbert_embeddings(
    model: SentenceTransformer,
    passages: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Generate dense normalized embeddings using InLegalBERT."""
    print(f"Generating InLegalBERT embeddings for {len(passages):,} passages...")
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


def save_inlegalbert_artefacts(embeddings: np.ndarray, config: Dict[str, Any]) -> None:
    """Save embeddings numpy array and configuration metadata."""
    INLEGALBERT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INLEGALBERT_EMBEDDINGS_FILE, embeddings)
    with open(INLEGALBERT_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved InLegalBERT embeddings to : {INLEGALBERT_EMBEDDINGS_FILE}")
    print(f"Saved InLegalBERT config to     : {INLEGALBERT_CONFIG_FILE}")


def create_or_get_inlegalbert_chromadb_collection(
    chroma_dir: Path = INLEGALBERT_CHROMA_DIR,
    collection_name: str = "nyayarag_inlegalbert",
) -> Tuple[Any, str]:
    """Initialize ChromaDB client and collection."""
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


def search_inlegalbert(
    query: str,
    model: SentenceTransformer,
    collection,
    embeddings: Optional[np.ndarray],
    chunks_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Execute semantic retrieval on InLegalBERT representations."""
    t0 = time.time()
    query_embedding = model.encode(
        query,
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
        except Exception:
            pass

    if embeddings is not None:
        return direct_vector_search(query_embedding, embeddings, chunks_df, top_k=top_k)

    raise ValueError("Neither ChromaDB collection nor embeddings array is available.")


def run_section_16(force_recompute: bool = False, max_docs: Optional[int] = 10) -> Dict[str, Any]:
    """Execute complete Section 16 pipeline."""
    print_header("Section 16: Legal Domain Semantic Retrieval III — InLegalBERT")

    chunks_df = load_chunks()
    print(f"Loaded chunk corpus: {len(chunks_df):,} records.")

    collection, client_mode = create_or_get_inlegalbert_chromadb_collection(INLEGALBERT_CHROMA_DIR)
    print(f"ChromaDB mode: {client_mode}")

    model = load_inlegalbert_model()

    if not force_recompute and INLEGALBERT_EMBEDDINGS_FILE.exists():
        print(f"Loading precomputed InLegalBERT embeddings from: {INLEGALBERT_EMBEDDINGS_FILE}")
        embeddings = np.load(INLEGALBERT_EMBEDDINGS_FILE)
    else:
        target_df = chunks_df if max_docs is None else chunks_df.head(max_docs)
        print(f"Encoding {len(target_df):,} chunks with {INLEGALBERT_MODEL_NAME}...")
        passages = format_inlegalbert_passages(target_df)
        embeddings = generate_inlegalbert_embeddings(model, passages, batch_size=32)

        config = {
            "model_name": INLEGALBERT_MODEL_NAME,
            "embedding_dimension": int(embeddings.shape[1]),
            "total_chunks": int(len(target_df)),
            "metric": "cosine",
            "domain": "Indian Legal Corpus (Supreme Court / High Courts)",
        }
        save_inlegalbert_artefacts(embeddings, config)

    print("\nInLegalBERT Vector Index Characteristics:")
    print(f"  - Model Architecture   : {INLEGALBERT_MODEL_NAME}")
    print(f"  - Embedding Dimension  : {embeddings.shape[1]}")
    print(f"  - Total Embeddings     : {len(embeddings):,}")

    test_query = "murder conviction under section 302 ipc"
    print(f"\nSample Legal Search Query: \"{test_query}\"")
    results_df = search_inlegalbert(test_query, model, collection, embeddings, chunks_df.head(len(embeddings)), top_k=3)

    for _, r in results_df.iterrows():
        print(f"\n[Rank {r['rank']}] Cosine Sim: {r['score']:.4f} (Dist: {r['distance']:.4f}) | Chunk ID: {r['chunk_id']}")
        print(f"Case   : {r['case_title']}")
        print(f"Preview: {r['preview']}...")

    print(f"\nQuery Latency: {results_df.iloc[0]['latency_ms']} ms")
    print("\n Section 16 InLegalBERT legal domain retrieval completed successfully.")
    return {"model": model, "results_df": results_df}


if __name__ == "__main__":
    run_section_16()
