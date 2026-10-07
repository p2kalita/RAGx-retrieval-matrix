"""
Section 8: Retrieval Index Preparation
(Part 3: Dense Retrieval — BGE Embedding Preparation and ChromaDB Indexing)

The BGE (BAAI General Embedding) model is the second dense retrieval model evaluated in this study. Similar to E5, BGE represents each legal text chunk as a dense semantic vector, enabling retrieval based on semantic similarity rather than lexical overlap. The generated embeddings are validated, stored, indexed in ChromaDB, and reused during retrieval experiments.
"""

import json
import os
from pathlib import Path
import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_bge_model(model_name="BAAI/bge-base-en-v1.5"):
    """
    Loading the BGE Embedding Model

    The BGE embedding model is loaded using the SentenceTransformers framework. This model generates dense semantic vector representations for legal text chunks and is widely used for information retrieval applications.
    """
    bge_model = SentenceTransformer(model_name)

    print("=" * 70)
    print("BGE Model Loaded Successfully")
    print("=" * 70)
    print(f"Model                 : {model_name}")
    print(f"Maximum Token Length  : {bge_model.max_seq_length}")
    print(f"Embedding Dimension   : {bge_model.get_sentence_embedding_dimension()}")
    return bge_model


def generate_and_save_bge_embeddings(bge_model, selected_chunks, bge_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\bge", bge_model_name="BAAI/bge-base-en-v1.5"):
    """
    8.6.3 Generation of BGE Chunk Embeddings

    Each legal text chunk in the common corpus is encoded using the BGE embedding model. Unlike E5, the BGE document embeddings are generated directly from the chunk text without adding document prefixes.

    The embeddings are generated in batches, normalised for cosine similarity, and saved immediately to persistent storage to avoid unnecessary recomputation.
    """
    os.makedirs(bge_dir, exist_ok=True)

    # Use the raw chunk text for BGE
    bge_passages = selected_chunks["chunk_text"].astype(str).tolist()

    print("=" * 70)
    print("BGE Embedding Generation")
    print("=" * 70)
    print(f"Model                 : {bge_model_name}")
    print(f"Total Passages        : {len(bge_passages):,}")
    print(f"Maximum Token Length  : {bge_model.max_seq_length}")
    print(f"Embedding Dimension   : {bge_model.get_sentence_embedding_dimension()}")
    print("Generating embeddings...")

    bge_embeddings = bge_model.encode(
        bge_passages,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save immediately
    bge_embeddings_path = os.path.join(
        bge_dir,
        "bge_embeddings.npy"
    )

    np.save(bge_embeddings_path, bge_embeddings)

    print("\n" + "=" * 70)
    print("BGE Embeddings Generated and Saved")
    print("=" * 70)
    print(f"Embedding Shape : {bge_embeddings.shape}")
    print(f"Data Type       : {bge_embeddings.dtype}")
    print(f"Saved At        : {bge_embeddings_path}")
    return bge_embeddings


def validate_bge_embeddings(bge_embeddings, selected_chunks):
    """
    8.6.4 Validation of the Generated BGE Embeddings

    The generated BGE embedding matrix is validated before vector indexing. The validation checks that the number of embeddings matches the number of legal text chunks, the embedding dimension is consistent, no NaN or infinite values are present, and the vectors are correctly normalised for cosine similarity.
    """
    print("=" * 70)
    print("BGE Embedding Validation")
    print("=" * 70)

    print(f"Embedding Matrix Shape      : {bge_embeddings.shape}")
    print(f"Expected Chunk Count        : {len(selected_chunks):,}")
    print(f"Embedding Row Count         : {bge_embeddings.shape[0]:,}")
    print(f"Embedding Dimension         : {bge_embeddings.shape[1]}")

    print(f"\nContains NaN Values         : {np.isnan(bge_embeddings).any()}")
    print(f"Contains Infinite Values    : {np.isinf(bge_embeddings).any()}")

    bge_vector_norms = np.linalg.norm(bge_embeddings, axis=1)

    print(f"\nMinimum Vector Norm         : {bge_vector_norms.min():.6f}")
    print(f"Maximum Vector Norm         : {bge_vector_norms.max():.6f}")
    print(f"Average Vector Norm         : {bge_vector_norms.mean():.6f}")


def create_bge_chromadb_collection(bge_chroma_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\bge\chromadb", collection_name="bge_collection"):
    """
    8.6.5 Construction of the BGE ChromaDB Collection

    The validated BGE embeddings are indexed in ChromaDB to enable efficient semantic similarity search. Each embedding is stored together with its corresponding chunk identifier, legal text, and metadata, ensuring that retrieved vectors remain linked to their original Supreme Court judgments.
    """
    os.makedirs(bge_chroma_dir, exist_ok=True)

    # Create persistent client
    bge_client = chromadb.PersistentClient(path=bge_chroma_dir)

    # Remove existing collection if it already exists
    try:
        bge_client.delete_collection(collection_name)
    except:
        pass

    # Create collection
    bge_collection = bge_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print("=" * 70)
    print("BGE ChromaDB Collection Created")
    print("=" * 70)
    return bge_client, bge_collection


def index_bge_embeddings_to_chromadb(bge_collection, selected_chunks, bge_embeddings, batch_size=500):
    """
    8.6.6 Indexing BGE Embeddings into ChromaDB

    The validated BGE embeddings are inserted into the ChromaDB collection together with their corresponding chunk identifiers, chunk text, and metadata. Embeddings are indexed in batches to improve memory efficiency and ensure reliable construction of the persistent vector database.
    """
    print("=" * 70)
    print("Adding BGE Embeddings to ChromaDB")
    print("=" * 70)

    for start in range(0, len(selected_chunks), batch_size):

        end = min(start + batch_size, len(selected_chunks))
        batch = selected_chunks.iloc[start:end]

        bge_collection.add(
            ids=batch["global_chunk_id"].astype(str).tolist(),
            embeddings=bge_embeddings[start:end].tolist(),
            documents=batch["chunk_text"].tolist(),
            metadatas=[
                {
                    "judgment_id": str(row["judgment_id"]),
                    "case_title": str(row["case_title"]),
                    "configuration": str(row["configuration"]),
                    "local_chunk_id": str(row["local_chunk_id"])
                }
                for _, row in batch.iterrows()
            ]
        )

    print("\n" + "=" * 70)
    print("BGE ChromaDB Indexing Completed")
    print("=" * 70)
    print(f"Indexed Chunks : {bge_collection.count():,}")


def validate_bge_chromadb_collection(bge_collection, selected_chunks):
    """
    8.6.7 Validation of the BGE ChromaDB Collection

    The completed BGE vector collection is validated to confirm that all legal text chunks were indexed successfully and that the associated metadata remains correctly aligned with each stored embedding.
    """
    print("=" * 70)
    print("BGE ChromaDB Collection Validation")
    print("=" * 70)

    indexed_count = bge_collection.count()

    print(f"Collection Name      : {bge_collection.name}")
    print(f"Indexed Chunks       : {indexed_count:,}")
    print(f"Expected Chunks      : {len(selected_chunks):,}")

    # Retrieve one stored record
    sample_record = bge_collection.get(
        limit=1,
        include=["documents", "metadatas"]
    )

    print(f"\nSample ID            : {sample_record['ids'][0]}")
    print(f"Case Title           : {sample_record['metadatas'][0]['case_title']}")
    print(f"Judgment ID          : {sample_record['metadatas'][0]['judgment_id']}")
    print(f"Configuration        : {sample_record['metadatas'][0]['configuration']}")

    # Final validation
    assert indexed_count == len(selected_chunks), (
        "The indexed chunk count does not match the selected corpus."
    )

    print("\nValidation Status     : PASSED")


def sample_bge_semantic_retrieval(bge_collection, bge_model, query="murder conviction under section 302 ipc"):
    """
    8.6.8 Sample Semantic Retrieval Using the BGE Model

    A sample legal query is executed against the BGE ChromaDB collection to verify that the indexed embeddings retrieve semantically relevant Supreme Court judgment chunks. The retrieved results provide an initial qualitative assessment of the semantic retrieval capability of the BGE model.

    Observation:
    The BGE embedding model successfully retrieved semantically relevant Supreme Court judgment chunks for the sample query on murder conviction under Section 302 IPC. The retrieved results included judgments discussing conviction, alteration of offences from Section 302 IPC to Section 304 IPC, sentencing, and criminal appeals. The low cosine distances indicate strong semantic similarity between the query and the retrieved legal text, confirming that the BGE embeddings and ChromaDB index function correctly for semantic legal retrieval over the indexed corpus
    """
    # Generate query embedding (no prefix required for BGE)
    query_embedding = bge_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    results = bge_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    print("=" * 70)
    print("Sample Semantic Retrieval Using BGE")
    print("=" * 70)

    for i in range(len(results["ids"][0])):

        print(f"\nRank {i+1}")
        print("-" * 60)

        print(f"Chunk ID      : {results['ids'][0][i]}")
        print(f"Distance      : {results['distances'][0][i]:.4f}")
        print(f"Case Title    : {results['metadatas'][0][i]['case_title']}")
        print(f"Judgment ID   : {results['metadatas'][0][i]['judgment_id']}")

        preview = results["documents"][0][i][:500].replace("\n", " ")
        print(f"\nPreview:\n{preview}...")


def save_bge_configuration(bge_model, bge_model_name, bge_embeddings, bge_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\bge"):
    """
    8.6.9 Saving the BGE Model Configuration

    The configuration used for generating the BGE embeddings is stored to ensure that the embedding model and indexing parameters can be reproduced consistently during future experiments and evaluations.
    """
    config = {
        "model_name": bge_model_name,
        "embedding_dimension": bge_embeddings.shape[1],
        "normalize_embeddings": True,
        "max_sequence_length": bge_model.max_seq_length,
        "total_embeddings": int(bge_embeddings.shape[0])
    }

    config_path = os.path.join(bge_dir, "bge_config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("=" * 70)
    print("BGE Configuration Saved")
    print("=" * 70)
    print(f"Saved At : {config_path}")


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    chunk_file = PROCESSED_DIR / "chunks_384_40.parquet"
    bge_dir = PROJECT_DIR / "models" / "bge"
    bge_chroma_dir = bge_dir / "chromadb"

    selected_chunks = pd.read_parquet(chunk_file)

    bge_model_name = "BAAI/bge-base-en-v1.5"
    bge_model = load_bge_model(bge_model_name)

    bge_embeddings_path = bge_dir / "bge_embeddings.npy"
    if bge_embeddings_path.exists():
        bge_embeddings = np.load(bge_embeddings_path)
    else:
        bge_embeddings = generate_and_save_bge_embeddings(bge_model, selected_chunks, str(bge_dir), bge_model_name)

    validate_bge_embeddings(bge_embeddings, selected_chunks)
    bge_client, bge_collection = create_bge_chromadb_collection(str(bge_chroma_dir), "bge_collection")
    index_bge_embeddings_to_chromadb(bge_collection, selected_chunks, bge_embeddings, batch_size=500)
    validate_bge_chromadb_collection(bge_collection, selected_chunks)
    sample_bge_semantic_retrieval(bge_collection, bge_model)
    save_bge_configuration(bge_model, bge_model_name, bge_embeddings, str(bge_dir))
