"""
Section 8: Retrieval Index Preparation
(Part 2: Dense Retrieval — E5 Embedding Preparation and ChromaDB Indexing)

The E5 (Embeddings from Weakly Supervised Contrastive Pre-training) model is the first dense retrieval model evaluated in this study. Unlike lexical retrieval methods such as TF-IDF and BM25, E5 represents each text chunk as a dense semantic vector in a continuous embedding space. Similarity between a legal query and judgment chunks is computed using vector similarity rather than exact keyword overlap, enabling retrieval of semantically related legal concepts even when different terminology is used.

Each chunk from the common corpus is encoded once using the E5 embedding model. The generated embeddings are validated, saved, indexed in ChromaDB, and reused throughout the retrieval experiments.
"""

import json
import os
from pathlib import Path
import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_e5_model(model_name="intfloat/e5-base-v2"):
    """
    Loading the E5 Embedding Model

    The multilingual E5 embedding model is loaded using the SentenceTransformers framework. This model generates fixed-length dense vector representations for each legal text chunk.
    """
    e5_model = SentenceTransformer(model_name)

    print("=" * 70)
    print("E5 Model Loaded Successfully")
    print("=" * 70)
    print(f"Model : {model_name}")
    return e5_model


def generate_and_save_e5_embeddings(e5_model, selected_chunks, e5_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\e5", e5_model_name="intfloat/e5-base-v2"):
    """
    Generation of E5 Chunk Embeddings

    Each chunk in the common corpus is prefixed with `passage:` before encoding, as required by the E5 retrieval architecture. The model converts every legal text chunk into a normalised dense vector representation.

    Embeddings are generated in batches to control memory consumption. The resulting embedding matrix is saved immediately to persistent Google Drive storage to avoid recomputation in later sessions.

    Observations:
    The E5 embedding model successfully generated dense semantic representations for all 30,521 legal text chunks. The resulting embedding matrix has dimensions 30,521 × 768, where each row corresponds to a unique chunk and each column represents a learned semantic feature. Embeddings were generated in normalized float32 format and saved successfully for subsequent vector indexing and semantic retrieval experiments.
    """
    os.makedirs(e5_dir, exist_ok=True)

    # Prefix every chunk with "passage:" as recommended for E5
    e5_passages = [
        f"passage: {text}"
        for text in selected_chunks["chunk_text"].astype(str).tolist()
    ]

    print("=" * 70)
    print("E5 Embedding Generation")
    print("=" * 70)
    print(f"Model                 : {e5_model_name}")
    print(f"Total Passages        : {len(e5_passages):,}")
    print(f"Maximum Token Length  : {e5_model.max_seq_length}")
    print(f"Embedding Dimension   : {e5_model.get_sentence_embedding_dimension()}")
    print("Generating embeddings...")

    # Generate embeddings
    e5_embeddings = e5_model.encode(
        e5_passages,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save embeddings
    e5_embeddings_path = os.path.join(
        e5_dir,
        "e5_embeddings.npy"
    )

    np.save(e5_embeddings_path, e5_embeddings)

    print("\n" + "=" * 70)
    print("E5 Embeddings Generated and Saved")
    print("=" * 70)
    print(f"Embedding Shape : {e5_embeddings.shape}")
    print(f"Data Type       : {e5_embeddings.dtype}")
    print(f"Saved At        : {e5_embeddings_path}")

    return e5_embeddings


def validate_e5_embeddings(e5_embeddings, selected_chunks):
    """
    8.5.4 Validation of the Generated E5 Embeddings

    The generated embedding matrix is validated to ensure that it has been created correctly before vector indexing. Validation confirms that the number of embeddings matches the number of text chunks, the embedding dimension is consistent, no missing or invalid values are present, and all vectors have been correctly normalised for cosine similarity.

    Observations:
    Validation confirmed that the E5 embedding matrix contains 30,521 dense semantic vectors, exactly matching the number of legal text chunks in the common corpus. Each embedding has a consistent dimensionality of 768, with no missing (NaN) or infinite values detected. All embedding vectors have a unit L2 norm of 1.0, confirming successful normalization for cosine similarity. The validated embeddings are suitable for semantic vector indexing and retrieval.
    """
    print("=" * 70)
    print("E5 Embedding Validation")
    print("=" * 70)

    print(f"Embedding Matrix Shape      : {e5_embeddings.shape}")
    print(f"Expected Chunk Count        : {len(selected_chunks):,}")
    print(f"Embedding Row Count         : {e5_embeddings.shape[0]:,}")
    print(f"Embedding Dimension         : {e5_embeddings.shape[1]}")

    print(f"\nContains NaN Values         : {np.isnan(e5_embeddings).any()}")
    print(f"Contains Infinite Values    : {np.isinf(e5_embeddings).any()}")

    # Check vector norms
    vector_norms = np.linalg.norm(e5_embeddings, axis=1)

    print(f"\nMinimum Vector Norm         : {vector_norms.min():.6f}")
    print(f"Maximum Vector Norm         : {vector_norms.max():.6f}")
    print(f"Average Vector Norm         : {vector_norms.mean():.6f}")


def create_e5_chromadb_collection(e5_chroma_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\e5\chromadb", collection_name="e5_collection"):
    """
    8.5.5 Construction of the E5 ChromaDB Collection

    The validated E5 embeddings are indexed in ChromaDB to enable efficient semantic similarity search. Each embedding is stored together with its corresponding chunk identifier, text, and metadata, ensuring that retrieved vectors can be traced back to the original Supreme Court judgment.
    """
    os.makedirs(e5_chroma_dir, exist_ok=True)

    # Create persistent client
    client = chromadb.PersistentClient(path=e5_chroma_dir)

    # Remove existing collection if present
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print("=" * 70)
    print("E5 ChromaDB Collection Created")
    print("=" * 70)
    return client, collection


def index_e5_embeddings_to_chromadb(collection, selected_chunks, e5_embeddings, batch_size=500):
    """
    8.5.6 Indexing E5 Embeddings into ChromaDB

    The validated E5 embeddings are inserted into the ChromaDB collection together with their corresponding chunk identifiers, chunk text, and metadata. The embeddings are indexed in batches to improve memory efficiency and ensure reliable construction of the persistent vector database.
    """
    print("=" * 70)
    print("Adding E5 Embeddings to ChromaDB")
    print("=" * 70)

    for start in range(0, len(selected_chunks), batch_size):

        end = min(start + batch_size, len(selected_chunks))
        batch = selected_chunks.iloc[start:end]

        collection.add(
            ids=batch["global_chunk_id"].astype(str).tolist(),
            embeddings=e5_embeddings[start:end].tolist(),
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
    print("E5 ChromaDB Indexing Completed")
    print("=" * 70)
    print(f"Indexed Chunks : {collection.count():,}")


def validate_e5_chromadb_collection(collection):
    """
    8.5.7 Validation of the E5 ChromaDB Collection

    The constructed ChromaDB collection is validated to verify that all legal text chunks have been indexed successfully. Validation confirms the number of indexed vectors and checks that the stored metadata remains correctly associated with each embedding.

    Observation:
    Validation confirmed that the E5 ChromaDB collection was successfully created and contains all 30,521 indexed legal text chunks. The stored metadata, including the chunk identifier, judgment identifier, case title, and chunk configuration, remained correctly associated with each embedding. These results confirm the integrity of the semantic vector index and its suitability for semantic retrieval experiments.
    """
    print("=" * 70)
    print("E5 ChromaDB Validation")
    print("=" * 70)

    print(f"Collection Name      : {collection.name}")
    print(f"Indexed Chunks       : {collection.count():,}")

    sample = collection.get(limit=1)

    print(f"Sample ID            : {sample['ids'][0]}")
    print(f"Case Title           : {sample['metadatas'][0]['case_title']}")
    print(f"Judgment ID          : {sample['metadatas'][0]['judgment_id']}")
    print(f"Configuration        : {sample['metadatas'][0]['configuration']}")


def sample_e5_semantic_retrieval(collection, e5_model, query="murder conviction under section 302 ipc"):
    """
    8.5.7 Sample Semantic Retrieval Using the E5 ChromaDB Collection

    A representative legal query is encoded using the E5 embedding model and submitted to the ChromaDB collection. The retrieved results are inspected to verify that the semantic vector index returns legally relevant judgment chunks based on semantic similarity rather than exact keyword matching.

    Observation:
    The E5 semantic retrieval model successfully returned legally relevant Supreme Court judgment chunks for the sample query relating to murder conviction under Section 302 IPC. The retrieved results included judgments discussing conviction, alteration of offences from Section 302 IPC to Section 304 IPC, sentencing, and criminal appeals. The low cosine distances indicate a high degree of semantic similarity between the query and the retrieved legal text. These results confirm that the E5 embeddings and ChromaDB collection are functioning correctly for semantic legal information retrieval.
    """
    # E5 requires the "query:" prefix
    query_embedding = e5_model.encode(
        f"query: {query}",
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    print("=" * 70)
    print("E5 Semantic Retrieval")
    print("=" * 70)

    for rank in range(len(results["ids"][0])):

        print(f"\nRank {rank+1}")
        print("-" * 70)

        print(f"Chunk ID   : {results['ids'][0][rank]}")
        print(f"Distance   : {results['distances'][0][rank]:.4f}")
        print(f"Case Title : {results['metadatas'][0][rank]['case_title']}")

        print("\nPreview:")
        print(results["documents"][0][rank][:350])


def save_e5_configuration(e5_model, e5_model_name, e5_embeddings, e5_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\e5"):
    """
    8.5.8 Saving E5 Model Configuration

    The configuration used to generate the E5 embeddings is saved to ensure reproducibility of the semantic retrieval experiments. The configuration records the embedding model, embedding dimension, normalization setting, and indexing parameters.
    """
    config = {
        "model_name": e5_model_name,
        "embedding_dimension": e5_embeddings.shape[1],
        "normalize_embeddings": True,
        "max_sequence_length": e5_model.max_seq_length,
        "document_prefix": "passage:",
        "query_prefix": "query:",
        "total_embeddings": int(e5_embeddings.shape[0])
    }

    config_path = os.path.join(e5_dir, "e5_config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("=" * 70)
    print("E5 Configuration Saved")
    print("=" * 70)
    print(f"Saved At : {config_path}")


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    chunk_file = PROCESSED_DIR / "chunks_384_40.parquet"
    e5_dir = PROJECT_DIR / "models" / "e5"
    e5_chroma_dir = e5_dir / "chromadb"

    selected_chunks = pd.read_parquet(chunk_file)

    e5_model_name = "intfloat/e5-base-v2"
    e5_model = load_e5_model(e5_model_name)

    e5_embeddings_path = e5_dir / "e5_embeddings.npy"
    if e5_embeddings_path.exists():
        e5_embeddings = np.load(e5_embeddings_path)
    else:
        e5_embeddings = generate_and_save_e5_embeddings(e5_model, selected_chunks, str(e5_dir), e5_model_name)

    validate_e5_embeddings(e5_embeddings, selected_chunks)
    client, collection = create_e5_chromadb_collection(str(e5_chroma_dir), "e5_collection")
    index_e5_embeddings_to_chromadb(collection, selected_chunks, e5_embeddings, batch_size=500)
    validate_e5_chromadb_collection(collection)
    sample_e5_semantic_retrieval(collection, e5_model)
    save_e5_configuration(e5_model, e5_model_name, e5_embeddings, str(e5_dir))
