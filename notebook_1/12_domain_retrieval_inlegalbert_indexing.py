"""
Section 8: Retrieval Index Preparation
(Part 4: Domain-Specific Dense Retrieval — InLegalBERT Representation Preparation and ChromaDB Indexing)

InLegalBERT is a transformer model pre-trained on Indian legal text. Unlike general-purpose embedding models, it is designed to capture the terminology, context, and linguistic characteristics of Indian legal documents. In this study, InLegalBERT is evaluated as a domain-specific dense retrieval model by generating vector representations for the legal text chunks, indexing them in ChromaDB, and assessing their retrieval performance over the Indian Supreme Court judgment corpus.
"""

import json
import os
from pathlib import Path
import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_inlegalbert_model(model_name="law-ai/InLegalBERT"):
    """
    Load InLegalBERT Model

    The InLegalBERT model is loaded using the SentenceTransformers framework.
    """
    inlegalbert_model = SentenceTransformer(model_name)

    print("=" * 70)
    print("InLegalBERT Model Loaded Successfully")
    print("=" * 70)
    print(f"Model                 : {model_name}")
    print(f"Maximum Token Length  : {inlegalbert_model.max_seq_length}")
    print(f"Embedding Dimension   : {inlegalbert_model.get_sentence_embedding_dimension()}")
    return inlegalbert_model


def generate_and_save_inlegalbert_embeddings(inlegalbert_model, selected_chunks, inlegalbert_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\inlegalbert", inlegalbert_model_name="law-ai/InLegalBERT"):
    """
    8.7.1 Generation of InLegalBERT Embeddings

    Each legal text chunk is encoded using the InLegalBERT embedding model to generate dense semantic representations. The embeddings are generated in batches, normalised for cosine similarity, and immediately saved to persistent storage for reuse during retrieval experiments.
    """
    os.makedirs(inlegalbert_dir, exist_ok=True)

    # Use raw chunk text
    inlegalbert_passages = selected_chunks["chunk_text"].astype(str).tolist()

    print("=" * 70)
    print("InLegalBERT Embedding Generation")
    print("=" * 70)
    print(f"Model                 : {inlegalbert_model_name}")
    print(f"Total Passages        : {len(inlegalbert_passages):,}")
    print(f"Maximum Token Length  : {inlegalbert_model.max_seq_length}")
    print(f"Embedding Dimension   : {inlegalbert_model.get_sentence_embedding_dimension()}")
    print("Generating embeddings...")

    inlegalbert_embeddings = inlegalbert_model.encode(
        inlegalbert_passages,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save embeddings
    inlegalbert_embeddings_path = os.path.join(
        inlegalbert_dir,
        "inlegalbert_embeddings.npy"
    )

    np.save(inlegalbert_embeddings_path, inlegalbert_embeddings)

    print("\n" + "=" * 70)
    print("InLegalBERT Embeddings Generated and Saved")
    print("=" * 70)
    print(f"Embedding Shape : {inlegalbert_embeddings.shape}")
    print(f"Data Type       : {inlegalbert_embeddings.dtype}")
    print(f"Saved At        : {inlegalbert_embeddings_path}")
    return inlegalbert_embeddings


def validate_inlegalbert_embeddings(inlegalbert_embeddings, selected_chunks):
    """
    8.7.2 Validation of the Generated InLegalBERT Embeddings

    The generated InLegalBERT embedding matrix is validated before vector indexing. The validation verifies that the number of embeddings matches the number of legal text chunks, the embedding dimension is correct, no NaN or infinite values are present, and the embeddings are correctly normalised for cosine similarity.

    Observation:
    The InLegalBERT embedding matrix was successfully validated. The number of generated embeddings matches the selected corpus, each embedding has a dimensionality of 768, no invalid values were detected, and all vectors were correctly normalised for cosine similarity. The validated embeddings are therefore suitable for semantic indexing and retrieval.
    """
    print("=" * 70)
    print("InLegalBERT Embedding Validation")
    print("=" * 70)

    print(f"Embedding Matrix Shape      : {inlegalbert_embeddings.shape}")
    print(f"Expected Chunk Count        : {len(selected_chunks):,}")
    print(f"Embedding Row Count         : {inlegalbert_embeddings.shape[0]:,}")
    print(f"Embedding Dimension         : {inlegalbert_embeddings.shape[1]}")

    print(f"\nContains NaN Values         : {np.isnan(inlegalbert_embeddings).any()}")
    print(f"Contains Infinite Values    : {np.isinf(inlegalbert_embeddings).any()}")

    vector_norms = np.linalg.norm(inlegalbert_embeddings, axis=1)

    print(f"\nMinimum Vector Norm         : {vector_norms.min():.6f}")
    print(f"Maximum Vector Norm         : {vector_norms.max():.6f}")
    print(f"Average Vector Norm         : {vector_norms.mean():.6f}")


def create_inlegalbert_chromadb_collection(inlegalbert_chroma_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\inlegalbert\chromadb", collection_name="inlegalbert_collection"):
    """
    8.7.3 Construction of the InLegalBERT ChromaDB Collection

    The validated InLegalBERT embeddings are indexed in ChromaDB to enable efficient semantic similarity search. Each embedding is stored together with its corresponding chunk identifier, legal text, and metadata, ensuring that retrieved vectors remain linked to their original Supreme Court judgments.
    """
    os.makedirs(inlegalbert_chroma_dir, exist_ok=True)

    # Create persistent client
    inlegalbert_client = chromadb.PersistentClient(path=inlegalbert_chroma_dir)

    # Remove existing collection if it already exists
    try:
        inlegalbert_client.delete_collection(collection_name)
    except:
        pass

    # Create collection
    inlegalbert_collection = inlegalbert_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print("=" * 70)
    print("InLegalBERT ChromaDB Collection Created")
    print("=" * 70)
    return inlegalbert_client, inlegalbert_collection


def index_inlegalbert_embeddings_to_chromadb(inlegalbert_collection, selected_chunks, inlegalbert_embeddings, batch_size=500):
    """
    8.7.6 Indexing InLegalBERT Embeddings into ChromaDB

    The validated InLegalBERT embeddings are inserted into the ChromaDB collection together with their corresponding chunk identifiers, chunk text, and metadata. The embeddings are indexed in batches to improve memory efficiency and ensure reliable construction of the persistent vector database.
    """
    print("=" * 70)
    print("Adding InLegalBERT Embeddings to ChromaDB")
    print("=" * 70)

    for start in range(0, len(selected_chunks), batch_size):

        end = min(start + batch_size, len(selected_chunks))
        batch = selected_chunks.iloc[start:end]

        inlegalbert_collection.add(
            ids=batch["global_chunk_id"].astype(str).tolist(),
            embeddings=inlegalbert_embeddings[start:end].tolist(),
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
    print("InLegalBERT ChromaDB Indexing Completed")
    print("=" * 70)
    print(f"Indexed Chunks : {inlegalbert_collection.count():,}")


def validate_inlegalbert_chromadb_collection(inlegalbert_collection, selected_chunks):
    """
    8.7.7 Validate the InLegalBERT ChromaDB collection

    The completed InLegalBERT vector collection is validated to verify that all legal text chunks were indexed successfully and that the associated metadata remains correctly linked with each stored embedding.
    """
    print("=" * 70)
    print("InLegalBERT ChromaDB Collection Validation")
    print("=" * 70)

    indexed_count = inlegalbert_collection.count()

    print(f"Collection Name      : {inlegalbert_collection.name}")
    print(f"Indexed Chunks       : {indexed_count:,}")
    print(f"Expected Chunks      : {len(selected_chunks):,}")

    sample_record = inlegalbert_collection.get(
        limit=1,
        include=["documents", "metadatas"]
    )

    print(f"\nSample ID            : {sample_record['ids'][0]}")
    print(f"Case Title           : {sample_record['metadatas'][0]['case_title']}")
    print(f"Judgment ID          : {sample_record['metadatas'][0]['judgment_id']}")
    print(f"Configuration        : {sample_record['metadatas'][0]['configuration']}")

    assert indexed_count == len(selected_chunks), \
        "Indexed chunk count does not match corpus."

    print("\nValidation Status     : PASSED")


def sample_inlegalbert_semantic_retrieval(inlegalbert_collection, inlegalbert_model, query="murder conviction under section 302 ipc"):
    """
    8.7.8 Sample semantic retrieval using InLegalBERT

    A sample legal query is executed against the InLegalBERT ChromaDB collection to verify that the indexed embeddings retrieve semantically relevant Supreme Court judgment chunks. The retrieved results provide an initial qualitative assessment of the retrieval capability of the domain-specific embedding model.

    Observation:
    The InLegalBERT model successfully retrieved semantically relevant Supreme Court judgment chunks for the sample query on murder conviction under Section 302 IPC. The retrieved results primarily comprised criminal law judgments involving Section 302 IPC, murder trials, convictions, and related appellate proceedings. Compared with the earlier qualitative retrieval examples using E5 and BGE, the retrieved results exhibited broader semantic relevance, indicating that InLegalBERT effectively captures legal context while varying in retrieval specificity for individual queries. The successful retrieval confirms that the InLegalBERT embeddings and ChromaDB index are functioning correctly and are suitable for subsequent quantitative evaluation.
    """
    query_embedding = inlegalbert_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    results = inlegalbert_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    print("=" * 70)
    print("Sample Semantic Retrieval Using InLegalBERT")
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


def save_inlegalbert_configuration(inlegalbert_model, inlegalbert_model_name, inlegalbert_embeddings, inlegalbert_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\inlegalbert"):
    """
    8.7.9 Save the InLegalBERT configuration

    The configuration used to generate the InLegalBERT embeddings is stored to ensure reproducibility of the retrieval experiments. The configuration records the embedding model, embedding dimension, normalization setting, and indexing parameters.
    """
    config = {
        "model_name": inlegalbert_model_name,
        "embedding_dimension": inlegalbert_embeddings.shape[1],
        "normalize_embeddings": True,
        "max_sequence_length": inlegalbert_model.max_seq_length,
        "total_embeddings": int(inlegalbert_embeddings.shape[0])
    }

    config_path = os.path.join(
        inlegalbert_dir,
        "inlegalbert_config.json"
    )

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("=" * 70)
    print("InLegalBERT Configuration Saved")
    print("=" * 70)
    print(f"Saved At : {config_path}")


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    chunk_file = PROCESSED_DIR / "chunks_384_40.parquet"
    inlegalbert_dir = PROJECT_DIR / "models" / "inlegalbert"
    inlegalbert_chroma_dir = inlegalbert_dir / "chromadb"

    selected_chunks = pd.read_parquet(chunk_file)

    inlegalbert_model_name = "law-ai/InLegalBERT"
    inlegalbert_model = load_inlegalbert_model(inlegalbert_model_name)

    inlegalbert_embeddings_path = inlegalbert_dir / "inlegalbert_embeddings.npy"
    if inlegalbert_embeddings_path.exists():
        inlegalbert_embeddings = np.load(inlegalbert_embeddings_path)
    else:
        inlegalbert_embeddings = generate_and_save_inlegalbert_embeddings(
            inlegalbert_model, selected_chunks, str(inlegalbert_dir), inlegalbert_model_name
        )

    validate_inlegalbert_embeddings(inlegalbert_embeddings, selected_chunks)
    inlegalbert_client, inlegalbert_collection = create_inlegalbert_chromadb_collection(
        str(inlegalbert_chroma_dir), "inlegalbert_collection"
    )
    index_inlegalbert_embeddings_to_chromadb(
        inlegalbert_collection, selected_chunks, inlegalbert_embeddings, batch_size=500
    )
    validate_inlegalbert_chromadb_collection(inlegalbert_collection, selected_chunks)
    sample_inlegalbert_semantic_retrieval(inlegalbert_collection, inlegalbert_model)
    save_inlegalbert_configuration(
        inlegalbert_model, inlegalbert_model_name, inlegalbert_embeddings, str(inlegalbert_dir)
    )
