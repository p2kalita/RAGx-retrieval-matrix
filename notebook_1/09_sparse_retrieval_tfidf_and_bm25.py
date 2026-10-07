"""
Section 8: Retrieval Index Preparation
(Part 1: Sparse Retrieval — TF-IDF and BM25 Index Preparation)

Following the selection and validation of the 384-word chunk configuration with approximately 10% overlap (chunk_384_40), the selected chunk corpus is prepared for retrieval experiments. This stage constructs the retrieval indexes required by the evaluated retrieval models.

The same validated chunk corpus is used for all retrieval approaches to ensure a fair comparison. Sparse retrieval models (TF-IDF and BM25) and dense retrieval models (E5, BGE, LegalBERT, and InLegalBERT) are prepared independently using identical chunk identifiers and metadata. Model-specific indexes are generated and saved for subsequent retrieval evaluation.
"""

import os
from pathlib import Path
import random
import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


def load_and_validate_selected_chunks(chunk_file=r"D:\_ResearchNotebooks\NyayaRAG\data\processed\chunks_384_40.parquet"):
    """
    8.2 Load the Selected Chunk Corpus

    The validated chunk corpus generated using the selected chunk_384_40 configuration is loaded from disk. This corpus serves as the common retrieval corpus for all subsequent retrieval models, ensuring that every model is evaluated using identical textual content and metadata.

    Observation:
    The selected chunk corpus was successfully loaded from disk. The corpus consists of 30,521 validated chunks represented by nine metadata fields, including unique chunk identifiers, judgment identifiers, case titles, chunk text, and structural statistics. These metadata attributes provide the information required for sparse and dense retrieval while preserving traceability between retrieved chunks and their source judgments.

    Chunk Corpus Validation:
    The selected chunk corpus was successfully loaded and validated. The corpus contains 30,521 unique chunks, each associated with a unique global chunk identifier and complete metadata. No missing or empty chunk text was identified, and all chunks retain their corresponding judgment identifiers and case titles. These results confirm that the selected corpus is structurally complete and suitable for constructing both sparse and dense retrieval indexes.
    """
    selected_chunks = pd.read_parquet(chunk_file)

    print("=" * 70)
    print("Selected Chunk Corpus")
    print("=" * 70)

    print(f"Total Chunks : {len(selected_chunks):,}")
    print(f"Columns      : {selected_chunks.shape[1]}")

    print("\nColumns")
    print("-" * 70)

    for column in selected_chunks.columns:
        print(column)

    print("=" * 70)
    print("Chunk Corpus Validation")
    print("=" * 70)

    print(f"Total Chunks            : {len(selected_chunks):,}")
    print(f"Unique Chunk IDs        : {selected_chunks['global_chunk_id'].nunique():,}")
    print(f"Missing Chunk Text      : {selected_chunks['chunk_text'].isna().sum():,}")
    print(f"Missing Judgment IDs    : {selected_chunks['judgment_id'].isna().sum():,}")
    print(f"Missing Case Titles     : {selected_chunks['case_title'].isna().sum():,}")
    print(f"Empty Chunk Text        : {selected_chunks['chunk_text'].fillna('').str.strip().eq('').sum():,}")

    return selected_chunks


def build_tfidf_representation(selected_chunks):
    """
    8.4 TF-IDF Index Preparation

    Term Frequency–Inverse Document Frequency (TF-IDF) is a classical sparse retrieval approach that represents each chunk as a weighted vector based on term importance. Frequently occurring words within a chunk receive higher weights, while terms that appear across many chunks receive lower weights. This representation emphasizes discriminative vocabulary and serves as a keyword-based baseline for evaluating retrieval performance.

    The TF-IDF vectorizer is fitted using the validated chunk corpus. The resulting sparse document-term matrix is retained for retrieval experiments and saved for reproducibility.

    Observation:
    The TF-IDF vectorizer was successfully fitted on the selected corpus of 30,521 validated legal text chunks using a minimum document frequency of 5. This produced a sparse document-term matrix comprising 230,196 unigram and bigram features. Compared with lower document frequency thresholds, the selected configuration substantially reduced the vocabulary by eliminating extremely rare terms while preserving the lexical information required for retrieval. The resulting TF-IDF representation provides an efficient and reproducible baseline for keyword-based retrieval experiments.
    """
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=5
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(
        selected_chunks["chunk_text"]
    )

    print("=" * 70)
    print("TF-IDF Index")
    print("=" * 70)

    print(f"Vocabulary Size : {len(tfidf_vectorizer.vocabulary_):,}")
    print(f"Matrix Shape    : {tfidf_matrix.shape}")
    print(f"Non-zero Values : {tfidf_matrix.nnz:,}")

    return tfidf_vectorizer, tfidf_matrix


def inspect_tfidf_vocabulary(tfidf_vectorizer):
    """
    8.4.1 Inspection of the TF-IDF Vocabulary

    Before saving the fitted TF-IDF representation, the generated vocabulary is inspected to verify that it contains meaningful legal terminology and common legal phrases. This step helps identify any preprocessing artefacts or unexpected tokens that could adversely affect retrieval quality.

    Observation:
    Random inspection of the TF-IDF vocabulary confirmed that the generated features consist predominantly of meaningful legal terminology, statutory references, judicial phrases, and case-specific expressions. Although the vocabulary also contains names and citation-related terms, these are naturally present in judicial documents and may contribute to lexical retrieval. The inspection did not reveal any significant preprocessing artefacts, indicating that the fitted TF-IDF representation appropriately captures the linguistic characteristics of the Supreme Court judgment corpus.
    """
    feature_names = tfidf_vectorizer.get_feature_names_out()

    print("=" * 70)
    print("TF-IDF Vocabulary Inspection")
    print("=" * 70)

    print(f"Total Features : {len(feature_names):,}")

    print("\nFirst 50 Features")
    print("-" * 70)
    print(feature_names[:50])

    print("=" * 70)
    print("Random TF-IDF Vocabulary Sample")
    print("=" * 70)

    random_features = random.sample(list(feature_names), 50)
    print(random_features)


def inspect_tfidf_weights(tfidf_vectorizer, tfidf_matrix, sample_chunk=0):
    """
    8.4.2 Inspection of TF-IDF Weights

    To verify the correctness of the TF-IDF representation, a sample legal text chunk is examined together with its highest-weighted terms. The inspection confirms that legally significant words and phrases receive greater importance than common vocabulary.

    Observation:
    Inspection of the highest-weighted TF-IDF terms confirmed that the vectorizer assigns greater importance to discriminative legal words and phrases, including party names, references to accused persons, offences, and judicial institutions. Although a small number of common function words also appeared among the weighted terms, the overall representation appropriately emphasizes chunk-specific legal vocabulary. This indicates that the TF-IDF weighting scheme is functioning as expected and provides a suitable lexical baseline for retrieval experiments.
    """
    feature_names = tfidf_vectorizer.get_feature_names_out()

    row = tfidf_matrix[sample_chunk]
    indices = row.indices
    weights = row.data

    top_indices = weights.argsort()[::-1][:20]

    print("=" * 70)
    print("Top TF-IDF Terms")
    print("=" * 70)

    for idx in top_indices:
        print(f"{feature_names[indices[idx]]:<35} {weights[idx]:.4f}")


def save_tfidf_artefacts(tfidf_vectorizer, tfidf_matrix, tfidf_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\tfidf"):
    """
    Save TF-IDF Artefacts to disk
    """
    os.makedirs(tfidf_dir, exist_ok=True)

    # Save vectorizer
    joblib.dump(
        tfidf_vectorizer,
        os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
    )

    # Save sparse matrix
    sparse.save_npz(
        os.path.join(tfidf_dir, "tfidf_matrix.npz"),
        tfidf_matrix
    )

    print("=" * 70)
    print("TF-IDF Artefacts Saved Successfully")
    print("=" * 70)
    print(f"Directory : {tfidf_dir}")
    print("Files:")
    print("  • tfidf_vectorizer.pkl")
    print("  • tfidf_matrix.npz")


def validate_saved_tfidf_artefacts(tfidf_dir, selected_chunks):
    """
    8.4.4 Validation of Saved TF-IDF Artefacts

    The saved TF-IDF vectorizer and sparse document-term matrix are reloaded from disk to verify that the artefacts were stored correctly. The reloaded vocabulary size, matrix dimensions, and number of non-zero values are compared with the original representation.

    Observations:
    The saved TF-IDF vectorizer and sparse matrix were successfully reloaded from disk. The reloaded representation retained the original vocabulary size of 230,196 features, matrix dimensions of 30,521 × 230,196, and 12,406,627 non-zero values. The number of matrix rows also matched the number of corpus chunks, confirming correct alignment and successful persistence of the TF-IDF retrieval artefacts.
    """
    loaded_tfidf_vectorizer = joblib.load(
        os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
    )

    loaded_tfidf_matrix = sparse.load_npz(
        os.path.join(tfidf_dir, "tfidf_matrix.npz")
    )

    print("=" * 70)
    print("Saved TF-IDF Artefact Validation")
    print("=" * 70)

    print(f"Vocabulary Size : {len(loaded_tfidf_vectorizer.vocabulary_):,}")
    print(f"Matrix Shape    : {loaded_tfidf_matrix.shape}")
    print(f"Non-zero Values : {loaded_tfidf_matrix.nnz:,}")
    print(f"Chunk Alignment : {loaded_tfidf_matrix.shape[0] == len(selected_chunks)}")


def build_bm25_index(selected_chunks):
    """
    8.5 BM25 Index Preparation

    Best Matching 25 (BM25) is a probabilistic keyword-based retrieval model widely used in information retrieval systems. Unlike TF-IDF, BM25 incorporates document length normalization and term frequency saturation, allowing it to rank documents more effectively when query terms appear multiple times.

    The BM25 index is constructed using the validated chunk corpus. Each chunk is tokenized into individual terms, and the resulting tokenized corpus is used to build the BM25 retrieval model. The fitted BM25 index is subsequently saved for retrieval experiments.
    """
    tokenized_corpus = [
        chunk.lower().split()
        for chunk in selected_chunks["chunk_text"]
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print("=" * 70)
    print("BM25 Index")
    print("=" * 70)

    print(f"Total Documents : {len(tokenized_corpus):,}")
    print(f"Average Document Length : {sum(len(doc) for doc in tokenized_corpus)/len(tokenized_corpus):.2f} words")

    return bm25, tokenized_corpus


def inspect_bm25_retrieval(bm25, selected_chunks, query="murder conviction under section 302 ipc"):
    """
    8.5.1 Inspection of BM25 Retrieval

    To verify that the BM25 index has been constructed correctly, a sample legal query is executed against the indexed corpus. The highest-ranked chunks are inspected to confirm that the retrieval model returns legally relevant text from the corpus.

    Observation:
    Inspection of the BM25 retrieval results confirmed that the index successfully retrieved legally relevant Supreme Court judgment chunks for the sample query. The highest-ranked results contained judicial discussions relating to conviction under Section 302 IPC, murder, sentencing, and related criminal law principles. The retrieval scores decreased progressively across the ranked results, indicating appropriate document ranking. These observations confirm that the BM25 index has been correctly constructed and is suitable for subsequent retrieval evaluation.
    """
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    top_k = 5
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    print("=" * 70)
    print("BM25 Retrieval Inspection")
    print("=" * 70)

    for rank, idx in enumerate(top_indices, start=1):
        print(f"\nRank {rank}")
        print("-" * 70)
        print(f"Case Title : {selected_chunks.iloc[idx]['case_title']}")
        print(f"Chunk ID   : {selected_chunks.iloc[idx]['global_chunk_id']}")
        print(f"Score      : {scores[idx]:.4f}")

        preview = selected_chunks.iloc[idx]["chunk_text"][:300]
        print(f"Preview    : {preview}\n")


def inspect_bm25_corpus(tokenized_corpus):
    """
    8.5.2 Inspection of the Indexed BM25 Corpus

    The tokenized corpus used for BM25 indexing is inspected to verify that all chunks have been successfully indexed. Basic corpus statistics are computed to confirm the total number of indexed documents and the average document length.

    Observation:
    The BM25 index was successfully constructed using 30,521 tokenized legal text chunks comprising 11,089,911 indexed tokens. The average document length of 363.35 words per chunk is consistent with the selected chunk_384_40 configuration, confirming that the complete corpus has been indexed correctly and is ready for probabilistic lexical retrieval experiments.
    """
    total_tokens = sum(len(doc) for doc in tokenized_corpus)
    average_tokens = total_tokens / len(tokenized_corpus)

    print("=" * 70)
    print("BM25 Indexed Corpus Statistics")
    print("=" * 70)

    print(f"Indexed Documents      : {len(tokenized_corpus):,}")
    print(f"Total Tokens Indexed   : {total_tokens:,}")
    print(f"Average Tokens/Chunk   : {average_tokens:.2f}")


def save_bm25_artefacts(bm25, tokenized_corpus, bm25_dir=r"D:\_ResearchNotebooks\NyayaRAG\models\bm25"):
    """
    Save BM25 Artefacts
    """
    os.makedirs(bm25_dir, exist_ok=True)

    # Save BM25 model
    joblib.dump(
        bm25,
        os.path.join(bm25_dir, "bm25.pkl")
    )

    # Save tokenized corpus
    joblib.dump(
        tokenized_corpus,
        os.path.join(bm25_dir, "tokenized_corpus.pkl")
    )

    print("=" * 70)
    print("BM25 Artefacts Saved Successfully")
    print("=" * 70)
    print(f"Directory : {bm25_dir}")
    print("Files:")
    print("  • bm25.pkl")
    print("  • tokenized_corpus.pkl")


def validate_saved_bm25_artefacts(bm25_dir, selected_chunks, query="murder conviction under section 302 ipc"):
    """
    8.5.4 Reloading and Validation of the Saved BM25 Artefacts

    The saved BM25 retrieval model and tokenized corpus are reloaded from disk to verify that they have been stored correctly. A sample legal query is executed using the reloaded model to confirm that retrieval performance remains unchanged after persistence.

    Observation:
    The saved BM25 retrieval model and tokenized corpus were successfully reloaded from persistent storage. Validation using the same sample legal query produced identical top-ranked judgment chunks and corresponding BM25 relevance scores as the original model. The consistency of both document rankings and retrieval scores confirms that the BM25 artefacts were stored and restored correctly without any loss of information, ensuring reproducibility of subsequent retrieval experiments.
    """
    bm25_loaded = joblib.load(os.path.join(bm25_dir, "bm25.pkl"))
    tokenized_corpus_loaded = joblib.load(
        os.path.join(bm25_dir, "tokenized_corpus.pkl")
    )

    print("=" * 70)
    print("BM25 Artefacts Reloaded Successfully")
    print("=" * 70)

    print(f"Indexed Documents    : {len(tokenized_corpus_loaded):,}")
    print(f"Average Doc Length   : {bm25_loaded.avgdl:.2f} words")

    scores = bm25_loaded.get_scores(query.lower().split())

    top_k = 5
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    print("=" * 70)
    print("Reloaded BM25 Retrieval Validation")
    print("=" * 70)

    for rank, idx in enumerate(top_indices, start=1):
        print(f"\nRank {rank}")
        print("-" * 70)
        print(f"Case Title : {selected_chunks.iloc[idx]['case_title']}")
        print(f"Chunk ID   : {selected_chunks.iloc[idx]['global_chunk_id']}")
        print(f"Score      : {scores[idx]:.4f}")


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    chunk_file = PROCESSED_DIR / "chunks_384_40.parquet"
    models_dir = PROJECT_DIR / "models"
    tfidf_dir = models_dir / "tfidf"
    bm25_dir = models_dir / "bm25"

    selected_chunks = load_and_validate_selected_chunks(str(chunk_file))

    # TF-IDF Pipeline
    tfidf_vectorizer, tfidf_matrix = build_tfidf_representation(selected_chunks)
    inspect_tfidf_vocabulary(tfidf_vectorizer)
    inspect_tfidf_weights(tfidf_vectorizer, tfidf_matrix, sample_chunk=0)
    save_tfidf_artefacts(tfidf_vectorizer, tfidf_matrix, str(tfidf_dir))
    validate_saved_tfidf_artefacts(str(tfidf_dir), selected_chunks)

    # BM25 Pipeline
    bm25, tokenized_corpus = build_bm25_index(selected_chunks)
    inspect_bm25_retrieval(bm25, selected_chunks)
    inspect_bm25_corpus(tokenized_corpus)
    save_bm25_artefacts(bm25, tokenized_corpus, str(bm25_dir))
    validate_saved_bm25_artefacts(str(bm25_dir), selected_chunks)
