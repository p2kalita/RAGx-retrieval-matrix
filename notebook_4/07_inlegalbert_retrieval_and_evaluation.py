"""
=============================================================================
NyayaRAG Pilot Evaluation — Script 07: InLegalBERT Dense Retrieval & Evaluation
=============================================================================

This script executes Part Q and Part R of the NyayaRAG pilot evaluation pipeline:
  1. Inspects and validates the saved InLegalBERT configuration.
  2. Loads and memory-maps the InLegalBERT embedding matrix (30,521 x 768).
  3. Loads the InLegalBERT query encoder (law-ai/InLegalBERT).
  4. Validates retrieval on the first pilot question (RAW001).
  5. Retrieves Top-5 candidate chunks for all 10 pilot questions.
  6. Saves retrieval output and generates the annotation template.
  7. Loads and validates the annotated InLegalBERT pilot dataset.
  8. Computes Precision@5, Hit@5, MRR, and nDCG@5.
  9. Compares cumulative pilot performance across TF-IDF, BM25, E5, BGE,
     and InLegalBERT using rich formatted tables.
=============================================================================
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Ensure UTF-8 output on Windows if available
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(legacy_windows=False)

# =============================================================================
# 1. Directory and Path Configurations
# =============================================================================

BASE_DIR = Path("d:/_ResearchNotebooks/NyayaRAG")
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation" / "pilot"
PILOT_BENCHMARK_DIR = EVAL_DIR / "benchmark"
PILOT_RETRIEVAL_DIR = EVAL_DIR / "retrieval"
PILOT_ANNOTATION_DIR = EVAL_DIR / "annotation"

CHUNKS_FILE = DATA_DIR / "chunks_384_40.parquet"
PILOT_QUESTIONS_FILE = BASE_DIR / "benchmark" / "final" / "pilot_questions.csv"

INLEGALBERT_MODEL_DIR = MODELS_DIR / "inlegalbert"
INLEGALBERT_CONFIG_FILE = INLEGALBERT_MODEL_DIR / "inlegalbert_config.json"
INLEGALBERT_EMBEDDINGS_FILE = INLEGALBERT_MODEL_DIR / "inlegalbert_embeddings.npy"
INLEGALBERT_OUTPUT_FILE = PILOT_RETRIEVAL_DIR / "inlegalbert_pilot_retrieval.csv"
INLEGALBERT_TEMPLATE_FILE = PILOT_ANNOTATION_DIR / "inlegalbert_annotation_template.csv"
INLEGALBERT_ANNOTATION_FILE = PILOT_ANNOTATION_DIR / "inlegalbert_pilot_annotations.csv"

TOP_K = 5


# =============================================================================
# 2. Configuration & Resource Validation Functions
# =============================================================================

def load_inlegalbert_configuration(config_file: Path = INLEGALBERT_CONFIG_FILE) -> dict:
    """Load and display the InLegalBERT configuration."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    table = Table(title="InLegalBERT Model Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for k, v in config.items():
        table.add_row(str(k), str(v))

    console.print(table)
    return config


def load_and_validate_inlegalbert_embeddings(
    embeddings_file: Path,
    config: dict,
    df_chunks: pd.DataFrame
) -> np.ndarray:
    """Memory-map and validate InLegalBERT embedding matrix."""
    embeddings = np.load(embeddings_file, mmap_mode="r")

    table = Table(title="InLegalBERT Embedding Matrix Alignment", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Embedding Matrix Shape", str(embeddings.shape))
    table.add_row("Data Type", str(embeddings.dtype))
    table.add_row("Corpus Chunks Count", f"{len(df_chunks):,}")
    table.add_row("Config Dimension", str(config["embedding_dimension"]))

    console.print(table)

    assert embeddings.shape[0] == len(df_chunks), (
        "InLegalBERT embedding rows do not match chunk count."
    )
    assert embeddings.shape[1] == config["embedding_dimension"], (
        "InLegalBERT embedding dimension does not match config."
    )
    assert np.isfinite(embeddings).all(), "InLegalBERT embeddings contain non-finite values."

    console.print("[bold green][OK] InLegalBERT embeddings alignment validated successfully.[/bold green]")
    return embeddings


def load_inlegalbert_query_encoder(config: dict):
    """Load the InLegalBERT sentence transformer model."""
    from sentence_transformers import SentenceTransformer

    console.print(f"[cyan]Loading InLegalBERT encoder: {config['model_name']}...[/cyan]")
    model = SentenceTransformer(config["model_name"])
    model.max_seq_length = config["max_sequence_length"]

    console.print(f"[bold green][OK] InLegalBERT query encoder loaded (max_seq_length={model.max_seq_length}).[/bold green]")
    return model


# =============================================================================
# 3. Retrieval Function
# =============================================================================

def retrieve_inlegalbert(
    query: str,
    inlegalbert_model,
    inlegalbert_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    config: dict,
    top_k: int = TOP_K
) -> pd.DataFrame:
    """
    Retrieve top-k relevant chunks using normalized InLegalBERT cosine similarity.
    """
    query_embedding = inlegalbert_model.encode(
        str(query).strip(),
        normalize_embeddings=config.get("normalize_embeddings", True),
        convert_to_numpy=True,
        show_progress_bar=False
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    similarity_scores = np.asarray(inlegalbert_embeddings @ query_embedding, dtype=float)
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = df_chunks.iloc[top_indices].copy()
    results["InLegalBERT_Score"] = similarity_scores[top_indices]

    return results[[
        "global_chunk_id",
        "judgment_id",
        "case_title",
        "local_chunk_id",
        "InLegalBERT_Score",
        "chunk_text"
    ]].reset_index(drop=True)


# =============================================================================
# 4. Sample Validation and Complete Retrieval
# =============================================================================

def validate_inlegalbert_sample_retrieval(
    sample_question: str,
    inlegalbert_model,
    inlegalbert_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    config: dict,
    top_k: int = TOP_K
) -> pd.DataFrame:
    """Validate InLegalBERT retrieval on the sample question."""
    results = retrieve_inlegalbert(
        sample_question,
        inlegalbert_model,
        inlegalbert_embeddings,
        df_chunks,
        config,
        top_k=top_k
    )

    expected_columns = [
        "global_chunk_id", "judgment_id", "case_title",
        "local_chunk_id", "InLegalBERT_Score", "chunk_text"
    ]
    assert len(results) == top_k, f"Expected {top_k} results, got {len(results)}"
    assert results.columns.tolist() == expected_columns, "Columns mismatch"
    assert results["InLegalBERT_Score"].notna().all(), "Missing similarity scores"
    assert (
        results["InLegalBERT_Score"].to_numpy()[:-1] >= results["InLegalBERT_Score"].to_numpy()[1:]
    ).all(), "Scores not in descending order"
    assert results["InLegalBERT_Score"].between(-1, 1).all(), "Scores out of [-1, 1] range"

    console.print(Panel(
        f"[bold cyan]Validation Question:[/bold cyan]\n{sample_question}",
        title="[bold green]InLegalBERT Pilot Retrieval Validation (First Question)[/bold green]",
        box=box.ROUNDED
    ))

    for idx, row in results.iterrows():
        console.print(Panel(
            f"[bold]Rank {idx + 1}[/bold] | [cyan]InLegalBERT Cosine: {row['InLegalBERT_Score']:.4f}[/cyan]\n"
            f"[yellow]{row['case_title']}[/yellow] ({row['judgment_id']}) "
            f"Global Chunk: {row['global_chunk_id']} | Local: {row['local_chunk_id']}\n\n"
            f"{row['chunk_text'][:250]}...",
            box=box.MINIMAL
        ))

    return results


def retrieve_all_inlegalbert_pilot(
    df_pilot: pd.DataFrame,
    inlegalbert_model,
    inlegalbert_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    config: dict,
    top_k: int = TOP_K
) -> pd.DataFrame:
    """Retrieve top-k chunks for all pilot benchmark questions."""
    all_results = []
    for _, row in df_pilot.iterrows():
        q_id = row["Question_ID"]
        q_text = row["Question"]

        retrieved = retrieve_inlegalbert(
            q_text,
            inlegalbert_model,
            inlegalbert_embeddings,
            df_chunks,
            config,
            top_k=top_k
        )
        retrieved.insert(0, "Question", q_text)
        retrieved.insert(0, "Question_ID", q_id)
        retrieved["Rank"] = range(1, len(retrieved) + 1)
        all_results.append(retrieved)

    df_all = pd.concat(all_results, ignore_index=True)

    expected_rows = len(df_pilot) * top_k
    assert len(df_all) == expected_rows, f"Expected {expected_rows} rows, found {len(df_all)}"
    assert (df_all.groupby("Question_ID").size() == top_k).all(), "Incomplete question results"
    assert df_all["Question_ID"].nunique() == len(df_pilot), "Missing question IDs"
    assert df_all["InLegalBERT_Score"].notna().all(), "Missing scores"

    console.print(f"[bold green]✓ InLegalBERT pilot retrieval completed ({len(df_all)} chunks total).[/bold green]")
    return df_all


def save_inlegalbert_results_and_template(
    df_results: pd.DataFrame,
    output_file: Path = INLEGALBERT_OUTPUT_FILE,
    template_file: Path = INLEGALBERT_TEMPLATE_FILE
):
    """Save retrieval outputs and create annotation template."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False)
    console.print(f"[bold green]✓ Saved InLegalBERT pilot retrieval to {output_file}[/bold green]")

    template_file.parent.mkdir(parents=True, exist_ok=True)
    template_df = df_results.copy()
    template_df["Relevance"] = ""
    template_df["Relevance_Level"] = ""
    template_df["Comments"] = ""
    template_df.to_csv(template_file, index=False)
    console.print(f"[bold green]✓ Created InLegalBERT annotation template at {template_file}[/bold green]")


# =============================================================================
# 5. Annotation Loading & Metric Computation
# =============================================================================

def load_and_validate_inlegalbert_annotations(
    annotation_file: Path = INLEGALBERT_ANNOTATION_FILE,
    df_pilot: pd.DataFrame = None
) -> pd.DataFrame:
    """Load and validate human relevance annotations."""
    annotated_df = pd.read_csv(
        annotation_file,
        encoding="utf-8",
        encoding_errors="replace"
    )

    console.print(f"[cyan]Loaded InLegalBERT Annotations: {len(annotated_df)} rows[/cyan]")

    annotated_df["Relevance_Level"] = (
        annotated_df["Relevance_Level"]
        .astype(str)
        .str.strip()
        .replace({"Relevant": "Partially Relevant"})
    )

    annotated_df["Binary_Relevance"] = (annotated_df["Relevance"] > 0).astype(int)

    assert set(annotated_df["Binary_Relevance"].unique()).issubset({0, 1}), (
        "Binary relevance labels must only contain 0 and 1."
    )
    if df_pilot is not None:
        assert annotated_df["Question_ID"].nunique() == len(df_pilot), (
            "Annotated records do not cover all pilot questions."
        )

    return annotated_df


def compute_inlegalbert_metrics(annotated_df: pd.DataFrame, top_k: int = TOP_K):
    """
    Compute Precision@5, Hit@5, MRR, and nDCG@5 for InLegalBERT.
    Returns per-question dataframes and summary dataframe.
    """
    # Precision@5
    precision_per_q = (
        annotated_df.groupby("Question_ID")["Binary_Relevance"]
        .mean()
        .reset_index(name="Precision@5")
    )
    mean_precision = precision_per_q["Precision@5"].mean()

    # Hit@5
    hit_per_q = (
        annotated_df.groupby("Question_ID")["Binary_Relevance"]
        .max()
        .reset_index(name="Hit@5")
    )
    mean_hit = hit_per_q["Hit@5"].mean()

    # MRR
    def calc_rr(group):
        rel = group[group["Binary_Relevance"] == 1]
        if rel.empty:
            return 0.0
        return 1.0 / rel["Rank"].min()

    mrr_per_q = (
        annotated_df.groupby("Question_ID")
        .apply(calc_rr, include_groups=False)
        .reset_index(name="Reciprocal_Rank")
    )
    mean_mrr = mrr_per_q["Reciprocal_Rank"].mean()

    # nDCG@5
    ndcg_records = []
    score_col = "InLegalBERT_Score" if "InLegalBERT_Score" in annotated_df.columns else "Similarity_Score"

    for q_id, group in annotated_df.groupby("Question_ID"):
        grp = group.sort_values("Rank")
        true_rel = grp["Relevance"].to_numpy(dtype=float).reshape(1, -1)
        pred_scores = grp[score_col].to_numpy(dtype=float).reshape(1, -1)
        val = ndcg_score(true_rel, pred_scores, k=top_k)
        ndcg_records.append({"Question_ID": q_id, "nDCG@5": val})

    ndcg_per_q = pd.DataFrame(ndcg_records)
    mean_ndcg = ndcg_per_q["nDCG@5"].mean()

    # Validations
    assert precision_per_q["Precision@5"].between(0, 1).all()
    assert hit_per_q["Hit@5"].isin([0, 1]).all()
    assert mrr_per_q["Reciprocal_Rank"].between(0, 1).all()
    assert ndcg_per_q["nDCG@5"].between(0, 1).all()

    summary_df = pd.DataFrame({
        "Retriever": ["InLegalBERT"],
        "Precision@5": [mean_precision],
        "Hit@5": [mean_hit],
        "MRR": [mean_mrr],
        "nDCG@5": [mean_ndcg]
    })

    return precision_per_q, hit_per_q, mrr_per_q, ndcg_per_q, summary_df


def display_cumulative_performance(
    inlegalbert_summary: pd.DataFrame,
    tfidf_scores=(0.6400, 0.9000, 0.7167, 0.7744),
    bm25_scores=(0.6800, 0.9000, 0.7167, 0.7521),
    e5_scores=(0.5600, 0.8000, 0.7000, 0.6673),
    bge_scores=(0.6400, 0.8000, 0.7333, 0.7072)
):
    """Display cumulative pilot evaluation performance table."""
    inleg_p = inlegalbert_summary["Precision@5"].iloc[0]
    inleg_h = inlegalbert_summary["Hit@5"].iloc[0]
    inleg_m = inlegalbert_summary["MRR"].iloc[0]
    inleg_n = inlegalbert_summary["nDCG@5"].iloc[0]

    cum_df = pd.DataFrame({
        "Retriever": ["TF-IDF", "BM25", "E5", "BGE", "InLegalBERT"],
        "Type": ["Lexical", "Lexical", "Semantic", "Semantic", "Semantic"],
        "Precision@5": [tfidf_scores[0], bm25_scores[0], e5_scores[0], bge_scores[0], inleg_p],
        "Hit@5": [tfidf_scores[1], bm25_scores[1], e5_scores[1], bge_scores[1], inleg_h],
        "MRR": [tfidf_scores[2], bm25_scores[2], e5_scores[2], bge_scores[2], inleg_m],
        "nDCG@5": [tfidf_scores[3], bm25_scores[3], e5_scores[3], bge_scores[3], inleg_n]
    }).round(4)

    table = Table(
        title="Cumulative Pilot Retrieval Performance: TF-IDF, BM25, E5, BGE, and InLegalBERT",
        box=box.ROUNDED
    )
    table.add_column("Retriever", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Precision@5", justify="right", style="green")
    table.add_column("Hit@5", justify="right", style="green")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("nDCG@5", justify="right", style="green")

    for _, row in cum_df.iterrows():
        table.add_row(
            row["Retriever"],
            row["Type"],
            f"{row['Precision@5']:.4f}",
            f"{row['Hit@5']:.4f}",
            f"{row['MRR']:.4f}",
            f"{row['nDCG@5']:.4f}"
        )

    console.print(table)
    return cum_df


# =============================================================================
# 6. Main Execution Pipeline
# =============================================================================

def main():
    console.print(Panel(
        "[bold cyan]NyayaRAG Pilot Evaluation — Script 07[/bold cyan]\n"
        "[white]InLegalBERT Domain-Specific Dense Retrieval & Pilot Evaluation[/white]",
        box=box.DOUBLE
    ))

    # Load corpus chunks & pilot benchmark
    df_chunks = pd.read_parquet(CHUNKS_FILE)
    df_pilot = pd.read_csv(PILOT_QUESTIONS_FILE)
    console.print(f"[cyan]Loaded {len(df_chunks):,} corpus chunks and {len(df_pilot)} pilot questions.[/cyan]")

    # Load InLegalBERT config & embeddings
    config = load_inlegalbert_configuration(INLEGALBERT_CONFIG_FILE)
    embeddings = load_and_validate_inlegalbert_embeddings(
        INLEGALBERT_EMBEDDINGS_FILE,
        config,
        df_chunks
    )

    # Load query encoder
    inlegalbert_model = load_inlegalbert_query_encoder(config)

    # Validate on sample question
    sample_question = df_pilot.iloc[0]["Question"]
    validate_inlegalbert_sample_retrieval(
        sample_question,
        inlegalbert_model,
        embeddings,
        df_chunks,
        config,
        top_k=TOP_K
    )

    # Retrieve all pilot questions
    df_results = retrieve_all_inlegalbert_pilot(
        df_pilot,
        inlegalbert_model,
        embeddings,
        df_chunks,
        config,
        top_k=TOP_K
    )

    # Save results & annotation template
    save_inlegalbert_results_and_template(
        df_results,
        INLEGALBERT_OUTPUT_FILE,
        INLEGALBERT_TEMPLATE_FILE
    )

    # Load & evaluate human annotations
    annotated_df = load_and_validate_inlegalbert_annotations(
        INLEGALBERT_ANNOTATION_FILE,
        df_pilot
    )

    p_per_q, h_per_q, m_per_q, n_per_q, inlegalbert_summary = compute_inlegalbert_metrics(
        annotated_df,
        top_k=TOP_K
    )

    # Print summary table
    table = Table(title="InLegalBERT Pilot Retrieval Metrics", box=box.ROUNDED)
    table.add_column("Retriever", style="cyan")
    table.add_column("Precision@5", justify="right", style="green")
    table.add_column("Hit@5", justify="right", style="green")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("nDCG@5", justify="right", style="green")

    table.add_row(
        "InLegalBERT",
        f"{inlegalbert_summary['Precision@5'].iloc[0]:.4f}",
        f"{inlegalbert_summary['Hit@5'].iloc[0]:.4f}",
        f"{inlegalbert_summary['MRR'].iloc[0]:.4f}",
        f"{inlegalbert_summary['nDCG@5'].iloc[0]:.4f}"
    )
    console.print(table)

    # Cumulative performance comparison
    display_cumulative_performance(inlegalbert_summary)

    console.print(Panel(
        "[bold green]✓ InLegalBERT dense retrieval and pilot evaluation finished successfully![/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
