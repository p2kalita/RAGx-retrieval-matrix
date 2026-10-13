"""
Section 5: E5 Dense Semantic Retrieval and Evaluation
======================================================

Covers Part M (E5 Semantic Retrieval Evaluation) and Part N (E5 Retrieval Evaluation Metrics)
from Notebook 4 (Cells 133 to 172).

This module inspects the trained E5 model configuration, loads and validates the 768-dimensional
E5 chunk embedding matrix, initializes the sentence transformer query encoder with the required
'query: ' prefix, executes dense semantic retrieval across the pilot benchmark, saves the outputs,
creates the annotation template, evaluates annotated performance metrics (Precision@5, Hit@5,
MRR, nDCG@5), and updates the cumulative multi-model performance table.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure UTF-8 output encoding across consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Default Project Paths
DEFAULT_PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")


def print_banner(title: str, subtitle: Optional[str] = None, console: Optional[Console] = None) -> None:
    """Print an eye-catching Rich section banner."""
    if console is None:
        console = Console(legacy_windows=False)
    content = f"[bold bright_white]{title}[/bold bright_white]"
    if subtitle:
        content += f"\n[dim cyan]{subtitle}[/dim cyan]"
    console.print()
    console.print(
        Panel(
            content,
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(0, 2),
        )
    )


def inspect_e5_configuration(config_file: Path, console: Optional[Console] = None) -> Dict[str, Any]:
    """
    Load and display saved E5 configuration.
    (Notebook Cells 133 to 134)
    """
    if console is None:
        console = Console(legacy_windows=False)

    if not config_file.exists():
        raise FileNotFoundError(f"E5 configuration missing at: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    table = Table(
        title="[bold bright_cyan]E5 Retrieval Model Configuration[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="white")

    for k, v in config.items():
        table.add_row(str(k), str(v))

    console.print(table)
    return config


def load_e5_embeddings(
    embeddings_file: Path,
    df_chunks: pd.DataFrame,
    console: Optional[Console] = None,
) -> np.ndarray:
    """
    Load and validate dense E5 embedding matrix from disk.
    (Notebook Cells 135 to 136)
    """
    if console is None:
        console = Console(legacy_windows=False)

    if not embeddings_file.exists():
        raise FileNotFoundError(f"E5 embeddings missing at: {embeddings_file}")

    e5_embeddings = np.load(embeddings_file)

    # Validations from Cell 136
    assert e5_embeddings.ndim == 2, "Embeddings must be 2-dimensional."
    assert e5_embeddings.shape[0] == len(df_chunks), (
        f"Row count mismatch: {e5_embeddings.shape[0]} embeddings vs {len(df_chunks)} chunks."
    )
    assert not np.isnan(e5_embeddings).any(), "NaN values found in E5 embeddings."

    norms = np.linalg.norm(e5_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit-normalized for cosine similarity."

    table = Table(box=box.ROUNDED, show_header=False, border_style="bright_blue")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Matrix Dimensions", f"{e5_embeddings.shape[0]:,} rows × {e5_embeddings.shape[1]} dimensions")
    table.add_row("Data Type", str(e5_embeddings.dtype))
    table.add_row("L2 Normalization", "[bold green][OK] Unit L2-norm verified across all chunks[/bold green]")

    console.print(
        Panel(
            table,
            title="[bold bright_cyan] E5 Embedding Matrix Validation [/bold bright_cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return e5_embeddings


def load_e5_query_encoder(
    model_name: str = "intfloat/e5-base-v2",
    console: Optional[Console] = None,
) -> SentenceTransformer:
    """
    Initialize SentenceTransformer E5 model for query encoding.
    (Notebook Cells 137 to 138)
    """
    if console is None:
        console = Console(legacy_windows=False)

    e5_model = SentenceTransformer(model_name)
    console.print(f"[bold green]Loaded E5 Query Encoder: {model_name}[/bold green]")
    return e5_model


def retrieve_e5(
    query: str,
    e5_model: SentenceTransformer,
    e5_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve Top-k chunks using dense E5 semantic cosine similarity.
    (Notebook Cell 142)
    """
    # E5 query prefix requirement
    formatted_query = f"query: {query.strip()}"
    query_vector = e5_model.encode(
        formatted_query,
        normalize_embeddings=True,
    )

    similarity_scores = np.dot(e5_embeddings, query_vector)
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = df_chunks.iloc[top_indices].copy()
    results["E5_Score"] = similarity_scores[top_indices]
    results["Similarity_Score"] = similarity_scores[top_indices]

    columns_to_return = [
        "global_chunk_id",
        "judgment_id",
        "case_title",
        "local_chunk_id",
        "E5_Score",
        "Similarity_Score",
        "chunk_text",
    ]
    existing_cols = [c for c in columns_to_return if c in results.columns]
    return results[existing_cols].reset_index(drop=True)


def display_retrieved_chunks(
    results_df: pd.DataFrame,
    query_str: str,
    display_k: int = 3,
    console: Optional[Console] = None,
) -> None:
    """
    Display top retrieved chunks formatted with Rich for manual inspection.
    (Notebook Cell 148)
    """
    if console is None:
        console = Console(legacy_windows=False)

    console.print()
    console.print(
        Panel(
            f"[bold white]Inspected Query:[/bold white] [bold yellow]\"{query_str}\"[/bold yellow]",
            title="[bold cyan] E5 Dense Semantic Retrieval Inspection [/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    for i in range(min(display_k, len(results_df))):
        row = results_df.iloc[i]
        case_title = str(row.get("case_title", "Unknown"))
        score_val = float(row.get("E5_Score", 0.0))
        chunk_id = str(row.get("global_chunk_id", i + 1))
        local_id = str(row.get("local_chunk_id", 1))
        j_id = str(row.get("judgment_id", "Unknown"))
        passage = str(row.get("chunk_text", ""))[:450] + "..."

        chunk_panel = Panel(
            f"[bold cyan]Case Title:[/bold cyan] [bold white]{case_title}[/bold white]\n"
            f"[dim]Judgment ID: {j_id} | Global Chunk: {chunk_id} | Local Chunk: {local_id}[/dim]\n\n"
            f"[italic white]{passage}[/italic white]",
            title=f"[bold green]Rank {i+1}[/bold green] — [bold magenta]E5 Cosine Similarity: {score_val:.4f}[/bold magenta]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 2),
        )
        console.print(chunk_panel)


def run_pilot_e5_retrieval(
    df_pilot: pd.DataFrame,
    e5_model: SentenceTransformer,
    e5_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    top_k: int = 5,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Retrieve Top-k results for all questions in the pilot benchmark using E5.
    (Notebook Cells 150 to 153)
    """
    if console is None:
        console = Console(legacy_windows=False)

    all_e5_results = []
    for _, row in df_pilot.iterrows():
        question_id = row["Question_ID"]
        question = row["Question"]

        retrieved = retrieve_e5(
            question,
            e5_model=e5_model,
            e5_embeddings=e5_embeddings,
            df_chunks=df_chunks,
            top_k=top_k,
        )

        retrieved.insert(0, "Question", question)
        retrieved.insert(0, "Question_ID", question_id)
        retrieved["Rank"] = range(1, len(retrieved) + 1)
        all_e5_results.append(retrieved)

    df_e5_results = pd.concat(all_e5_results, ignore_index=True)

    table = Table(
        title=f"[bold bright_magenta]Consolidated E5 Pilot Retrieval ({len(df_e5_results)} Chunks)[/bold bright_magenta]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Q ID", style="bold cyan", justify="center", width=8)
    table.add_column("Rank", style="bold yellow", justify="center", width=6)
    table.add_column("E5 Cosine", style="bold green", justify="right", width=12)
    table.add_column("Chunk ID", style="dim cyan", justify="center", width=10)
    table.add_column("Case Title", style="bold white", width=40)

    for _, row in df_e5_results.head(10).iterrows():
        title = str(row.get("case_title", ""))
        if len(title) > 38:
            title = title[:35] + "..."
        table.add_row(
            str(row["Question_ID"]),
            str(row["Rank"]),
            f"{float(row['E5_Score']):.4f}",
            str(row.get("global_chunk_id", "")),
            title,
        )

    console.print(table)
    return df_e5_results


def save_e5_pilot_results(
    df_e5_results: pd.DataFrame,
    output_file: Path,
    console: Optional[Console] = None,
) -> None:
    """Save raw E5 retrieval outputs to CSV (Notebook Cells 154 to 155)."""
    if console is None:
        console = Console(legacy_windows=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_e5_results.to_csv(output_file, index=False)

    console.print(
        Panel(
            f"[bold green]Saved E5 Pilot Retrieval Results[/bold green]\n"
            f"[white]Destination: {output_file}[/white]\n"
            f"[dim]Total records: {len(df_e5_results):,} rows[/dim]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def create_and_save_e5_annotation_template(
    df_e5_results: pd.DataFrame,
    annotation_file: Path,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """Generate and persist the E5 manual annotation template (Notebook Cells 156 to 158)."""
    if console is None:
        console = Console(legacy_windows=False)

    annotation_df = df_e5_results.copy()
    annotation_df["Relevance"] = ""
    annotation_df["Relevance_Level"] = ""
    annotation_df["Comments"] = ""

    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(annotation_file, index=False)

    console.print(
        Panel(
            f"[bold cyan]Created E5 Annotation Template[/bold cyan]\n"
            f"[white]Destination: {annotation_file}[/white]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return annotation_df


def load_and_validate_e5_annotations(
    annotation_file: Path,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Robustly load annotated E5 results handling character encodings.
    (Notebook Cells 160 to 164)
    """
    if console is None:
        console = Console(legacy_windows=False)

    try:
        e5_annotated_df = pd.read_csv(annotation_file, encoding="utf-8")
    except UnicodeDecodeError:
        e5_annotated_df = pd.read_csv(annotation_file, encoding="utf-8", encoding_errors="replace")

    assert len(e5_annotated_df) == 50, f"Expected 50 records, found {len(e5_annotated_df)}"
    console.print(f"[bold green]Loaded E5 Annotations ({len(e5_annotated_df)} records)[/bold green]")
    return e5_annotated_df


def compute_e5_metrics(
    annotated_df: pd.DataFrame,
    top_k: int = 5,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """
    Compute Precision@5, Hit@5, MRR, and nDCG@5 for E5.
    (Notebook Cells 165 to 171)
    """
    if console is None:
        console = Console(legacy_windows=False)

    df = annotated_df.copy()
    df["Binary_Relevance"] = (df["Relevance"] > 0).astype(int)

    precision_df = df.groupby("Question_ID")["Binary_Relevance"].mean().reset_index(name="Precision@5")
    mean_precision = float(precision_df["Precision@5"].mean())

    hit_df = df.groupby("Question_ID")["Binary_Relevance"].max().reset_index(name="Hit@5")
    mean_hit = float(hit_df["Hit@5"].mean())

    def reciprocal_rank(group: pd.DataFrame) -> float:
        rel_rows = group[group["Binary_Relevance"] == 1]
        if rel_rows.empty:
            return 0.0
        return 1.0 / float(rel_rows["Rank"].min())

    mrr_df = df.groupby("Question_ID").apply(reciprocal_rank, include_groups=False).reset_index(name="Reciprocal_Rank")
    mean_mrr = float(mrr_df["Reciprocal_Rank"].mean())

    score_col = "E5_Score" if "E5_Score" in df.columns else "Similarity_Score"

    ndcg_scores = []
    for q_id, group in df.groupby("Question_ID"):
        true_rel = group["Relevance"].to_numpy().reshape(1, -1)
        pred_scores = group[score_col].to_numpy().reshape(1, -1)
        score = float(ndcg_score(true_rel, pred_scores, k=top_k))
        ndcg_scores.append({"Question_ID": q_id, "nDCG@5": score})
    ndcg_df = pd.DataFrame(ndcg_scores)
    mean_ndcg = float(ndcg_df["nDCG@5"].mean())

    summary_df = pd.DataFrame({
        "Retriever": ["E5"],
        "Precision@5": [mean_precision],
        "Hit@5": [mean_hit],
        "MRR": [mean_mrr],
        "nDCG@5": [mean_ndcg],
    })

    table = Table(
        title="[bold bright_green]E5 Pilot Retrieval Evaluation Metrics Summary[/bold bright_green]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_green",
    )
    table.add_column("Retriever", style="bold cyan")
    table.add_column("Precision@5", style="bold white", justify="right")
    table.add_column("Hit@5", style="bold yellow", justify="right")
    table.add_column("MRR", style="bold magenta", justify="right")
    table.add_column("nDCG@5", style="bold green", justify="right")

    table.add_row("E5", f"{mean_precision:.4f}", f"{mean_hit:.4f}", f"{mean_mrr:.4f}", f"{mean_ndcg:.4f}")
    console.print(table)

    return {
        "annotated_df": df,
        "precision_df": precision_df,
        "hit_df": hit_df,
        "mrr_df": mrr_df,
        "ndcg_df": ndcg_df,
        "summary": summary_df,
    }


def update_cumulative_results(
    previous_cumulative: pd.DataFrame,
    e5_summary: pd.DataFrame,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Update cumulative performance table (TF-IDF, BM25, E5).
    (Notebook Cell 172)
    """
    if console is None:
        console = Console(legacy_windows=False)

    cumulative_df = pd.concat([previous_cumulative, e5_summary], ignore_index=True)

    table = Table(
        title="[bold bright_cyan]Cumulative Pilot Retrieval Performance: Lexical vs Dense Semantic[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Retriever", style="bold cyan")
    table.add_column("Precision@5", justify="right")
    table.add_column("Hit@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("nDCG@5", justify="right")

    for _, r in cumulative_df.iterrows():
        table.add_row(
            str(r["Retriever"]),
            f"{r['Precision@5']:.4f}",
            f"{r['Hit@5']:.4f}",
            f"{r['MRR']:.4f}",
            f"{r['nDCG@5']:.4f}",
        )

    console.print(table)
    return cumulative_df


def run_e5_pipeline() -> Dict[str, Any]:
    """Execute complete E5 dense retrieval and evaluation pipeline."""
    console = Console(legacy_windows=False)
    print_banner(
        "E5 Dense Semantic Retrieval Evaluation",
        "Part M-N: Embeddings, Dense Vector Querying, Metrics, and Cumulative Comparison",
        console=console,
    )

    models_dir = DEFAULT_PROJECT_DIR / "models" / "e5"
    data_dir = DEFAULT_PROJECT_DIR / "data" / "processed"
    benchmark_file = DEFAULT_PROJECT_DIR / "benchmark" / "final" / "pilot_questions.csv"
    retrieval_out = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "retrieval" / "e5_pilot_retrieval.csv"
    annotation_out = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "annotation" / "e5_annotation_template.csv"
    annotated_file = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "annotation" / "e5_pilot_annotations.csv"

    df_chunks = pd.read_parquet(data_dir / "chunks_384_40.parquet")
    df_pilot = pd.read_csv(benchmark_file)

    config = inspect_e5_configuration(models_dir / "e5_config.json", console=console)
    embeddings = load_e5_embeddings(models_dir / "e5_embeddings.npy", df_chunks, console=console)
    e5_model = load_e5_query_encoder("intfloat/e5-base-v2", console=console)

    sample_q = df_pilot.iloc[0]["Question"]
    sample_res = retrieve_e5(sample_q, e5_model, embeddings, df_chunks, top_k=5)
    display_retrieved_chunks(sample_res, sample_q, display_k=3, console=console)

    pilot_res = run_pilot_e5_retrieval(df_pilot, e5_model, embeddings, df_chunks, top_k=5, console=console)
    save_e5_pilot_results(pilot_res, retrieval_out, console=console)
    create_and_save_e5_annotation_template(pilot_res, annotation_out, console=console)

    annotated_df = load_and_validate_e5_annotations(annotated_file, console=console)
    e5_eval = compute_e5_metrics(annotated_df, top_k=5, console=console)

    # Previous baseline comparison (hardcoded baseline numbers from TF-IDF & BM25 evaluations)
    prev_df = pd.DataFrame([
        {"Retriever": "TF-IDF", "Precision@5": 0.6400, "Hit@5": 0.9000, "MRR": 0.7167, "nDCG@5": 0.7744},
        {"Retriever": "BM25", "Precision@5": 0.6800, "Hit@5": 0.9000, "MRR": 0.7167, "nDCG@5": 0.7521},
    ])
    cumulative_df = update_cumulative_results(prev_df, e5_eval["summary"], console=console)

    return {
        "config": config,
        "embeddings": embeddings,
        "e5_model": e5_model,
        "pilot_results": pilot_res,
        "evaluation": e5_eval,
        "cumulative_summary": cumulative_df,
    }


if __name__ == "__main__":
    run_e5_pipeline()
