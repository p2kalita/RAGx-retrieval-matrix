"""
Section 2: TF-IDF Retrieval and Pilot Benchmark Evaluation
===========================================================

Covers Part E (TF-IDF Retrieval Validation), Part F (TF-IDF Retrieval for the Pilot Benchmark),
Part G (Pilot Relevance Annotation Template), and Part H (Manual Relevance Annotation Guidelines)
from Notebook 4 (Cells 036 to 062).

This module loads the fitted TF-IDF artefacts, validates corpus alignment, implements the
sparse TF-IDF cosine retrieval function, validates retrieval on a sample pilot query,
executes retrieval across all 10 pilot benchmark questions, saves the retrieved results,
and generates the standard manual annotation template.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure UTF-8 output encoding across consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

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


def load_tfidf_artefacts(
    vectorizer_file: Path,
    matrix_file: Path,
    df_chunks: pd.DataFrame,
    console: Optional[Console] = None,
) -> Tuple[Any, Any]:
    """
    Load TF-IDF vectorizer and sparse document-term matrix from disk and validate corpus alignment.
    (Notebook Cells 037 to 040)
    """
    if console is None:
        console = Console(legacy_windows=False)

    if not vectorizer_file.exists() or not matrix_file.exists():
        raise FileNotFoundError(f"TF-IDF artefacts missing in {vectorizer_file.parent}")

    tfidf_vectorizer = joblib.load(vectorizer_file)
    tfidf_matrix = sparse.load_npz(matrix_file)

    # Validate alignment
    assert tfidf_matrix.shape[0] == len(df_chunks), (
        f"Mismatch: TF-IDF matrix has {tfidf_matrix.shape[0]} rows, but chunk corpus has {len(df_chunks)}."
    )

    table = Table(
        title="[bold bright_cyan]TF-IDF Retrieval Artefacts & Corpus Alignment[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Vocabulary Size", f"{len(tfidf_vectorizer.vocabulary_):,} features")
    table.add_row("Matrix Dimensions", f"{tfidf_matrix.shape[0]:,} rows × {tfidf_matrix.shape[1]:,} cols")
    table.add_row("Non-Zero Term Elements", f"{tfidf_matrix.nnz:,}")
    table.add_row("Corpus Chunks Aligned", f"[bold green][OK] {len(df_chunks):,} chunks verified[/bold green]")

    console.print(table)
    return tfidf_vectorizer, tfidf_matrix


def retrieve_tfidf(
    query: str,
    tfidf_vectorizer: Any,
    tfidf_matrix: Any,
    df_chunks: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve Top-k most relevant chunks using sparse TF-IDF cosine similarity.
    (Notebook Cell 044)
    """
    # 1. Transform query into TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query])

    # 2. Compute cosine similarity against all document chunks
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # 3. Top-k ranking indices
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    # 4. Assemble result DataFrame
    results = df_chunks.iloc[top_indices].copy()
    results["Similarity_Score"] = similarity_scores[top_indices]

    columns_to_return = [
        "global_chunk_id",
        "judgment_id",
        "case_title",
        "local_chunk_id",
        "Similarity_Score",
        "chunk_text",
    ]
    # Ensure all required columns exist
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
    (Notebook Cell 047)
    """
    if console is None:
        console = Console(legacy_windows=False)

    console.print()
    console.print(
        Panel(
            f"[bold white]Inspected Query:[/bold white] [bold yellow]\"{query_str}\"[/bold yellow]",
            title="[bold cyan] TF-IDF Top Retrieval Inspection [/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    for i in range(min(display_k, len(results_df))):
        row = results_df.iloc[i]
        case_title = str(row.get("case_title", "Unknown"))
        sim_score = float(row.get("Similarity_Score", 0.0))
        chunk_id = str(row.get("global_chunk_id", i + 1))
        local_id = str(row.get("local_chunk_id", 1))
        j_id = str(row.get("judgment_id", "Unknown"))
        passage = str(row.get("chunk_text", ""))[:450] + "..."

        chunk_panel = Panel(
            f"[bold cyan]Case Title:[/bold cyan] [bold white]{case_title}[/bold white]\n"
            f"[dim]Judgment ID: {j_id} | Global Chunk: {chunk_id} | Local Chunk: {local_id}[/dim]\n\n"
            f"[italic white]{passage}[/italic white]",
            title=f"[bold green]Rank {i+1}[/bold green] — [bold magenta]Similarity Score: {sim_score:.4f}[/bold magenta]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 2),
        )
        console.print(chunk_panel)


def run_pilot_tfidf_retrieval(
    df_pilot: pd.DataFrame,
    tfidf_vectorizer: Any,
    tfidf_matrix: Any,
    df_chunks: pd.DataFrame,
    top_k: int = 5,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Retrieve Top-k results for all questions in the pilot benchmark.
    (Notebook Cells 051 to 052)
    """
    if console is None:
        console = Console(legacy_windows=False)

    all_tfidf_results = []
    for _, row in df_pilot.iterrows():
        question_id = row["Question_ID"]
        question = row["Question"]

        retrieved = retrieve_tfidf(
            question,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            df_chunks=df_chunks,
            top_k=top_k,
        )

        retrieved.insert(0, "Question", question)
        retrieved.insert(0, "Question_ID", question_id)
        retrieved["Rank"] = range(1, len(retrieved) + 1)
        all_tfidf_results.append(retrieved)

    df_tfidf_results = pd.concat(all_tfidf_results, ignore_index=True)

    table = Table(
        title=f"[bold bright_magenta]Consolidated TF-IDF Pilot Retrieval ({len(df_tfidf_results)} Chunks)[/bold bright_magenta]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Q ID", style="bold cyan", justify="center", width=8)
    table.add_column("Rank", style="bold yellow", justify="center", width=6)
    table.add_column("Similarity", style="bold green", justify="right", width=12)
    table.add_column("Chunk ID", style="dim cyan", justify="center", width=10)
    table.add_column("Case Title", style="bold white", width=40)

    # Display first 10 rows
    for _, row in df_tfidf_results.head(10).iterrows():
        title = str(row.get("case_title", ""))
        if len(title) > 38:
            title = title[:35] + "..."
        table.add_row(
            str(row["Question_ID"]),
            str(row["Rank"]),
            f"{float(row['Similarity_Score']):.4f}",
            str(row.get("global_chunk_id", "")),
            title,
        )

    console.print(table)
    return df_tfidf_results


def save_tfidf_pilot_results(
    df_tfidf_results: pd.DataFrame,
    output_file: Path,
    console: Optional[Console] = None,
) -> None:
    """
    Save the raw TF-IDF retrieval outputs to CSV.
    (Notebook Cells 053 to 054)
    """
    if console is None:
        console = Console(legacy_windows=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_tfidf_results.to_csv(output_file, index=False)

    console.print(
        Panel(
            f"[bold green]Saved TF-IDF Pilot Retrieval Results[/bold green]\n"
            f"[white]Destination: {output_file}[/white]\n"
            f"[dim]Total records: {len(df_tfidf_results):,} rows across {df_tfidf_results['Question_ID'].nunique()} queries[/dim]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def create_and_save_annotation_template(
    df_tfidf_results: pd.DataFrame,
    annotation_file: Path,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Generate and persist the manual annotation template.
    (Notebook Cells 056 to 060)
    """
    if console is None:
        console = Console(legacy_windows=False)

    annotation_df = df_tfidf_results.copy()
    annotation_df["Relevance"] = ""
    annotation_df["Relevance_Level"] = ""
    annotation_df["Comments"] = ""

    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(annotation_file, index=False)

    console.print(
        Panel(
            f"[bold cyan]Created TF-IDF Annotation Template[/bold cyan]\n"
            f"[white]Destination: {annotation_file}[/white]\n"
            f"[dim]Template ready for expert manual review (2=Highly Relevant, 1=Partially Relevant, 0=Not Relevant)[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return annotation_df


def run_tfidf_pipeline() -> Dict[str, Any]:
    """Execute complete TF-IDF retrieval and annotation setup pipeline."""
    console = Console(legacy_windows=False)
    print_banner(
        "TF-IDF Retrieval and Pilot Benchmark Evaluation",
        "Part E-H: Model Loading, Retrieval Execution, and Annotation Setup",
        console=console,
    )

    # 1. Paths
    models_dir = DEFAULT_PROJECT_DIR / "models" / "tfidf"
    data_dir = DEFAULT_PROJECT_DIR / "data" / "processed"
    benchmark_file = DEFAULT_PROJECT_DIR / "benchmark" / "final" / "pilot_questions.csv"
    retrieval_out = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "retrieval" / "tfidf_pilot_retrieval.csv"
    annotation_out = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "annotation" / "tfidf_annotation_template.csv"

    # 2. Load inputs
    df_chunks = pd.read_parquet(data_dir / "chunks_384_40.parquet")
    df_pilot = pd.read_csv(benchmark_file)

    # 3. Load TF-IDF model & validate
    vec, mat = load_tfidf_artefacts(
        models_dir / "tfidf_vectorizer.pkl",
        models_dir / "tfidf_matrix.npz",
        df_chunks,
        console=console,
    )

    # 4. Single query validation
    sample_q = df_pilot.iloc[0]["Question"]
    sample_res = retrieve_tfidf(sample_q, vec, mat, df_chunks, top_k=5)
    display_retrieved_chunks(sample_res, sample_q, display_k=3, console=console)

    # 5. Full pilot retrieval
    pilot_res = run_pilot_tfidf_retrieval(df_pilot, vec, mat, df_chunks, top_k=5, console=console)
    save_tfidf_pilot_results(pilot_res, retrieval_out, console=console)

    # 6. Annotation template
    annot_df = create_and_save_annotation_template(pilot_res, annotation_out, console=console)

    return {
        "vectorizer": vec,
        "matrix": mat,
        "pilot_results": pilot_res,
        "annotation_df": annot_df,
    }


if __name__ == "__main__":
    run_tfidf_pipeline()
