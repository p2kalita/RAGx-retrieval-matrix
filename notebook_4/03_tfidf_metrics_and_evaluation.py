"""
Section 3: TF-IDF Retrieval Evaluation Metrics
===============================================

Covers Part I (Retrieval Evaluation Metrics: Precision@5, Hit@5, MRR, nDCG@5)
from Notebook 4 (Cells 063 to 086).

This module loads the human-annotated TF-IDF pilot retrieval results, validates label completeness
and consistency, constructs binary relevance mappings, and evaluates rank-aware performance metrics
(Precision@5, Hit@5, Mean Reciprocal Rank, and nDCG@5) with rich statistical summaries.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure UTF-8 output encoding across consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Default Project Paths
DEFAULT_PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
DEFAULT_ANNOTATION_FILE = (
    DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "annotation" / "tfidf_pilot_annotations.csv"
)


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


def load_annotated_tfidf_results(
    annotation_file: Path = DEFAULT_ANNOTATION_FILE,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Load the manually annotated TF-IDF pilot retrieval results from CSV.
    (Notebook Cells 064 to 065)
    """
    if console is None:
        console = Console(legacy_windows=False)

    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotated TF-IDF file missing at: {annotation_file}")

    annotated_df = pd.read_csv(annotation_file)

    console.print(
        Panel(
            f"[bold white]Loaded Annotated TF-IDF Dataset:[/bold white] [bold green]{len(annotated_df)} records[/bold green]\n"
            f"[dim]Columns: {', '.join(annotated_df.columns)}[/dim]",
            title="[bold cyan] Annotation Dataset Loaded [/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return annotated_df


def validate_annotation_dataset(
    annotated_df: pd.DataFrame,
    expected_records: int = 50,
    console: Optional[Console] = None,
) -> None:
    """
    Validate dataset completeness and relevance label distributions.
    (Notebook Cells 066 to 067)
    """
    if console is None:
        console = Console(legacy_windows=False)

    assert len(annotated_df) == expected_records, (
        f"Expected {expected_records} annotations, found {len(annotated_df)}."
    )
    assert not annotated_df["Relevance"].isna().any(), (
        "Missing relevance values detected in annotations."
    )

    table = Table(
        title="[bold bright_yellow]Relevance Annotation Distribution[/bold bright_yellow]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Relevance Grade", style="bold cyan", justify="center")
    table.add_column("Meaning", style="white")
    table.add_column("Count", style="bold green", justify="right")
    table.add_column("Percentage", style="bold magenta", justify="right")

    total = len(annotated_df)
    labels_map = {
        2: "Highly Relevant (Direct Legal Principle)",
        1: "Partially Relevant (Context / Argument)",
        0: "Not Relevant (Off-topic / Immaterial)",
    }

    counts = annotated_df["Relevance"].value_counts().to_dict()
    for grade in [2, 1, 0]:
        cnt = counts.get(grade, 0)
        table.add_row(str(grade), labels_map.get(grade, "Unknown"), str(cnt), f"{(cnt/total)*100:.1f}%")

    console.print(table)


def create_binary_relevance_labels(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform graded relevance (0, 1, 2) into binary relevance (0 or 1).
    (Notebook Cells 068 to 069)
    """
    df = annotated_df.copy()
    df["Binary_Relevance"] = (df["Relevance"] > 0).astype(int)
    return df


def compute_precision_at_k(
    annotated_df: pd.DataFrame,
    top_k: int = 5,
    console: Optional[Console] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Precision@k per question and overall mean.
    (Notebook Cells 070 to 072)
    """
    if console is None:
        console = Console(legacy_windows=False)

    precision_at_5 = (
        annotated_df
        .groupby("Question_ID")["Binary_Relevance"]
        .mean()
        .reset_index(name="Precision@5")
    )
    mean_precision = float(precision_at_5["Precision@5"].mean())

    # Validations from Cell 072
    counts = annotated_df.groupby("Question_ID").size()
    assert (counts == top_k).all(), f"Each question must have exactly {top_k} retrieved chunks."
    assert precision_at_5["Precision@5"].between(0, 1).all(), "Precision values must be in [0, 1]."

    return precision_at_5, mean_precision


def compute_hit_at_k(
    annotated_df: pd.DataFrame,
    console: Optional[Console] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Hit@k per question and overall mean.
    (Notebook Cells 074 to 076)
    """
    if console is None:
        console = Console(legacy_windows=False)

    hit_at_5 = (
        annotated_df
        .groupby("Question_ID")["Binary_Relevance"]
        .max()
        .reset_index(name="Hit@5")
    )
    mean_hit = float(hit_at_5["Hit@5"].mean())

    assert hit_at_5["Hit@5"].isin([0, 1]).all(), "Hit@5 values must be 0 or 1."
    return hit_at_5, mean_hit


def compute_mrr(
    annotated_df: pd.DataFrame,
    console: Optional[Console] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Mean Reciprocal Rank (MRR) per question and overall mean.
    (Notebook Cells 078 to 080)
    """
    if console is None:
        console = Console(legacy_windows=False)

    def reciprocal_rank(group: pd.DataFrame) -> float:
        rel_rows = group[group["Binary_Relevance"] == 1]
        if rel_rows.empty:
            return 0.0
        first_rel_rank = rel_rows["Rank"].min()
        return 1.0 / float(first_rel_rank)

    mrr_per_q = (
        annotated_df
        .groupby("Question_ID")
        .apply(reciprocal_rank, include_groups=False)
        .reset_index(name="Reciprocal_Rank")
    )
    mean_mrr = float(mrr_per_q["Reciprocal_Rank"].mean())

    assert mrr_per_q["Reciprocal_Rank"].between(0, 1).all(), "MRR values must be in [0, 1]."
    return mrr_per_q, mean_mrr


def compute_ndcg_at_k(
    annotated_df: pd.DataFrame,
    top_k: int = 5,
    console: Optional[Console] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG@k) per question and overall mean.
    (Notebook Cells 082 to 084)
    """
    if console is None:
        console = Console(legacy_windows=False)

    ndcg_scores = []
    for question_id, group in annotated_df.groupby("Question_ID"):
        true_rel = group["Relevance"].to_numpy().reshape(1, -1)
        pred_scores = group["Similarity_Score"].to_numpy().reshape(1, -1)

        score = float(ndcg_score(true_rel, pred_scores, k=top_k))
        ndcg_scores.append({"Question_ID": question_id, "nDCG@5": score})

    ndcg_results = pd.DataFrame(ndcg_scores)
    mean_ndcg = float(ndcg_results["nDCG@5"].mean())

    assert ndcg_results["nDCG@5"].between(0, 1).all(), "nDCG values must be in [0, 1]."
    return ndcg_results, mean_ndcg


def compile_tfidf_evaluation_summary(
    mean_precision: float,
    mean_hit: float,
    mean_mrr: float,
    mean_ndcg: float,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Consolidate mean evaluation metrics into a summary table.
    (Notebook Cell 086)
    """
    if console is None:
        console = Console(legacy_windows=False)

    summary_df = pd.DataFrame({
        "Retriever": ["TF-IDF"],
        "Precision@5": [mean_precision],
        "Hit@5": [mean_hit],
        "MRR": [mean_mrr],
        "nDCG@5": [mean_ndcg],
    })

    table = Table(
        title="[bold bright_green]TF-IDF Pilot Retrieval Evaluation Metrics Summary[/bold bright_green]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_green",
    )
    table.add_column("Retriever", style="bold cyan", justify="left")
    table.add_column("Precision@5", style="bold white", justify="right")
    table.add_column("Hit@5", style="bold yellow", justify="right")
    table.add_column("MRR", style="bold magenta", justify="right")
    table.add_column("nDCG@5", style="bold green", justify="right")

    table.add_row(
        "TF-IDF",
        f"{mean_precision:.4f}",
        f"{mean_hit:.4f}",
        f"{mean_mrr:.4f}",
        f"{mean_ndcg:.4f}",
    )
    console.print(table)
    return summary_df


def run_tfidf_evaluation() -> Dict[str, Any]:
    """Execute complete TF-IDF metric evaluation pipeline."""
    console = Console(legacy_windows=False)
    print_banner(
        "TF-IDF Retrieval Evaluation Metrics",
        "Part I: Precision@5, Hit@5, MRR, and nDCG@5 Computation",
        console=console,
    )

    # 1. Load & validate
    annotated_df = load_annotated_tfidf_results(DEFAULT_ANNOTATION_FILE, console=console)
    validate_annotation_dataset(annotated_df, console=console)

    # 2. Binary labels
    df_binary = create_binary_relevance_labels(annotated_df)

    # 3. Compute metrics
    precision_df, mean_prec = compute_precision_at_k(df_binary, top_k=5, console=console)
    hit_df, mean_hit = compute_hit_at_k(df_binary, console=console)
    mrr_df, mean_mrr = compute_mrr(df_binary, console=console)
    ndcg_df, mean_ndcg = compute_ndcg_at_k(annotated_df, top_k=5, console=console)

    # 4. Consolidated Summary
    summary_df = compile_tfidf_evaluation_summary(mean_prec, mean_hit, mean_mrr, mean_ndcg, console=console)

    # Per-question breakdown table
    breakdown_df = precision_df.merge(hit_df, on="Question_ID").merge(mrr_df, on="Question_ID").merge(ndcg_df, on="Question_ID")
    q_table = Table(
        title="[bold bright_cyan]Per-Question TF-IDF Evaluation Breakdown[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    q_table.add_column("Question ID", style="bold cyan", justify="center")
    q_table.add_column("Precision@5", justify="right")
    q_table.add_column("Hit@5", justify="right")
    q_table.add_column("Reciprocal Rank", justify="right")
    q_table.add_column("nDCG@5", justify="right")

    for _, r in breakdown_df.iterrows():
        q_table.add_row(
            str(r["Question_ID"]),
            f"{r['Precision@5']:.2f}",
            f"{int(r['Hit@5'])}",
            f"{r['Reciprocal_Rank']:.4f}",
            f"{r['nDCG@5']:.4f}",
        )
    console.print(q_table)

    return {
        "annotated_df": df_binary,
        "precision_df": precision_df,
        "hit_df": hit_df,
        "mrr_df": mrr_df,
        "ndcg_df": ndcg_df,
        "summary": summary_df,
        "breakdown": breakdown_df,
    }


if __name__ == "__main__":
    run_tfidf_evaluation()
