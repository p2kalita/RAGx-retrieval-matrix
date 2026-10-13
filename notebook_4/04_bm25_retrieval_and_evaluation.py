"""
Section 4: BM25 Probabilistic Lexical Retrieval and Evaluation
==============================================================

Covers Part J (BM25 Retrieval Evaluation), Part K (BM25 Retrieval Evaluation Metrics),
and Part L (Cumulative Pilot Retrieval Performance: TF-IDF vs BM25)
from Notebook 4 (Cells 087 to 132).

This module loads the trained BM25Okapi inverted index, validates tokenized corpus alignment,
executes probabilistic lexical retrieval across the pilot benchmark, saves retrieval results,
creates the annotation template, evaluates annotated performance metrics (Precision@5, Hit@5,
MRR, nDCG@5), and produces the comparative cumulative performance dashboard.
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


def load_bm25_artefacts(
    model_file: Path,
    tokenized_corpus_file: Path,
    df_chunks: pd.DataFrame,
    console: Optional[Console] = None,
) -> Tuple[Any, Any]:
    """
    Load BM25 model and tokenized corpus from disk and validate corpus alignment.
    (Notebook Cells 088 to 095)
    """
    if console is None:
        console = Console(legacy_windows=False)

    if not model_file.exists():
        raise FileNotFoundError(f"BM25 model missing at: {model_file}")

    bm25 = joblib.load(model_file)
    tokenized_corpus = joblib.load(tokenized_corpus_file) if tokenized_corpus_file.exists() else None

    # Validate alignment
    if tokenized_corpus is not None:
        assert len(tokenized_corpus) == len(df_chunks), (
            f"BM25 corpus has {len(tokenized_corpus)} items, but chunk corpus has {len(df_chunks)}."
        )

    table = Table(
        title="[bold bright_cyan]BM25 Retrieval Artefacts & Corpus Alignment[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("BM25 Algorithm", "BM25Okapi (Probabilistic Term Saturation)")
    table.add_row("Average Document Length (avgdl)", f"{bm25.avgdl:.2f} words")
    table.add_row("Corpus Alignment Status", f"[bold green][OK] {len(df_chunks):,} chunks verified[/bold green]")

    console.print(table)
    return bm25, tokenized_corpus


def retrieve_bm25(
    query: str,
    bm25: Any,
    df_chunks: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve Top-k most relevant chunks using BM25Okapi scores.
    (Notebook Cell 100)
    """
    # 1. Tokenize query
    tokens = query.lower().split()

    # 2. Compute BM25 relevance scores
    scores = np.array(bm25.get_scores(tokens))

    # 3. Top-k ranking indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    # 4. Assemble result DataFrame
    results = df_chunks.iloc[top_indices].copy()
    results["BM25_Score"] = scores[top_indices]
    results["Similarity_Score"] = scores[top_indices]

    columns_to_return = [
        "global_chunk_id",
        "judgment_id",
        "case_title",
        "local_chunk_id",
        "BM25_Score",
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
    (Notebook Cell 106)
    """
    if console is None:
        console = Console(legacy_windows=False)

    console.print()
    console.print(
        Panel(
            f"[bold white]Inspected Query:[/bold white] [bold yellow]\"{query_str}\"[/bold yellow]",
            title="[bold cyan] BM25 Top Retrieval Inspection [/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    for i in range(min(display_k, len(results_df))):
        row = results_df.iloc[i]
        case_title = str(row.get("case_title", "Unknown"))
        score_val = float(row.get("Similarity_Score", 0.0))
        chunk_id = str(row.get("global_chunk_id", i + 1))
        local_id = str(row.get("local_chunk_id", 1))
        j_id = str(row.get("judgment_id", "Unknown"))
        passage = str(row.get("chunk_text", ""))[:450] + "..."

        chunk_panel = Panel(
            f"[bold cyan]Case Title:[/bold cyan] [bold white]{case_title}[/bold white]\n"
            f"[dim]Judgment ID: {j_id} | Global Chunk: {chunk_id} | Local Chunk: {local_id}[/dim]\n\n"
            f"[italic white]{passage}[/italic white]",
            title=f"[bold green]Rank {i+1}[/bold green] — [bold magenta]BM25 Score: {score_val:.4f}[/bold magenta]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 2),
        )
        console.print(chunk_panel)


def run_pilot_bm25_retrieval(
    df_pilot: pd.DataFrame,
    bm25: Any,
    df_chunks: pd.DataFrame,
    top_k: int = 5,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Retrieve Top-k results for all questions in the pilot benchmark using BM25.
    (Notebook Cells 109 to 110)
    """
    if console is None:
        console = Console(legacy_windows=False)

    all_bm25_results = []
    for _, row in df_pilot.iterrows():
        question_id = row["Question_ID"]
        question = row["Question"]

        retrieved = retrieve_bm25(
            question,
            bm25=bm25,
            df_chunks=df_chunks,
            top_k=top_k,
        )

        retrieved.insert(0, "Question", question)
        retrieved.insert(0, "Question_ID", question_id)
        retrieved["Rank"] = range(1, len(retrieved) + 1)
        all_bm25_results.append(retrieved)

    df_bm25_results = pd.concat(all_bm25_results, ignore_index=True)

    table = Table(
        title=f"[bold bright_magenta]Consolidated BM25 Pilot Retrieval ({len(df_bm25_results)} Chunks)[/bold bright_magenta]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Q ID", style="bold cyan", justify="center", width=8)
    table.add_column("Rank", style="bold yellow", justify="center", width=6)
    table.add_column("BM25 Score", style="bold green", justify="right", width=12)
    table.add_column("Chunk ID", style="dim cyan", justify="center", width=10)
    table.add_column("Case Title", style="bold white", width=40)

    for _, row in df_bm25_results.head(10).iterrows():
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
    return df_bm25_results


def save_bm25_pilot_results(
    df_bm25_results: pd.DataFrame,
    output_file: Path,
    console: Optional[Console] = None,
) -> None:
    """Save raw BM25 retrieval outputs to CSV (Notebook Cells 111 to 112)."""
    if console is None:
        console = Console(legacy_windows=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_bm25_results.to_csv(output_file, index=False)

    console.print(
        Panel(
            f"[bold green]Saved BM25 Pilot Retrieval Results[/bold green]\n"
            f"[white]Destination: {output_file}[/white]\n"
            f"[dim]Total records: {len(df_bm25_results):,} rows[/dim]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def create_and_save_bm25_annotation_template(
    df_bm25_results: pd.DataFrame,
    annotation_file: Path,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """Generate and persist the BM25 manual annotation template (Notebook Cells 113 to 116)."""
    if console is None:
        console = Console(legacy_windows=False)

    annotation_df = df_bm25_results.copy()
    annotation_df["Relevance"] = ""
    annotation_df["Relevance_Level"] = ""
    annotation_df["Comments"] = ""

    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(annotation_file, index=False)

    console.print(
        Panel(
            f"[bold cyan]Created BM25 Annotation Template[/bold cyan]\n"
            f"[white]Destination: {annotation_file}[/white]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return annotation_df


def compute_bm25_metrics(
    annotated_df: pd.DataFrame,
    top_k: int = 5,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """
    Compute Precision@5, Hit@5, MRR, and nDCG@5 for BM25.
    (Notebook Cells 117 to 129)
    """
    if console is None:
        console = Console(legacy_windows=False)

    df = annotated_df.copy()
    df["Binary_Relevance"] = (df["Relevance"] > 0).astype(int)

    # 1. Precision@5
    precision_df = df.groupby("Question_ID")["Binary_Relevance"].mean().reset_index(name="Precision@5")
    mean_precision = float(precision_df["Precision@5"].mean())

    # 2. Hit@5
    hit_df = df.groupby("Question_ID")["Binary_Relevance"].max().reset_index(name="Hit@5")
    mean_hit = float(hit_df["Hit@5"].mean())

    # 3. MRR
    def reciprocal_rank(group: pd.DataFrame) -> float:
        rel_rows = group[group["Binary_Relevance"] == 1]
        if rel_rows.empty:
            return 0.0
        return 1.0 / float(rel_rows["Rank"].min())

    mrr_df = df.groupby("Question_ID").apply(reciprocal_rank, include_groups=False).reset_index(name="Reciprocal_Rank")
    mean_mrr = float(mrr_df["Reciprocal_Rank"].mean())

    score_col = "BM25_Score" if "BM25_Score" in df.columns else "Similarity_Score"

    # 4. nDCG@5
    ndcg_scores = []
    for q_id, group in df.groupby("Question_ID"):
        true_rel = group["Relevance"].to_numpy().reshape(1, -1)
        pred_scores = group[score_col].to_numpy().reshape(1, -1)
        score = float(ndcg_score(true_rel, pred_scores, k=top_k))
        ndcg_scores.append({"Question_ID": q_id, "nDCG@5": score})
    ndcg_df = pd.DataFrame(ndcg_scores)
    mean_ndcg = float(ndcg_df["nDCG@5"].mean())

    summary_df = pd.DataFrame({
        "Retriever": ["BM25"],
        "Precision@5": [mean_precision],
        "Hit@5": [mean_hit],
        "MRR": [mean_mrr],
        "nDCG@5": [mean_ndcg],
    })

    table = Table(
        title="[bold bright_green]BM25 Pilot Retrieval Evaluation Metrics Summary[/bold bright_green]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_green",
    )
    table.add_column("Retriever", style="bold cyan")
    table.add_column("Precision@5", style="bold white", justify="right")
    table.add_column("Hit@5", style="bold yellow", justify="right")
    table.add_column("MRR", style="bold magenta", justify="right")
    table.add_column("nDCG@5", style="bold green", justify="right")

    table.add_row("BM25", f"{mean_precision:.4f}", f"{mean_hit:.4f}", f"{mean_mrr:.4f}", f"{mean_ndcg:.4f}")
    console.print(table)

    return {
        "annotated_df": df,
        "precision_df": precision_df,
        "hit_df": hit_df,
        "mrr_df": mrr_df,
        "ndcg_df": ndcg_df,
        "summary": summary_df,
    }


def compare_cumulative_lexical_results(
    tfidf_summary: pd.DataFrame,
    bm25_summary: pd.DataFrame,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Cumulative comparison dashboard across lexical paradigms (TF-IDF vs BM25).
    (Notebook Cells 131 to 132)
    """
    if console is None:
        console = Console(legacy_windows=False)

    cumulative_df = pd.concat([tfidf_summary, bm25_summary], ignore_index=True)

    table = Table(
        title="[bold bright_cyan]Cumulative Pilot Retrieval Performance: Lexical Baseline Comparison[/bold bright_cyan]",
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


def run_bm25_pipeline() -> Dict[str, Any]:
    """Execute complete BM25 retrieval and evaluation pipeline."""
    console = Console(legacy_windows=False)
    print_banner(
        "BM25 Retrieval Evaluation and Lexical Performance",
        "Part J-L: Model Loading, Retrieval, Metrics, and Cumulative Comparison",
        console=console,
    )

    # 1. Paths
    models_dir = DEFAULT_PROJECT_DIR / "models" / "bm25"
    data_dir = DEFAULT_PROJECT_DIR / "data" / "processed"
    benchmark_file = DEFAULT_PROJECT_DIR / "benchmark" / "final" / "pilot_questions.csv"
    retrieval_out = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "retrieval" / "bm25_pilot_retrieval.csv"
    annotation_out = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "annotation" / "bm25_annotation_template.csv"
    annotated_file = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "annotation" / "bm25_pilot_annotations.csv"

    # 2. Inputs
    df_chunks = pd.read_parquet(data_dir / "chunks_384_40.parquet")
    df_pilot = pd.read_csv(benchmark_file)

    # 3. Model
    bm25, tokenized_corpus = load_bm25_artefacts(
        models_dir / "bm25.pkl",
        models_dir / "tokenized_corpus.pkl",
        df_chunks,
        console=console,
    )

    # 4. Sample query
    sample_q = df_pilot.iloc[0]["Question"]
    sample_res = retrieve_bm25(sample_q, bm25, df_chunks, top_k=5)
    display_retrieved_chunks(sample_res, sample_q, display_k=3, console=console)

    # 5. Full retrieval & template
    pilot_res = run_pilot_bm25_retrieval(df_pilot, bm25, df_chunks, top_k=5, console=console)
    save_bm25_pilot_results(pilot_res, retrieval_out, console=console)
    create_and_save_bm25_annotation_template(pilot_res, annotation_out, console=console)

    # 6. Evaluation metrics
    annotated_df = pd.read_csv(annotated_file)
    bm25_eval = compute_bm25_metrics(annotated_df, top_k=5, console=console)

    # 7. Cumulative comparison with TF-IDF
    import importlib.util
    sec3_path = Path(__file__).resolve().parent / "03_tfidf_metrics_and_evaluation.py"
    spec = importlib.util.spec_from_file_location("sec3_module", sec3_path)
    sec3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sec3)

    tfidf_annot_file = DEFAULT_PROJECT_DIR / "evaluation" / "pilot" / "annotation" / "tfidf_pilot_annotations.csv"
    df_tfidf_annot = pd.read_csv(tfidf_annot_file)
    df_tfidf_annot["Binary_Relevance"] = (df_tfidf_annot["Relevance"] > 0).astype(int)
    _, p_tfidf = sec3.compute_precision_at_k(df_tfidf_annot, top_k=5)
    _, h_tfidf = sec3.compute_hit_at_k(df_tfidf_annot)
    _, m_tfidf = sec3.compute_mrr(df_tfidf_annot)
    _, n_tfidf = sec3.compute_ndcg_at_k(df_tfidf_annot, top_k=5)
    tfidf_sum = pd.DataFrame([{"Retriever": "TF-IDF", "Precision@5": p_tfidf, "Hit@5": h_tfidf, "MRR": m_tfidf, "nDCG@5": n_tfidf}])

    cumulative_df = compare_cumulative_lexical_results(tfidf_sum, bm25_eval["summary"], console=console)

    return {
        "bm25": bm25,
        "pilot_results": pilot_res,
        "evaluation": bm25_eval,
        "cumulative_summary": cumulative_df,
    }


if __name__ == "__main__":
    run_bm25_pipeline()
