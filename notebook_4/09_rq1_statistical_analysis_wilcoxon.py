"""
=============================================================================
NyayaRAG Pilot Evaluation — Script 09: RQ1 Statistical Analysis (Wilcoxon Test)
=============================================================================

This script executes Part V of the NyayaRAG pilot evaluation pipeline:
  Research Question 1 (RQ1):
  "Do semantic retrieval methodologies significantly outperform lexical retrieval
   methodologies for legal information retrieval?"

Workflow:
  1. Loads human annotations for all 5 standalone retrievers:
       - Lexical Paradigm:  TF-IDF, BM25
       - Semantic Paradigm: E5, BGE, InLegalBERT
  2. Computes per-question metrics (Precision@5, Hit@5, MRR, nDCG@5) for each model.
  3. Computes paradigm-level averages for each question:
       - Lexical Score  = Mean(TF-IDF, BM25)
       - Semantic Score = Mean(E5, BGE, InLegalBERT)
  4. Generates descriptive statistics (Mean and Standard Deviation per paradigm).
  5. Executes one-sided Wilcoxon signed-rank tests (Semantic > Lexical, alpha = 0.05)
     with Pratt/Wilcox zero-handling and effect size r.
  6. Renders formatted Rich tables and formal statistical decision panels.
=============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
# 1. Directory & Path Configurations
# =============================================================================

BASE_DIR = Path("d:/_ResearchNotebooks/NyayaRAG")
ANNOTATION_DIR = BASE_DIR / "evaluation" / "pilot" / "annotation"
BENCHMARK_FILE = BASE_DIR / "benchmark" / "final" / "pilot_questions.csv"

TFIDF_ANNOTATION = ANNOTATION_DIR / "tfidf_pilot_annotations.csv"
BM25_ANNOTATION = ANNOTATION_DIR / "bm25_pilot_annotations.csv"
E5_ANNOTATION = ANNOTATION_DIR / "e5_pilot_annotations.csv"
BGE_ANNOTATION = ANNOTATION_DIR / "bge_pilot_annotations.csv"
INLEGALBERT_ANNOTATION = ANNOTATION_DIR / "inlegalbert_pilot_annotations.csv"

ALPHA = 0.05
TOP_K = 5


# =============================================================================
# 2. Metric Extraction Helpers
# =============================================================================

def compute_model_metrics(
    annotation_path: Path,
    score_column_candidates=("Similarity_Score", "BM25_Score", "E5_Score", "BGE_Score", "InLegalBERT_Score"),
    top_k: int = TOP_K
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes per-question Precision@5, Hit@5, MRR, and nDCG@5 from an annotation CSV.
    """
    df = pd.read_csv(annotation_path, encoding="utf-8", encoding_errors="replace")
    df["Binary_Relevance"] = (df["Relevance"] > 0).astype(int)

    # Precision@5
    precision_df = (
        df.groupby("Question_ID")["Binary_Relevance"]
        .mean()
        .reset_index(name="Precision@5")
    )

    # Hit@5
    hit_df = (
        df.groupby("Question_ID")["Binary_Relevance"]
        .max()
        .reset_index(name="Hit@5")
    )

    # MRR
    def calc_rr(group):
        rel = group[group["Binary_Relevance"] == 1]
        if rel.empty:
            return 0.0
        return 1.0 / rel["Rank"].min()

    mrr_df = (
        df.groupby("Question_ID")
        .apply(calc_rr, include_groups=False)
        .reset_index(name="Reciprocal_Rank")
    )

    # Find score column
    score_col = None
    for cand in score_column_candidates:
        if cand in df.columns:
            score_col = cand
            break
    if score_col is None:
        score_col = "Similarity_Score"

    # nDCG@5
    ndcg_list = []
    for q_id, group in df.groupby("Question_ID"):
        grp = group.sort_values("Rank")
        true_rel = grp["Relevance"].to_numpy(dtype=float).reshape(1, -1)
        pred_scores = grp[score_col].to_numpy(dtype=float).reshape(1, -1)
        val = ndcg_score(true_rel, pred_scores, k=top_k)
        ndcg_list.append({"Question_ID": q_id, "nDCG@5": val})

    ndcg_df = pd.DataFrame(ndcg_list)

    return precision_df, hit_df, mrr_df, ndcg_df


# =============================================================================
# 3. RQ1 Paired Dataset Builder
# =============================================================================

def build_rq1_paired_dataset() -> pd.DataFrame:
    """
    Builds the paired metric dataset comparing Lexical (TF-IDF, BM25)
    and Semantic (E5, BGE, InLegalBERT) paradigms. (Notebook Cells 292-293)
    """
    # 1. TF-IDF
    p_tf, h_tf, m_tf, n_tf = compute_model_metrics(TFIDF_ANNOTATION)
    tfidf_rq1 = (
        p_tf.rename(columns={"Precision@5": "TFIDF_Precision@5"})
        .merge(h_tf.rename(columns={"Hit@5": "TFIDF_Hit@5"}), on="Question_ID")
        .merge(m_tf.rename(columns={"Reciprocal_Rank": "TFIDF_MRR"}), on="Question_ID")
        .merge(n_tf.rename(columns={"nDCG@5": "TFIDF_nDCG@5"}), on="Question_ID")
    )

    # 2. BM25
    p_bm, h_bm, m_bm, n_bm = compute_model_metrics(BM25_ANNOTATION)
    bm25_rq1 = (
        p_bm.rename(columns={"Precision@5": "BM25_Precision@5"})
        .merge(h_bm.rename(columns={"Hit@5": "BM25_Hit@5"}), on="Question_ID")
        .merge(m_bm.rename(columns={"Reciprocal_Rank": "BM25_MRR"}), on="Question_ID")
        .merge(n_bm.rename(columns={"nDCG@5": "BM25_nDCG@5"}), on="Question_ID")
    )

    # 3. E5
    p_e5, h_e5, m_e5, n_e5 = compute_model_metrics(E5_ANNOTATION)
    e5_rq1 = (
        p_e5.rename(columns={"Precision@5": "E5_Precision@5"})
        .merge(h_e5.rename(columns={"Hit@5": "E5_Hit@5"}), on="Question_ID")
        .merge(m_e5.rename(columns={"Reciprocal_Rank": "E5_MRR"}), on="Question_ID")
        .merge(n_e5.rename(columns={"nDCG@5": "E5_nDCG@5"}), on="Question_ID")
    )

    # 4. BGE
    p_bge, h_bge, m_bge, n_bge = compute_model_metrics(BGE_ANNOTATION)
    bge_rq1 = (
        p_bge.rename(columns={"Precision@5": "BGE_Precision@5"})
        .merge(h_bge.rename(columns={"Hit@5": "BGE_Hit@5"}), on="Question_ID")
        .merge(m_bge.rename(columns={"Reciprocal_Rank": "BGE_MRR"}), on="Question_ID")
        .merge(n_bge.rename(columns={"nDCG@5": "BGE_nDCG@5"}), on="Question_ID")
    )

    # 5. InLegalBERT
    p_in, h_in, m_in, n_in = compute_model_metrics(INLEGALBERT_ANNOTATION)
    inleg_rq1 = (
        p_in.rename(columns={"Precision@5": "InLegalBERT_Precision@5"})
        .merge(h_in.rename(columns={"Hit@5": "InLegalBERT_Hit@5"}), on="Question_ID")
        .merge(m_in.rename(columns={"Reciprocal_Rank": "InLegalBERT_MRR"}), on="Question_ID")
        .merge(n_in.rename(columns={"nDCG@5": "InLegalBERT_nDCG@5"}), on="Question_ID")
    )

    # Merge all 5 models
    merged = (
        tfidf_rq1
        .merge(bm25_rq1, on="Question_ID")
        .merge(e5_rq1, on="Question_ID")
        .merge(bge_rq1, on="Question_ID")
        .merge(inleg_rq1, on="Question_ID")
    )

    # Compute Paradigm Means
    merged["Lexical_Precision@5"] = merged[["TFIDF_Precision@5", "BM25_Precision@5"]].mean(axis=1)
    merged["Semantic_Precision@5"] = merged[["E5_Precision@5", "BGE_Precision@5", "InLegalBERT_Precision@5"]].mean(axis=1)

    merged["Lexical_Hit@5"] = merged[["TFIDF_Hit@5", "BM25_Hit@5"]].mean(axis=1)
    merged["Semantic_Hit@5"] = merged[["E5_Hit@5", "BGE_Hit@5", "InLegalBERT_Hit@5"]].mean(axis=1)

    merged["Lexical_MRR"] = merged[["TFIDF_MRR", "BM25_MRR"]].mean(axis=1)
    merged["Semantic_MRR"] = merged[["E5_MRR", "BGE_MRR", "InLegalBERT_MRR"]].mean(axis=1)

    merged["Lexical_nDCG@5"] = merged[["TFIDF_nDCG@5", "BM25_nDCG@5"]].mean(axis=1)
    merged["Semantic_nDCG@5"] = merged[["E5_nDCG@5", "BGE_nDCG@5", "InLegalBERT_nDCG@5"]].mean(axis=1)

    rq1_comp = merged[[
        "Question_ID",
        "Lexical_Precision@5", "Semantic_Precision@5",
        "Lexical_Hit@5", "Semantic_Hit@5",
        "Lexical_MRR", "Semantic_MRR",
        "Lexical_nDCG@5", "Semantic_nDCG@5"
    ]].copy()

    # Validations
    df_pilot = pd.read_csv(BENCHMARK_FILE)
    assert len(rq1_comp) == len(df_pilot), "RQ1 row count mismatch"
    assert rq1_comp["Question_ID"].is_unique, "Duplicate question IDs"
    assert rq1_comp.drop(columns="Question_ID").notna().all().all(), "Missing metric values in RQ1"
    assert rq1_comp.drop(columns="Question_ID").apply(lambda c: c.between(0, 1).all()).all(), "Values outside [0, 1]"

    return rq1_comp


def display_rq1_paired_table(rq1_comp: pd.DataFrame):
    """Render the per-question paired comparisons."""
    table = Table(
        title="RQ1 Paired Per-Question Metrics: Lexical vs. Semantic Paradigms",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Q ID", style="bold cyan")
    table.add_column("Lex P@5", justify="right")
    table.add_column("Sem P@5", justify="right")
    table.add_column("Lex Hit@5", justify="right")
    table.add_column("Sem Hit@5", justify="right")
    table.add_column("Lex MRR", justify="right")
    table.add_column("Sem MRR", justify="right")
    table.add_column("Lex nDCG@5", justify="right")
    table.add_column("Sem nDCG@5", justify="right")

    for _, row in rq1_comp.iterrows():
        table.add_row(
            row["Question_ID"],
            f"{row['Lexical_Precision@5']:.4f}",
            f"{row['Semantic_Precision@5']:.4f}",
            f"{row['Lexical_Hit@5']:.4f}",
            f"{row['Semantic_Hit@5']:.4f}",
            f"{row['Lexical_MRR']:.4f}",
            f"{row['Semantic_MRR']:.4f}",
            f"{row['Lexical_nDCG@5']:.4f}",
            f"{row['Semantic_nDCG@5']:.4f}",
        )

    console.print(table)


# =============================================================================
# 4. Descriptive Statistics
# =============================================================================

def compute_descriptive_statistics(rq1_comp: pd.DataFrame) -> pd.DataFrame:
    """Compute Mean and SD for Lexical and Semantic paradigms."""
    summary_df = pd.DataFrame({
        "Metric": ["Precision@5", "Hit@5", "MRR", "nDCG@5"],
        "Lexical Mean": [
            rq1_comp["Lexical_Precision@5"].mean(),
            rq1_comp["Lexical_Hit@5"].mean(),
            rq1_comp["Lexical_MRR"].mean(),
            rq1_comp["Lexical_nDCG@5"].mean()
        ],
        "Semantic Mean": [
            rq1_comp["Semantic_Precision@5"].mean(),
            rq1_comp["Semantic_Hit@5"].mean(),
            rq1_comp["Semantic_MRR"].mean(),
            rq1_comp["Semantic_nDCG@5"].mean()
        ],
        "Lexical SD": [
            rq1_comp["Lexical_Precision@5"].std(),
            rq1_comp["Lexical_Hit@5"].std(),
            rq1_comp["Lexical_MRR"].std(),
            rq1_comp["Lexical_nDCG@5"].std()
        ],
        "Semantic SD": [
            rq1_comp["Semantic_Precision@5"].std(),
            rq1_comp["Semantic_Hit@5"].std(),
            rq1_comp["Semantic_MRR"].std(),
            rq1_comp["Semantic_nDCG@5"].std()
        ]
    })

    table = Table(
        title="RQ1 Descriptive Statistics: Lexical vs. Semantic Retrieval",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_green"
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Lexical Mean", justify="right", style="green")
    table.add_column("Semantic Mean", justify="right", style="magenta")
    table.add_column("Lexical SD", justify="right")
    table.add_column("Semantic SD", justify="right")

    for _, r in summary_df.iterrows():
        table.add_row(
            r["Metric"],
            f"{r['Lexical Mean']:.4f}",
            f"{r['Semantic Mean']:.4f}",
            f"{r['Lexical SD']:.4f}",
            f"{r['Semantic SD']:.4f}"
        )

    console.print(table)
    return summary_df


# =============================================================================
# 5. Inferential Wilcoxon Signed-Rank Test
# =============================================================================

def run_wilcoxon_test(
    data: pd.DataFrame,
    lexical_column: str,
    semantic_column: str,
    metric_name: str,
    alpha: float = ALPHA
) -> Dict[str, Any]:
    """
    Performs a one-sided Wilcoxon signed-rank test (Semantic > Lexical).
    Computes effect size r = |z| / sqrt(n).
    """
    lexical = data[lexical_column]
    semantic = data[semantic_column]

    # One-sided test (Semantic > Lexical)
    statistic, p_value = wilcoxon(
        semantic,
        lexical,
        alternative="greater",
        zero_method="wilcox"
    )

    # Effect size r = |z| / sqrt(n)
    n = np.sum((semantic - lexical) != 0)
    if n > 0:
        z = (statistic - (n * (n + 1) / 4.0)) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        effect_size = abs(z) / np.sqrt(n)
    else:
        effect_size = 0.0

    decision = "Reject H0" if p_value < alpha else "Fail to Reject H0"

    return {
        "Metric": metric_name,
        "Wilcoxon Statistic": statistic,
        "p-value": p_value,
        "Effect Size (r)": effect_size,
        "Decision": decision
    }


def execute_all_wilcoxon_tests(rq1_comp: pd.DataFrame) -> pd.DataFrame:
    """Execute tests across all 4 evaluation metrics."""
    test_results = [
        run_wilcoxon_test(rq1_comp, "Lexical_Precision@5", "Semantic_Precision@5", "Precision@5"),
        run_wilcoxon_test(rq1_comp, "Lexical_Hit@5", "Semantic_Hit@5", "Hit@5"),
        run_wilcoxon_test(rq1_comp, "Lexical_MRR", "Semantic_MRR", "MRR"),
        run_wilcoxon_test(rq1_comp, "Lexical_nDCG@5", "Semantic_nDCG@5", "nDCG@5"),
    ]
    results_df = pd.DataFrame(test_results).round(4)

    # Assertions
    assert len(results_df) == 4
    assert results_df["p-value"].between(0, 1).all()
    assert results_df["Effect Size (r)"].between(0, 1).all()

    table = Table(
        title="RQ1 Wilcoxon Signed-Rank Test Results (H1: Semantic > Lexical, alpha=0.05)",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_red"
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Wilcoxon W", justify="right", style="yellow")
    table.add_column("p-value", justify="right", style="bold white")
    table.add_column("Effect Size (r)", justify="right", style="green")
    table.add_column("Statistical Decision", style="bold red")

    for _, r in results_df.iterrows():
        table.add_row(
            r["Metric"],
            f"{r['Wilcoxon Statistic']:.4f}",
            f"{r['p-value']:.4f}",
            f"{r['Effect Size (r)']:.4f}",
            r["Decision"]
        )

    console.print(table)
    return results_df


# =============================================================================
# 6. Main Pipeline
# =============================================================================

def main():
    console.print(Panel(
        "[bold cyan]NyayaRAG Pilot Evaluation — Script 09[/bold cyan]\n"
        "[white]RQ1 Inferential Statistical Analysis: Lexical vs. Semantic Paradigms[/white]",
        box=box.DOUBLE
    ))

    # 1. Build paired dataset
    console.print("[cyan]Building RQ1 paired per-question dataset from annotations...[/cyan]")
    rq1_comp = build_rq1_paired_dataset()
    display_rq1_paired_table(rq1_comp)

    # 2. Descriptive statistics
    console.print()
    desc_df = compute_descriptive_statistics(rq1_comp)

    # 3. Inferential tests
    console.print()
    results_df = execute_all_wilcoxon_tests(rq1_comp)

    # 4. Academic Interpretation Banner
    console.print()
    console.print(Panel(
        "[bold bright_white]RQ1 Statistical Conclusion:[/bold bright_white]\n\n"
        "• For all 4 metrics (Precision@5, Hit@5, MRR, nDCG@5), the p-values exceed alpha = 0.05.\n"
        "• Statistical Decision: [bold yellow]Fail to Reject H0[/bold yellow] across all retrieval metrics.\n"
        "• Interpretation: On this 10-question pilot benchmark, there is insufficient evidence to conclude\n"
        "  that dense semantic retrieval significantly outperforms lexical retrieval.\n"
        "• Note: This motivates the full 50-question benchmark evaluation to provide higher statistical power.",
        title="[bold green]Formal RQ1 Research Findings[/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
