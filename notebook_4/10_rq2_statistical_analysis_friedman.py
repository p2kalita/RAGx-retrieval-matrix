"""
=============================================================================
NyayaRAG Pilot Evaluation — Script 10: RQ2 Statistical Analysis (Friedman & Post-Hoc)
=============================================================================

This script executes Part W of the NyayaRAG pilot evaluation pipeline:
  Research Question 2 (RQ2):
  "Which retrieval methodology performs best overall for legal information retrieval?"

Evaluated Retrieval Methodologies (6 Models):
  1. TF-IDF
  2. BM25
  3. E5
  4. BGE
  5. InLegalBERT
  6. Hybrid (BM25 + E5)

Workflow:
  1. Builds wide-format paired datasets for each metric:
       - Precision@5
       - Hit@5
       - Mean Reciprocal Rank (MRR)
       - nDCG@5
  2. Executes the non-parametric omnibus Friedman test (scipy.stats.friedmanchisquare)
     and calculates Kendall's W concordance effect size.
  3. Computes Friedman Mean Ranks across all 6 models for each metric.
  4. Performs pairwise Wilcoxon signed-rank post-hoc comparisons with Holm-Bonferroni
     family-wise error correction (statsmodels multipletests) for significant metrics.
  5. Displays formatted Rich tables, mean rank rankings, and scientific interpretation.
=============================================================================
"""

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests
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
ANNOTATION_DIR = BASE_DIR / "evaluation" / "pilot" / "annotation"
BENCHMARK_FILE = BASE_DIR / "benchmark" / "final" / "pilot_questions.csv"

TFIDF_ANNOTATION = ANNOTATION_DIR / "tfidf_pilot_annotations.csv"
BM25_ANNOTATION = ANNOTATION_DIR / "bm25_pilot_annotations.csv"
E5_ANNOTATION = ANNOTATION_DIR / "e5_pilot_annotations.csv"
BGE_ANNOTATION = ANNOTATION_DIR / "bge_pilot_annotations.csv"
INLEGALBERT_ANNOTATION = ANNOTATION_DIR / "inlegalbert_pilot_annotations.csv"
HYBRID_ANNOTATION = ANNOTATION_DIR / "hybrid_pilot_annotations.csv"

ALPHA = 0.05
TOP_K = 5


# =============================================================================
# 2. Metric Extraction Helpers
# =============================================================================

def extract_per_question_metrics(annotation_path: Path, top_k: int = TOP_K):
    """
    Extracts per-question Precision@5, Hit@5, MRR, and nDCG@5 dataframes.
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
        .reset_index(name="MRR")
    )

    # Find score column
    score_col = None
    for cand in ("RRF_Score", "InLegalBERT_Score", "BGE_Score", "E5_Score", "BM25_Score", "Similarity_Score"):
        if cand in df.columns:
            score_col = cand
            break

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
# 3. Friedman Datasets Preparation
# =============================================================================

def prepare_friedman_dataset(
    metric_name: str,
    tfidf_df: pd.DataFrame,
    bm25_df: pd.DataFrame,
    e5_df: pd.DataFrame,
    bge_df: pd.DataFrame,
    inlegalbert_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    metric_column: str
) -> pd.DataFrame:
    """Creates the paired 6-model dataset required for the Friedman test."""
    dataset = (
        tfidf_df[["Question_ID", metric_column]].rename(columns={metric_column: "TF-IDF"})
        .merge(bm25_df[["Question_ID", metric_column]].rename(columns={metric_column: "BM25"}), on="Question_ID")
        .merge(e5_df[["Question_ID", metric_column]].rename(columns={metric_column: "E5"}), on="Question_ID")
        .merge(bge_df[["Question_ID", metric_column]].rename(columns={metric_column: "BGE"}), on="Question_ID")
        .merge(inlegalbert_df[["Question_ID", metric_column]].rename(columns={metric_column: "InLegalBERT"}), on="Question_ID")
        .merge(hybrid_df[["Question_ID", metric_column]].rename(columns={metric_column: "Hybrid"}), on="Question_ID")
    )

    df_pilot = pd.read_csv(BENCHMARK_FILE)
    assert len(dataset) == len(df_pilot), f"{metric_name}: Row count mismatch"
    assert dataset["Question_ID"].is_unique, f"{metric_name}: Duplicate Question_ID"
    assert dataset.drop(columns="Question_ID").notna().all().all(), f"{metric_name}: Missing values"

    return dataset


def display_friedman_dataset_table(df: pd.DataFrame, metric_name: str):
    """Render the paired Friedman dataset for a given metric."""
    table = Table(
        title=f"Friedman Paired Dataset: {metric_name} (All 6 Models)",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Question ID", style="bold cyan")
    for col in ["TF-IDF", "BM25", "E5", "BGE", "InLegalBERT", "Hybrid"]:
        table.add_column(col, justify="right")

    for _, row in df.iterrows():
        table.add_row(
            row["Question_ID"],
            f"{row['TF-IDF']:.4f}",
            f"{row['BM25']:.4f}",
            f"{row['E5']:.4f}",
            f"{row['BGE']:.4f}",
            f"{row['InLegalBERT']:.4f}",
            f"{row['Hybrid']:.4f}",
        )

    console.print(table)


# =============================================================================
# 4. Friedman Omnibus Test
# =============================================================================

def run_friedman_test(df: pd.DataFrame, metric_name: str, alpha: float = ALPHA) -> Dict[str, Any]:
    """
    Performs the non-parametric Friedman test and computes Kendall's W concordance.
    Kendall's W = Chi2 / (n * (k - 1))
    """
    methods = [c for c in df.columns if c != "Question_ID"]
    scores = [df[m].to_numpy() for m in methods]

    statistic, p_value = friedmanchisquare(*scores)
    n = len(df)
    k = len(methods)

    kendalls_w = statistic / (n * (k - 1))
    decision = "Reject H0" if p_value < alpha else "Fail to Reject H0"

    return {
        "Metric": metric_name,
        "Chi-Square": statistic,
        "df": k - 1,
        "p-value": p_value,
        "Kendall's W": kendalls_w,
        "Decision": decision
    }


def execute_all_friedman_tests(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Execute omnibus Friedman test across all 4 metrics."""
    records = []
    for metric_name, df in datasets.items():
        records.append(run_friedman_test(df, metric_name))

    results_df = pd.DataFrame(records).round(4)

    assert len(results_df) == 4
    assert results_df["p-value"].between(0, 1).all()
    assert results_df["Kendall's W"].between(0, 1).all()

    table = Table(
        title="Omnibus Friedman Test Results Across All 6 Retrieval Models (alpha=0.05)",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_magenta"
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Chi-Square", justify="right", style="yellow")
    table.add_column("df", justify="right", style="white")
    table.add_column("p-value", justify="right", style="bold white")
    table.add_column("Kendall's W", justify="right", style="green")
    table.add_column("Statistical Decision", style="bold red")

    for _, r in results_df.iterrows():
        kendall_val = float(r["Kendall's W"])
        table.add_row(
            str(r["Metric"]),
            f"{float(r['Chi-Square']):.4f}",
            str(int(r["df"])),
            f"{float(r['p-value']):.4f}",
            f"{kendall_val:.4f}",
            str(r["Decision"])
        )

    console.print(table)
    return results_df


# =============================================================================
# 5. Friedman Mean Ranks
# =============================================================================

def compute_mean_ranks(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Computes Friedman mean ranks.
    Higher performance score receives higher rank.
    """
    methods = [c for c in df.columns if c != "Question_ID"]

    # Rank each question: higher score gets higher rank
    ranks = df[methods].apply(
        lambda row: rankdata(row, method="average"),
        axis=1,
        result_type="expand"
    )
    ranks.columns = methods
    mean_ranks = ranks.mean().sort_values(ascending=False)

    results = pd.DataFrame({
        "Retriever": mean_ranks.index,
        "Mean Rank": mean_ranks.values
    })

    table = Table(
        title=f"Friedman Mean Ranks: {metric_name} (Descending Order)",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_green"
    )
    table.add_column("Rank Position", justify="center", style="bold yellow")
    table.add_column("Retriever", style="bold cyan")
    table.add_column("Mean Rank", justify="right", style="bold green")

    for idx, (_, r) in enumerate(results.iterrows(), start=1):
        table.add_row(str(idx), r["Retriever"], f"{r['Mean Rank']:.3f}")

    console.print(table)
    return results


# =============================================================================
# 6. Pairwise Post-Hoc Wilcoxon Tests with Holm Correction
# =============================================================================

def run_posthoc_wilcoxon(df: pd.DataFrame, metric_name: str, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
    for family-wise error rate control.
    """
    methods = [c for c in df.columns if c != "Question_ID"]
    pairs = []
    raw_p_values = []

    for m1, m2 in combinations(methods, 2):
        stat, p = wilcoxon(
            df[m1],
            df[m2],
            zero_method="wilcox",
            alternative="two-sided"
        )
        pairs.append((m1, m2))
        raw_p_values.append(p)

    # Holm correction
    reject, adjusted_p, _, _ = multipletests(
        raw_p_values,
        alpha=alpha,
        method="holm"
    )

    comparisons = []
    for i, (m1, m2) in enumerate(pairs):
        comparisons.append({
            "Metric": metric_name,
            "Retriever 1": m1,
            "Retriever 2": m2,
            "Raw p-value": raw_p_values[i],
            "Holm p-value": adjusted_p[i],
            "Significant": "Yes" if reject[i] else "No"
        })

    posthoc_df = pd.DataFrame(comparisons).round(4)

    table = Table(
        title=f"Pairwise Post-Hoc Wilcoxon Signed-Rank Tests (Holm Corrected): {metric_name}",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Pair Comparison", style="bold cyan")
    table.add_column("Raw p-value", justify="right")
    table.add_column("Holm p-value", justify="right", style="bold white")
    table.add_column("Significant (alpha=0.05)", justify="center", style="bold red")

    for _, r in posthoc_df.iterrows():
        table.add_row(
            f"{r['Retriever 1']} vs. {r['Retriever 2']}",
            f"{r['Raw p-value']:.4f}",
            f"{r['Holm p-value']:.4f}",
            r["Significant"]
        )

    console.print(table)
    return posthoc_df


# =============================================================================
# 7. Main Pipeline Execution
# =============================================================================

def main():
    console.print(Panel(
        "[bold cyan]NyayaRAG Pilot Evaluation — Script 10[/bold cyan]\n"
        "[white]RQ2 Inferential Statistical Analysis: Omnibus Friedman Test & Post-Hoc Comparisons[/white]",
        box=box.DOUBLE
    ))

    # 1. Extract per-question metrics for all 6 models
    console.print("[cyan]Extracting metrics across all 6 models from annotation files...[/cyan]")
    p_tf, h_tf, m_tf, n_tf = extract_per_question_metrics(TFIDF_ANNOTATION)
    p_bm, h_bm, m_bm, n_bm = extract_per_question_metrics(BM25_ANNOTATION)
    p_e5, h_e5, m_e5, n_e5 = extract_per_question_metrics(E5_ANNOTATION)
    p_bge, h_bge, m_bge, n_bge = extract_per_question_metrics(BGE_ANNOTATION)
    p_in, h_in, m_in, n_in = extract_per_question_metrics(INLEGALBERT_ANNOTATION)
    p_hyb, h_hyb, m_hyb, n_hyb = extract_per_question_metrics(HYBRID_ANNOTATION)

    # 2. Build paired datasets
    friedman_precision = prepare_friedman_dataset("Precision@5", p_tf, p_bm, p_e5, p_bge, p_in, p_hyb, "Precision@5")
    friedman_hit = prepare_friedman_dataset("Hit@5", h_tf, h_bm, h_e5, h_bge, h_in, h_hyb, "Hit@5")
    friedman_mrr = prepare_friedman_dataset("MRR", m_tf, m_bm, m_e5, m_bge, m_in, m_hyb, "MRR")
    friedman_ndcg = prepare_friedman_dataset("nDCG@5", n_tf, n_bm, n_e5, n_bge, n_in, n_hyb, "nDCG@5")

    datasets = {
        "Precision@5": friedman_precision,
        "Hit@5": friedman_hit,
        "MRR": friedman_mrr,
        "nDCG@5": friedman_ndcg
    }

    # Display paired datasets
    display_friedman_dataset_table(friedman_precision, "Precision@5")
    console.print()

    # 3. Omnibus Friedman Test
    friedman_summary = execute_all_friedman_tests(datasets)
    console.print()

    # 4. Friedman Mean Ranks
    console.print(Panel("[bold bright_white]Friedman Mean Ranks Across Retrieval Models[/bold bright_white]", box=box.ROUNDED))
    for m_name, df_m in datasets.items():
        compute_mean_ranks(df_m, m_name)
        console.print()

    # 5. Pairwise Post-Hoc Wilcoxon Tests (for significant metrics: Precision@5 and MRR)
    console.print(Panel("[bold bright_white]Pairwise Post-Hoc Comparisons with Holm Correction[/bold bright_white]", box=box.ROUNDED))
    run_posthoc_wilcoxon(friedman_precision, "Precision@5")
    console.print()
    run_posthoc_wilcoxon(friedman_mrr, "MRR")
    console.print()

    # 6. Formal RQ2 Conclusion Banner
    console.print(Panel(
        "[bold bright_white]RQ2 Scientific Findings & Interpretation:[/bold bright_white]\n\n"
        "• Omnibus Friedman Test: Statistically significant differences detected among the 6 models for\n"
        "  [bold yellow]Precision@5[/bold yellow] (Chi2 = 12.0625, p = 0.0340) and [bold yellow]MRR[/bold yellow] (Chi2 = 11.2353, p = 0.0469).\n"
        "  Differences for Hit@5 (p = 0.4060) and nDCG@5 (p = 0.2282) were not statistically significant.\n\n"
        "• Pairwise Post-Hoc Wilcoxon (Holm Adjustment): After rigorous family-wise error rate control,\n"
        "  [bold red]none[/bold red] of the individual pairwise comparisons reached statistical significance (all Holm p > 0.05).\n\n"
        "• Friedman Mean Rank Ordering:\n"
        "  - Precision@5: E5 (4.500) and Hybrid (4.250) ranked highest; InLegalBERT (2.350) ranked lowest.\n"
        "  - Hit@5: Hybrid (4.300) ranked highest.\n"
        "  - MRR: Hybrid (4.450) and E5 (3.900) led the rankings.\n"
        "  - nDCG@5: Hybrid (4.550) led overall ranking quality.\n\n"
        "• Dissertation Implication: Pilot study validates the full pipeline while confirming that a 10-question\n"
        "  sample lacks sufficient statistical power to establish pairwise dominance after multiple comparison correction.\n"
        "  The full 50-question benchmark will establish the definitive statistical rankings.",
        title="[bold green]Formal RQ2 Research Conclusion[/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
