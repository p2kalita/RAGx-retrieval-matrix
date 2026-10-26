"""
NYAYaRAG: A Legal Research Assistant
Script 06: Statistical Analysis & Metric Visualization (Notebook 3 Simulated)

This script implements Step 7 of the LLM-as-a-Judge evaluation workflow:
- Computes rigorous descriptive statistics across all 8 evaluation dimensions.
- Calculates mean, standard deviation, quartiles, and min/max ranges.
- Generates a publication-quality Matplotlib bar chart of metric means.
- Saves the visualization artifact to `evaluation/nyayarag_metric_means.png`.
- Displays Rich summary statistics tables in the console.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all environments
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_3_DIR = PROJECT_ROOT / "notebook_3"
INPUT_RESULTS_FILE = PROJECT_ROOT / "evaluation" / "nyayarag_llm_judge_evaluation.csv"
OUTPUT_PLOT_FILE = PROJECT_ROOT / "evaluation" / "nyayarag_metric_means.png"

METRIC_COLUMNS: List[str] = [
    "humanization_score",
    "indian_english_authenticity_score",
    "conversational_naturalness_score",
    "experience_level_alignment_score",
    "clarity_score",
    "legal_terminology_context_score",
    "reconstruction_plausibility_score",
    "overall_quality_score",
]

FRIENDLY_NAMES: Dict[str, str] = {
    "humanization_score": "Humanization",
    "indian_english_authenticity_score": "Indian English",
    "conversational_naturalness_score": "Conversational",
    "experience_level_alignment_score": "Experience Align.",
    "clarity_score": "Clarity",
    "legal_terminology_context_score": "Legal Terminology",
    "reconstruction_plausibility_score": "Reconstruction Plaus.",
    "overall_quality_score": "Overall Quality",
}


def load_clean_metrics_dataset(file_path: Path = INPUT_RESULTS_FILE) -> pd.DataFrame:
    """
    Load evaluation results and coerce metric columns to numeric.
    Filters out any failed evaluation rows.
    """
    if not file_path.exists():
        s5_path = NOTEBOOK_3_DIR / "05_results_validation_and_export.py"
        spec5 = importlib.util.spec_from_file_location("s5", s5_path)
        s5 = importlib.util.module_from_spec(spec5)
        spec5.loader.exec_module(s5)
        return s5.main()

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    for col in METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    successful = df[df["error"].fillna("").eq("")].copy()
    return successful


def compute_metric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics (mean, std, min, 25%, 50%, 75%, max).
    (Notebook Cell 16)
    """
    stats_df = df[METRIC_COLUMNS].describe().round(2).T
    stats_df["dimension"] = [FRIENDLY_NAMES.get(col, col) for col in stats_df.index]
    return stats_df


def generate_and_save_plot(
    df: pd.DataFrame, output_path: Path = OUTPUT_PLOT_FILE
) -> Path:
    """
    Generate and save the bar chart visualization matching Notebook Cell 16.
    """
    means = df[METRIC_COLUMNS].mean().dropna()

    plt.figure(figsize=(11, 5), dpi=300)
    ax = means.plot(kind="bar", color="#2E86AB", edgecolor="#1C4E80", linewidth=1.2)

    plt.ylim(0, 100)
    plt.ylabel("Mean Score (0–100)", fontsize=11, fontweight="bold")
    plt.title("NyayaRAG Question Quality — LLM Judge Dimension Evaluation", fontsize=13, fontweight="bold", pad=15)

    # Format x-axis labels with friendly names
    labels = [FRIENDLY_NAMES.get(col, col) for col in means.index]
    plt.xticks(range(len(means)), labels, rotation=35, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Add value annotations on top of each bar
    for i, v in enumerate(means):
        ax.text(i, v + 1.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    # Also save a copy in notebook_3 directory
    local_path = NOTEBOOK_3_DIR / "nyayarag_metric_means.png"
    import shutil
    shutil.copyfile(output_path, local_path)

    return output_path


def display_statistical_summary(df: pd.DataFrame, stats_df: pd.DataFrame, plot_path: Path) -> None:
    """
    Render Rich descriptive statistics table.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 06[/bold bright_white]\n"
            "[bright_cyan]Statistical Analysis & Quality Metric Visualization[/bright_cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    console.print(f"[dim]Computed statistics on {len(df)} successful evaluations.[/dim]\n")

    table = Table(
        title="Descriptive Statistics Across Evaluation Dimensions",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Evaluation Dimension", style="bold cyan")
    table.add_column("Mean", justify="right", style="bold yellow")
    table.add_column("Std", justify="right", style="dim")
    table.add_column("Min", justify="right", style="red")
    table.add_column("25%", justify="right", style="white")
    table.add_column("Median (50%)", justify="right", style="green")
    table.add_column("75%", justify="right", style="white")
    table.add_column("Max", justify="right", style="bold green")

    for col_key, row in stats_df.iterrows():
        name = FRIENDLY_NAMES.get(str(col_key), str(col_key))
        is_overall = "overall" in str(col_key).lower()
        dim_style = f"[bold bright_white]{name}[/bold bright_white]" if is_overall else name
        table.add_row(
            dim_style,
            f"{row['mean']:.2f}",
            f"{row['std']:.2f}",
            f"{row['min']:.1f}",
            f"{row['25%']:.1f}",
            f"{row['50%']:.1f}",
            f"{row['75%']:.1f}",
            f"{row['max']:.1f}",
        )
    console.print(table)

    console.print(
        Panel(
            f"[bold green]Quality Visualization Plot Exported Successfully:[/bold green]\n"
            f"[white]{plot_path}[/white]\n"
            f"• Dimensions Plotted : {len(METRIC_COLUMNS)}\n"
            f"• Average Quality    : {df['overall_quality_score'].mean():.2f} / 100\n"
            f"• Output Resolution  : 300 DPI",
            title="[bold cyan]Visualization Artifact[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def main() -> pd.DataFrame:
    """
    Main execution workflow for Script 06.
    """
    df = load_clean_metrics_dataset()
    stats_df = compute_metric_statistics(df)
    plot_path = generate_and_save_plot(df)
    display_statistical_summary(df, stats_df, plot_path)
    return stats_df


if __name__ == "__main__":
    main()
