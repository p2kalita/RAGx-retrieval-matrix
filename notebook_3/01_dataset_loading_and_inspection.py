"""
NYAYaRAG: A Legal Research Assistant
Script 01: Dataset Acquisition, Loading, and Schema Validation (Notebook 3 Simulated)

This script implements Step 1 of the LLM-as-a-Judge evaluation workflow:
- Resolves and loads the reconstructed legal questions benchmark dataset.
- Replaces interactive notebook file uploads with automated filesystem resolution.
- Validates schema structure, columns, and data types.
- Provides Rich terminal dashboards summarizing question counts, provenance, and experience bands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
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

REQUIRED_COLUMNS: List[str] = [
    "Question_ID",
    "Experience",
    "Visible_Fragment",
    "Candidate_1",
    "Candidate_2",
    "Candidate_3",
    "Preferred_Reconstruction",
    "Confidence",
    "Reasoning",
    "Verification_Status",
    "Data_Provenance",
]


def resolve_dataset_path(custom_path: Optional[str] = None, project_root: Path = PROJECT_ROOT) -> Path:
    """
    Resolve the benchmark dataset path with fallback locations.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    candidates = [
        project_root / "evaluation" / "nyayarag_reconstructed_questions.csv",
        project_root / "benchmark" / "final" / "nyayarag_reconstructed_questions.csv",
        project_root / "notebook_3" / "nyayarag_reconstructed_questions.csv",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Reconstructed questions dataset not found in candidate locations:\n"
        + "\n".join(f" - {c}" for c in candidates)
    )


def load_and_validate_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV and validate its schema.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    return df


def display_dataset_summary(df: pd.DataFrame, source_path: Path) -> None:
    """
    Display Rich formatted summary tables and preview.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 01[/bold bright_white]\n"
            "[bright_cyan]Dataset Acquisition, Loading & Schema Inspection[/bright_cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    console.print(f"[dim]Loaded dataset from: {source_path}[/dim]\n")

    # Overview table
    summary_table = Table(title="Dataset Schema Overview", box=box.ROUNDED, header_style="bold magenta")
    summary_table.add_column("Metric", style="white")
    summary_table.add_column("Value", style="bold yellow", justify="right")

    summary_table.add_row("Total Questions / Rows", str(len(df)))
    summary_table.add_row("Total Columns", str(len(df.columns)))
    summary_table.add_row("Unique Question IDs", str(df["Question_ID"].nunique()))
    summary_table.add_row("Null Values in Preferred", str(df["Preferred_Reconstruction"].isna().sum()))
    summary_table.add_row("Data Provenance", str(df["Data_Provenance"].unique().tolist()))
    console.print(summary_table)

    # Experience Distribution
    exp_table = Table(title="Judicial Contributor Experience Breakdown", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    exp_table.add_column("Experience Band", style="white")
    exp_table.add_column("Question Count", justify="right", style="bold green")
    exp_table.add_column("Percentage", justify="right", style="bold bright_yellow")

    exp_counts = df["Experience"].value_counts()
    for band, count in exp_counts.items():
        pct = (count / len(df)) * 100
        exp_table.add_row(str(band), str(count), f"{pct:.1f}%")
    console.print(exp_table)

    # Preview Table
    preview_table = Table(
        title="Reconstructed Questions Sample Preview",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    preview_table.add_column("QID", style="bold cyan", justify="center")
    preview_table.add_column("Visible Fragment", style="dim", max_width=32)
    preview_table.add_column("Preferred Reconstruction", style="green", max_width=45)
    preview_table.add_column("Confidence", justify="center", style="yellow")
    preview_table.add_column("Status", justify="center", style="magenta")

    for _, row in df.head(5).iterrows():
        preview_table.add_row(
            str(row["Question_ID"]),
            str(row["Visible_Fragment"])[:30] + "...",
            str(row["Preferred_Reconstruction"])[:42] + "...",
            str(row["Confidence"]),
            str(row["Verification_Status"]),
        )
    console.print(preview_table)


def main() -> pd.DataFrame:
    """
    Main execution workflow for Script 01.
    """
    data_path = resolve_dataset_path()
    df = load_and_validate_dataset(data_path)
    display_dataset_summary(df, data_path)
    return df


if __name__ == "__main__":
    main()
