"""
NYAYaRAG: A Legal Research Assistant
Script 07: Manual Review Screening & Risk Flagging (Notebook 3 Simulated)

This script implements Step 8 of the LLM-as-a-Judge evaluation workflow:
- Applies multi-criteria rule-based screening to isolate low-confidence or legally risky questions.
- Flagging Criteria:
  1. Overall Quality Score < 75
  2. Indian English Authenticity Score < 75
  3. Reconstruction Plausibility Score < 75
  4. LLM Judge Evaluation Confidence == 'Low'
  5. Non-empty Legal Risk Flags list
- Exports prioritized audit manifest to `evaluation/nyayarag_questions_for_manual_review.csv`.
- Renders detailed Rich audit tables and actionable revision recommendations.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List
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
NOTEBOOK_3_DIR = PROJECT_ROOT / "notebook_3"
INPUT_RESULTS_FILE = PROJECT_ROOT / "evaluation" / "nyayarag_llm_judge_evaluation.csv"
OUTPUT_REVIEW_FILE = PROJECT_ROOT / "evaluation" / "nyayarag_questions_for_manual_review.csv"

REVIEW_COLUMNS: List[str] = [
    "Question_ID",
    "Evaluated_Question",
    "humanization_score",
    "indian_english_authenticity_score",
    "reconstruction_plausibility_score",
    "overall_quality_score",
    "evaluation_confidence",
    "legal_risk_flags",
    "suggested_revision",
    "Needs_Review",
]


def load_successful_results(file_path: Path = INPUT_RESULTS_FILE) -> pd.DataFrame:
    """
    Load evaluated questions and filter for successful model evaluations.
    """
    if not file_path.exists():
        s5_path = NOTEBOOK_3_DIR / "05_results_validation_and_export.py"
        spec5 = importlib.util.spec_from_file_location("s5", s5_path)
        s5 = importlib.util.module_from_spec(spec5)
        spec5.loader.exec_module(s5)
        df = s5.main()
    else:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

    successful = df[df["error"].fillna("").eq("")].copy()
    return successful


def apply_review_screening(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply exact Notebook 3 Cell 18 screening rules to flag questions for judicial expert review.
    """
    review = df.copy()

    # Ensure numeric types
    for col in ["overall_quality_score", "indian_english_authenticity_score", "reconstruction_plausibility_score"]:
        if col in review.columns:
            review[col] = pd.to_numeric(review[col], errors="coerce")

    # Define flag conditions
    c1 = review["overall_quality_score"] < 75
    c2 = review["indian_english_authenticity_score"] < 75
    c3 = review["reconstruction_plausibility_score"] < 75
    c4 = review["evaluation_confidence"].fillna("").astype(str).str.strip().str.lower() == "low"
    c5 = review["legal_risk_flags"].apply(
        lambda x: (isinstance(x, list) and len(x) > 0)
        or (isinstance(x, str) and x.strip() not in ("[]", "", "None"))
    )

    review["Needs_Review"] = c1 | c2 | c3 | c4 | c5

    # Filter to requested review columns that exist in the dataframe
    avail_cols = [c for c in REVIEW_COLUMNS if c in review.columns]
    sorted_review = review[avail_cols].sort_values("overall_quality_score", na_position="first").reset_index(drop=True)
    return sorted_review


def export_manual_review_manifest(df: pd.DataFrame, target_path: Path = OUTPUT_REVIEW_FILE) -> Path:
    """
    Save the manual review candidate questions to CSV.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False, encoding="utf-8-sig")

    # Local copy in notebook_3 directory
    local_path = NOTEBOOK_3_DIR / "nyayarag_questions_for_manual_review.csv"
    df.to_csv(local_path, index=False, encoding="utf-8-sig")

    return target_path


def display_screening_report(df: pd.DataFrame, export_path: Path) -> None:
    """
    Render Rich report detailing flagged questions and review rationales.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 07[/bold bright_white]\n"
            "[bright_cyan]Manual Review Screening, Risk Flagging & Audit Prioritization[/bright_cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    flagged_df = df[df["Needs_Review"] == True]
    accepted_df = df[df["Needs_Review"] == False]

    summary_table = Table(title="Review Screening Outcome Summary", box=box.ROUNDED, header_style="bold magenta")
    summary_table.add_column("Category", style="white")
    summary_table.add_column("Count", justify="right", style="bold yellow")
    summary_table.add_column("Percentage", justify="right", style="bold green")

    summary_table.add_row("Total Questions Screened", str(len(df)), "100.0%")
    summary_table.add_row(
        "Passed Automated Validation",
        str(len(accepted_df)),
        f"{(len(accepted_df)/len(df))*100:.1f}%",
    )
    summary_table.add_row(
        "[bold yellow]Flagged for Manual Legal Audit[/bold yellow]",
        f"[bold yellow]{len(flagged_df)}[/bold yellow]",
        f"{(len(flagged_df)/len(df))*100:.1f}%",
    )
    console.print(summary_table)

    # Detailed Audit Table
    audit_table = Table(
        title="Prioritized Manual Audit Manifest (Sorted by Score)",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    audit_table.add_column("QID", style="bold cyan", justify="center")
    audit_table.add_column("Overall", justify="right", style="bold yellow")
    audit_table.add_column("Indian", justify="right", style="cyan")
    audit_table.add_column("Plausib.", justify="right", style="white")
    audit_table.add_column("Confidence", justify="center", style="magenta")
    audit_table.add_column("Needs Review", justify="center")
    audit_table.add_column("Suggested Revision / Notes", style="dim", max_width=35)

    for _, r in df.iterrows():
        needs_rev = r.get("Needs_Review", False)
        status = "[bold red]FLAGGED[/bold red]" if needs_rev else "[green]ACCEPT[/green]"
        revis = str(r.get("suggested_revision", ""))
        if not revis or revis == "nan" or revis == "None":
            revis = "-"
        audit_table.add_row(
            str(r.get("Question_ID")),
            str(r.get("overall_quality_score")),
            str(r.get("indian_english_authenticity_score")),
            str(r.get("reconstruction_plausibility_score")),
            str(r.get("evaluation_confidence")),
            status,
            revis[:32] + ("..." if len(revis) > 32 else ""),
        )
    console.print(audit_table)

    console.print(
        Panel(
            f"[bold green]Manual Review Dataset Exported:[/bold green]\n"
            f"[white]{export_path}[/white]\n"
            f"• Total Rows Exported : {len(df)}\n"
            f"• Flagged Questions   : {len(flagged_df)}\n"
            f"• Priority            : Ready for Judicial Officer / Legal Expert Review",
            title="[bold cyan]Export Manifest[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def main() -> pd.DataFrame:
    """
    Main execution workflow for Script 07.
    """
    successful_df = load_successful_results()
    review_df = apply_review_screening(successful_df)
    export_path = export_manual_review_manifest(review_df)
    display_screening_report(review_df, export_path)
    return review_df


if __name__ == "__main__":
    main()
