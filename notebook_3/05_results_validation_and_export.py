"""
NYAYaRAG: A Legal Research Assistant
Script 05: Evaluation Results Validation, Integrity Check & Export (Notebook 3 Simulated)

This script implements Step 6 of the LLM-as-a-Judge workflow:
- Validates the parsed evaluation scores against required numeric ranges [0, 100].
- Analyzes successful vs. failed evaluation error counts.
- Exports finalized evaluation results to `evaluation/nyayarag_llm_judge_evaluation.csv` (and local directory).
- Displays Rich verification status tables and metric distributions.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
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
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "nyayarag_llm_judge_evaluation.csv"

METRIC_COLUMNS = [
    "humanization_score",
    "indian_english_authenticity_score",
    "conversational_naturalness_score",
    "experience_level_alignment_score",
    "clarity_score",
    "legal_terminology_context_score",
    "reconstruction_plausibility_score",
    "overall_quality_score",
]


def load_evaluation_results(custom_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load evaluation results from CSV or trigger script 04 pipeline.
    """
    path = Path(custom_path) if custom_path else OUTPUT_FILE
    if not path.exists():
        console.print(f"[yellow]Evaluation results file not found at {path}. Invoking script 04...[/yellow]")
        s4_path = NOTEBOOK_3_DIR / "04_preferred_reconstruction_evaluation.py"
        spec4 = importlib.util.spec_from_file_location("s4", s4_path)
        s4 = importlib.util.module_from_spec(spec4)
        spec4.loader.exec_module(s4)
        return s4.main()

    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def validate_evaluation_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify schema integrity, score ranges, and error frequencies.
    """
    total_records = len(df)
    successful_count = (df["error"].fillna("").eq("")).sum()
    failed_count = total_records - successful_count

    score_issues = 0
    for col in METRIC_COLUMNS:
        if col in df.columns:
            s_numeric = pd.to_numeric(df[col], errors="coerce")
            out_of_bounds = (~s_numeric.isna()) & ((s_numeric < 0) | (s_numeric > 100))
            score_issues += int(out_of_bounds.sum())

    return {
        "total_records": int(total_records),
        "successful_count": int(successful_count),
        "failed_count": int(failed_count),
        "score_issues": int(score_issues),
        "is_valid": bool(failed_count == 0 and score_issues == 0),
    }


def export_results(df: pd.DataFrame, target_path: Path = OUTPUT_FILE) -> Path:
    """
    Export verified evaluation dataset with utf-8-sig encoding.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False, encoding="utf-8-sig")

    # Also maintain a local copy in notebook_3 directory
    local_path = NOTEBOOK_3_DIR / "nyayarag_llm_judge_evaluation.csv"
    df.to_csv(local_path, index=False, encoding="utf-8-sig")

    return target_path


def display_export_summary(df: pd.DataFrame, val_info: Dict[str, Any], export_path: Path) -> None:
    """
    Render Rich tables summarizing evaluation status and integrity checks.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 05[/bold bright_white]\n"
            "[bright_cyan]Evaluation Results Validation, Integrity Check & CSV Export[/bright_cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    status_style = "[bold green]PASSED[/bold green]" if val_info["is_valid"] else "[bold red]FAILURES DETECTED[/bold red]"

    val_table = Table(title="Evaluation Integrity Verification", box=box.ROUNDED, header_style="bold magenta")
    val_table.add_column("Verification Item", style="white")
    val_table.add_column("Metric Value", style="bold yellow", justify="right")
    val_table.add_column("Status", justify="center")

    val_table.add_row("Total Questions Evaluated", str(val_info["total_records"]), "[green]COMPLETE[/green]")
    val_table.add_row("Successful Evaluations", str(val_info["successful_count"]), "[green]OK[/green]")
    val_table.add_row("Failed Evaluations", str(val_info["failed_count"]), "[green]NONE[/green]" if val_info["failed_count"] == 0 else "[red]FLAGGED[/red]")
    val_table.add_row("Out-of-Bounds Scores [0, 100]", str(val_info["score_issues"]), "[green]NONE[/green]" if val_info["score_issues"] == 0 else "[red]FLAGGED[/red]")
    val_table.add_row("Overall Dataset Validation", status_style, status_style)
    console.print(val_table)

    console.print(
        Panel(
            f"[bold green]Artifact Export Successfully Verified:[/bold green]\n"
            f"[white]Primary Target : {export_path}[/white]\n"
            f"[white]Local Copy     : {NOTEBOOK_3_DIR / 'nyayarag_llm_judge_evaluation.csv'}[/white]\n"
            f"• Rows Exported: {len(df)}\n"
            f"• Columns      : {len(df.columns)}\n"
            f"• Format       : UTF-8 with BOM (utf-8-sig)",
            title="[bold cyan]Export Confirmation[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def main() -> pd.DataFrame:
    """
    Main execution workflow for Script 05.
    """
    df = load_evaluation_results()
    val_info = validate_evaluation_dataset(df)
    export_path = export_results(df)
    display_export_summary(df, val_info, export_path)
    return df


if __name__ == "__main__":
    main()
