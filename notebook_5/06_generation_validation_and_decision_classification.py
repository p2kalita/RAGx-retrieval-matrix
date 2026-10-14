"""
=============================================================================
NyayaRAG Generation & Evaluation — Script 06: Validation & Decision Classification
=============================================================================

This script implements Sections 7.1 and 7.2 of the generation notebook (Cells 46-50, 60-65):
  - Validates generation integrity:
      * 10 complete records, unique question IDs
      * 100% format validity (LEGAL ANSWER + AUTHORITIES RELIED UPON)
      * No empty answers, verified STOP finish reasons
  - Decision Outcome Classification:
      * 'Correct Answer Decision' : expected 'answerable' & predicted 'ANSWER'
      * 'Correct Abstention'      : expected 'insufficient_evidence' & predicted 'INSUFFICIENT_EVIDENCE'
      * 'False Abstention'        : expected 'answerable' & predicted 'INSUFFICIENT_EVIDENCE'
      * 'Unsafe Answer'           : expected 'insufficient_evidence' & predicted 'ANSWER'
  - Produces consolidated review table and exports:
      evaluation/pilot/generation_gemini/gemini35_pilot_generation_review.csv
=============================================================================
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

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

PROJECT_ROOT = Path(r"D:\_ResearchNotebooks\NyayaRAG")
GENERATION_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "pilot" / "generation_gemini"
GENERATION_RESULTS_FILE = GENERATION_OUTPUT_DIR / "gemini35_pilot_generation_results.csv"
PILOT_REVIEW_FILE = GENERATION_OUTPUT_DIR / "gemini35_pilot_generation_review.csv"


# =============================================================================
# 1. Output Integrity Validation
# =============================================================================

def validate_generation_results(df_results: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that generation outputs satisfy experimental standards.
    (Notebook Cells 46 & 60)
    """
    assert len(df_results) == 10, f"Expected 10 records, found {len(df_results)}"
    assert df_results["question_id"].is_unique, "Duplicate question IDs detected"

    successful_mask = df_results["generation_status"].eq("success")
    failed_count = (~successful_mask).sum()

    invalid_format_count = (successful_mask & ~df_results["format_valid"]).sum()
    empty_answer_count = (
        successful_mask
        & df_results["generated_answer"].fillna("").str.strip().eq("")
    ).sum()

    non_stop_count = (
        successful_mask
        & ~df_results["finish_reason"].fillna("").str.upper().str.contains("STOP")
    ).sum()

    valid_decisions = {"ANSWER", "INSUFFICIENT_EVIDENCE"}
    invalid_decisions = df_results.loc[
        successful_mask & ~df_results["model_decision"].isin(valid_decisions)
    ]

    assert failed_count == 0, f"Found {failed_count} failed generations"
    assert invalid_format_count == 0, f"Found {invalid_format_count} invalid format outputs"
    assert empty_answer_count == 0, f"Found {empty_answer_count} empty answers"
    assert non_stop_count == 0, f"Found {non_stop_count} non-STOP completions"
    assert invalid_decisions.empty, f"Found {len(invalid_decisions)} invalid model decisions"

    summary = {
        "records_count": len(df_results),
        "successful_count": int(successful_mask.sum()),
        "failed_count": int(failed_count),
        "valid_format_count": f"{int(df_results['format_valid'].sum())}/10",
        "empty_answers": int(empty_answer_count),
        "non_stop_count": int(non_stop_count),
        "invalid_decisions": len(invalid_decisions),
        "status": "PASSED"
    }

    table = Table(title="Pilot Generation Output Validation", box=box.ROUNDED)
    table.add_column("Integrity Check", style="cyan")
    table.add_column("Result", style="bold white", justify="right")
    table.add_column("Status", justify="center")

    table.add_row("Total Records", str(summary["records_count"]), "[bold green]VALID[/bold green]")
    table.add_row("Successful API Calls", str(summary["successful_count"]), "[bold green]10/10[/bold green]")
    table.add_row("Format Compliance", summary["valid_format_count"], "[bold green]100%[/bold green]")
    table.add_row("Empty Substantive Answers", str(summary["empty_answers"]), "[bold green]NONE (0)[/bold green]")
    table.add_row("Normal STOP Terminations", str(summary["records_count"] - summary["non_stop_count"]), "[bold green]10/10[/bold green]")

    console.print(table)
    return summary


# =============================================================================
# 2. Decision Outcome Classification
# =============================================================================

def classify_decision(row: pd.Series) -> str:
    """
    Classify model answer/abstention behavior against expert ground truth.
    (Notebook Cell 63)
    """
    expected = row["answerability_label"]
    predicted = row["model_decision"]

    if expected == "answerable" and predicted == "ANSWER":
        return "Correct Answer Decision"
    if expected == "insufficient_evidence" and predicted == "INSUFFICIENT_EVIDENCE":
        return "Correct Abstention"
    if expected == "answerable" and predicted == "INSUFFICIENT_EVIDENCE":
        return "False Abstention"
    if expected == "insufficient_evidence" and predicted == "ANSWER":
        return "Unsafe Answer"
    return "Unclassified"


def build_consolidated_review_table(
    pilot_questions: pd.DataFrame,
    df_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge questions, answerability labels, and model decisions into review dataset.
    (Notebook Cells 63-64)
    """
    # Classify outcomes
    df_merged = pilot_questions[["question_id", "question", "answerability_label"]].merge(
        df_results[[
            "question_id", "model_name", "model_decision", "generated_answer",
            "authorities_text", "format_valid", "finish_reason", "elapsed_seconds"
        ]],
        on="question_id",
        how="left",
        validate="one_to_one"
    )

    df_merged["decision_outcome"] = df_merged.apply(classify_decision, axis=1)

    assert len(df_merged) == 10, "Row count mismatch in consolidated review table"
    assert df_merged["question_id"].is_unique, "Duplicate question IDs found"
    assert df_merged["generated_answer"].notna().all(), "Missing generated answers"
    assert df_merged["decision_outcome"].notna().all(), "Missing decision classifications"

    return df_merged


# =============================================================================
# 3. Export and Rich Display
# =============================================================================

def export_review_table(df_review: pd.DataFrame, output_path: Path = PILOT_REVIEW_FILE) -> None:
    """Export consolidated review table to CSV (Notebook Cell 65)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_review.to_csv(output_path, index=False, encoding="utf-8-sig")
    console.print(f"[bold green][OK] Consolidated review table exported to {output_path}[/bold green]")


def display_review_table(df_review: pd.DataFrame) -> None:
    """Display decision outcomes in a Rich formatted table."""
    table = Table(
        title="Consolidated Pilot Generation Review & Decision Outcomes (10 Questions)",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Question ID", style="bold cyan", no_wrap=True)
    table.add_column("Expert Ground Truth", justify="center")
    table.add_column("Model Decision", justify="center", style="bold yellow")
    table.add_column("Decision Outcome", justify="center")
    table.add_column("Latency (s)", justify="right")

    for _, row in df_review.iterrows():
        outcome = row["decision_outcome"]
        if outcome in ("Correct Answer Decision", "Correct Abstention"):
            outcome_cell = f"[bold green]{outcome}[/bold green]"
        elif outcome == "Unsafe Answer":
            outcome_cell = f"[bold red]{outcome}[/bold red]"
        else:
            outcome_cell = f"[bold magenta]{outcome}[/bold magenta]"

        gt_style = "green" if row["answerability_label"] == "answerable" else "magenta"
        gt_cell = f"[{gt_style}]{row['answerability_label']}[/{gt_style}]"

        table.add_row(
            row["question_id"],
            gt_cell,
            row["model_decision"],
            outcome_cell,
            f"{row['elapsed_seconds']:.2f}"
        )

    console.print(table)

    # Outcome breakdown
    counts = df_review["decision_outcome"].value_counts()
    count_table = Table(title="Decision Classification Breakdown", box=box.ROUNDED)
    table_colors = {
        "Correct Answer Decision": "green",
        "Correct Abstention": "cyan",
        "Unsafe Answer": "red",
        "False Abstention": "yellow"
    }
    count_table.add_column("Decision Outcome Category", style="bold white")
    count_table.add_column("Question Count", justify="right")
    count_table.add_column("Percentage", justify="right")

    for outcome_name, count in counts.items():
        col_style = table_colors.get(outcome_name, "white")
        pct = (count / len(df_review)) * 100
        count_table.add_row(
            f"[{col_style}]{outcome_name}[/{col_style}]",
            str(count),
            f"{pct:.1f}%"
        )

    console.print(count_table)


# =============================================================================
# 4. Main Pipeline Execution
# =============================================================================

def main():
    console.print(Panel(
        "[bold bright_white]NYAYARAG GENERATION & EVALUATION — SCRIPT 06[/bold bright_white]\n"
        "[dim cyan]Output Validation & Answerability Decision Classification[/dim cyan]",
        box=box.DOUBLE,
        border_style="bright_blue"
    ))

    script_dir = Path(__file__).resolve().parent

    # Load questions with revised ground truth from script 03
    s3_path = script_dir / "03_expert_answerability_ground_truth.py"
    spec3 = importlib.util.spec_from_file_location("s3", s3_path)
    s3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(s3)

    s1_path = script_dir / "01_environment_and_benchmark_setup.py"
    spec1 = importlib.util.spec_from_file_location("s1", s1_path)
    s1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(s1)
    pilot_questions, _ = s1.load_pilot_benchmark_and_retrieval()
    df_questions_labelled = s3.assign_answerability_labels(pilot_questions, revised=True)

    # Load generation outputs from script 05
    s5_path = script_dir / "05_evidence_grounded_generation_pipeline.py"
    spec5 = importlib.util.spec_from_file_location("s5", s5_path)
    s5 = importlib.util.module_from_spec(spec5)
    spec5.loader.exec_module(s5)
    df_results = s5.load_or_run_generation(force_regenerate=False)

    # Validate output integrity
    console.print("\n[cyan]Validating generation result integrity...[/cyan]")
    validate_generation_results(df_results)

    # Build review table
    console.print("\n[cyan]Classifying decision outcomes against revised ground truth...[/cyan]")
    df_review = build_consolidated_review_table(df_questions_labelled, df_results)

    # Export review table
    export_review_table(df_review, PILOT_REVIEW_FILE)

    # Display review table and breakdown
    console.print()
    display_review_table(df_review)

    console.print(Panel(
        "[bold green][OK] Script 06 finished: Generation validation & decision classification completed.[/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
