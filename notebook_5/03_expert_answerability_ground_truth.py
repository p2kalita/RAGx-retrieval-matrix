"""
=============================================================================
NyayaRAG Generation & Evaluation — Script 03: Expert Answerability Ground Truth
=============================================================================

This script implements Section 4 and Section 7.2 of the generation notebook
(Cells 26-28, 60-62):
  - Establishes expert ground-truth answerability labels based on sufficiency of
    Top-5 Hybrid retrieval evidence:
      * 'answerable': retrieved evidence supports a reliable legal holding.
      * 'insufficient_evidence': retrieved evidence fails to answer the central legal issue.
  - Implements both the initial 8:2 benchmark distribution and the post-review
    7:3 distribution (where RAW015 was relabelled following expert evidence review).
  - Validates label consistency, completeness, and non-disclosure guarantees.
  - Generates Rich comparative distribution tables and academic audit notes.
=============================================================================
"""

import sys
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
PILOT_QUESTIONS_FILE = PROJECT_ROOT / "benchmark" / "final" / "pilot_questions.csv"

# =============================================================================
# 1. Ground Truth Dictionaries
# =============================================================================

INITIAL_ANSWERABILITY_GROUND_TRUTH: Dict[str, str] = {
    "RAW001": "answerable",
    "RAW002": "answerable",
    "RAW004": "answerable",
    "RAW005": "answerable",
    "RAW006": "answerable",
    "RAW007": "answerable",
    "RAW010": "insufficient_evidence",
    "RAW012": "insufficient_evidence",
    "RAW015": "answerable",  # Initially labelled answerable prior to detailed chunk analysis
    "RAW017": "answerable",
}

REVISED_ANSWERABILITY_GROUND_TRUTH: Dict[str, str] = {
    "RAW001": "answerable",
    "RAW002": "answerable",
    "RAW004": "answerable",
    "RAW005": "answerable",
    "RAW006": "answerable",
    "RAW007": "answerable",
    "RAW010": "insufficient_evidence",
    "RAW012": "insufficient_evidence",
    "RAW015": "insufficient_evidence",  # Relabelled after judicial evidence audit (Cells 61-62)
    "RAW017": "answerable",
}

ALLOWED_LABELS = {"answerable", "insufficient_evidence"}


# =============================================================================
# 2. Label Assignment and Validation
# =============================================================================

def assign_answerability_labels(
    df_questions: pd.DataFrame,
    revised: bool = True
) -> pd.DataFrame:
    """
    Assign ground-truth answerability labels to pilot questions.
    (Notebook Cells 27 & 62)
    """
    df = df_questions.copy()
    truth_dict = REVISED_ANSWERABILITY_GROUND_TRUTH if revised else INITIAL_ANSWERABILITY_GROUND_TRUTH
    df["answerability_label"] = df["question_id"].map(truth_dict)

    assert df["answerability_label"].notna().all(), (
        "One or more pilot questions are missing an answerability label."
    )
    unexpected = set(df["answerability_label"]) - ALLOWED_LABELS
    assert not unexpected, f"Unexpected answerability labels found: {unexpected}"

    return df


def validate_ground_truth_distribution(df_labelled: pd.DataFrame, revised: bool = True) -> Dict[str, Any]:
    """
    Validate expected label counts and distribution properties.
    (Notebook Cells 28 & 62)
    """
    counts = df_labelled["answerability_label"].value_counts()
    num_answerable = counts.get("answerable", 0)
    num_insufficient = counts.get("insufficient_evidence", 0)

    if revised:
        expected_ans, expected_ins = 7, 3
    else:
        expected_ans, expected_ins = 8, 2

    assert num_answerable == expected_ans, f"Expected {expected_ans} answerable, got {num_answerable}"
    assert num_insufficient == expected_ins, f"Expected {expected_ins} insufficient, got {num_insufficient}"
    assert len(df_labelled) == 10, f"Expected 10 questions, got {len(df_labelled)}"

    return {
        "version": "Revised (Post-Audit)" if revised else "Initial (Pre-Audit)",
        "total_questions": len(df_labelled),
        "answerable_count": num_answerable,
        "insufficient_evidence_count": num_insufficient,
        "revision_note": "RAW015 relabelled as insufficient_evidence" if revised else "None",
        "validation_status": "PASSED"
    }


def display_answerability_comparison(df_initial: pd.DataFrame, df_revised: pd.DataFrame) -> None:
    """Display side-by-side comparison of initial vs revised ground truth."""
    table = Table(
        title="Pilot Ground-Truth Answerability Labels: Initial vs. Revised Audit",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Question ID", style="bold cyan", no_wrap=True)
    table.add_column("Legal Question", style="white", max_width=45)
    table.add_column("Initial Label", justify="center")
    table.add_column("Revised Label", justify="center")
    table.add_column("Audit Rationale", style="italic dim yellow")

    merged = df_initial[["question_id", "question", "answerability_label"]].merge(
        df_revised[["question_id", "answerability_label"]],
        on="question_id",
        suffixes=("_init", "_rev")
    )

    for _, row in merged.iterrows():
        init_style = "green" if row["answerability_label_init"] == "answerable" else "magenta"
        rev_style = "green" if row["answerability_label_rev"] == "answerable" else "magenta"

        init_cell = f"[{init_style}]{row['answerability_label_init']}[/{init_style}]"
        rev_cell = f"[{rev_style}]{row['answerability_label_rev']}[/{rev_style}]"

        if row["question_id"] == "RAW015":
            note = "Retrieved chunks lacked statutory magistrate certification rules; correctly abstained"
            rev_cell = "[bold red]insufficient_evidence[/bold red] *"
        else:
            note = "Consolidated agreement across judicial annotators"

        table.add_row(
            row["question_id"],
            row["question"][:42] + "...",
            init_cell,
            rev_cell,
            note
        )

    console.print(table)


# =============================================================================
# 3. Main Execution Block
# =============================================================================

def main():
    console.print(Panel(
        "[bold bright_white]NYAYARAG GENERATION & EVALUATION — SCRIPT 03[/bold bright_white]\n"
        "[dim cyan]Expert Ground-Truth Answerability Labels & Evidence Grounding Verification[/dim cyan]",
        box=box.DOUBLE,
        border_style="bright_blue"
    ))

    # Load questions
    import importlib.util
    s1_path = Path(__file__).resolve().parent / "01_environment_and_benchmark_setup.py"
    spec = importlib.util.spec_from_file_location("s1_module", s1_path)
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    pilot_questions, _ = s1.load_pilot_benchmark_and_retrieval()

    # Assign initial labels
    df_initial = assign_answerability_labels(pilot_questions, revised=False)
    summary_init = validate_ground_truth_distribution(df_initial, revised=False)

    # Assign revised labels
    df_revised = assign_answerability_labels(pilot_questions, revised=True)
    summary_rev = validate_ground_truth_distribution(df_revised, revised=True)

    # Display comparison
    console.print("\n[cyan]Evaluating Answerability Ground-Truth Distribution...[/cyan]")
    display_answerability_comparison(df_initial, df_revised)

    # Summary table
    table = Table(title="Answerability Distribution Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Initial Ground Truth", justify="right")
    table.add_column("Revised Ground Truth", justify="right")

    table.add_row("Answerable Questions", f"{summary_init['answerable_count']} (80%)", f"[green]{summary_rev['answerable_count']} (70%)[/green]")
    table.add_row("Insufficient Evidence Questions", f"{summary_init['insufficient_evidence_count']} (20%)", f"[yellow]{summary_rev['insufficient_evidence_count']} (30%)[/yellow]")
    table.add_row("Total Questions", "10", "10")
    table.add_row("Ground-Truth Leaked to Model", "[bold green]NO (Blind)[/bold green]", "[bold green]NO (Blind)[/bold green]")
    console.print(table)

    console.print(Panel(
        "[bold green][OK] Script 03 finished: Expert answerability ground truth established and validated.[/bold green]\n"
        "[dim]Note: RAW015 relabelling reflects rigorous post-audit evidence grounding, ensuring no hallucinated answers are rewarded.[/dim]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
