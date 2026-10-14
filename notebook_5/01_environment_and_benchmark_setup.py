"""
=============================================================================
NyayaRAG Generation & Evaluation — Script 01: Environment & Benchmark Setup
=============================================================================

This script implements Sections 1 and 3 of the generation notebook:
  - Establishes project directory paths and reproducibility seed (SEED = 42).
  - Loads the 10-question pilot benchmark (pilot_questions.csv).
  - Loads the Hybrid retrieval results (hybrid_pilot_retrieval.csv, 50 chunks).
  - Standardises column names to lowercase snake_case.
  - Validates schema consistency, uniqueness of question IDs, 5-chunk per-question
    alignment, and absence of empty fields.
  - Displays Rich summary tables and verification panels.
=============================================================================
"""

import os
import sys
import random
from pathlib import Path
from typing import Tuple, Dict, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

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
# 1. Project Root and Path Definitions
# =============================================================================

PROJECT_ROOT = Path(r"D:\_ResearchNotebooks\NyayaRAG")
BENCHMARK_DIR = PROJECT_ROOT / "benchmark" / "final"
RETRIEVAL_DIR = PROJECT_ROOT / "evaluation" / "pilot" / "retrieval"

PILOT_QUESTIONS_FILE = BENCHMARK_DIR / "pilot_questions.csv"
HYBRID_RETRIEVAL_FILE = RETRIEVAL_DIR / "hybrid_pilot_retrieval.csv"

SEED = 42


# =============================================================================
# 2. Reproducibility Configuration
# =============================================================================

def configure_reproducibility(seed: int = SEED) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (Notebook Cell 05)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    table = Table(title="Environment & Reproducibility Settings", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Project Root", str(PROJECT_ROOT))
    table.add_row("Random Seed", str(seed))
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("GPU Device", torch.cuda.get_device_name(0))

    console.print(table)


# =============================================================================
# 3. Data Loading and Column Standardization
# =============================================================================

def standardise_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase snake_case column names (Notebook Cell 24)."""
    result = dataframe.copy()
    result.columns = (
        result.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return result


def load_pilot_benchmark_and_retrieval(
    questions_file: Path = PILOT_QUESTIONS_FILE,
    retrieval_file: Path = HYBRID_RETRIEVAL_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and standardize pilot questions and hybrid retrieval outputs.
    (Notebook Cells 23-24)
    """
    assert questions_file.exists(), f"Pilot-question file not found: {questions_file}"
    assert retrieval_file.exists(), f"Hybrid retrieval file not found: {retrieval_file}"

    pilot_questions = pd.read_csv(questions_file)
    hybrid_retrieval = pd.read_csv(retrieval_file)

    # Standardise column names
    pilot_questions = standardise_columns(pilot_questions)
    hybrid_retrieval = standardise_columns(hybrid_retrieval)

    # Map legacy retrieval column names to standard generation names
    hybrid_retrieval = hybrid_retrieval.rename(
        columns={
            "global_chunk_id": "chunk_id",
            "rrf_score": "retrieval_score",
            "rank": "retrieval_rank"
        }
    )

    return pilot_questions, hybrid_retrieval


# =============================================================================
# 4. Rigorous Schema and Data Integrity Validation
# =============================================================================

def validate_pilot_data(
    pilot_questions: pd.DataFrame,
    hybrid_retrieval: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate pilot benchmark and hybrid retrieval consistency.
    (Notebook Cell 25)
    """
    required_question_columns = {"question_id", "question"}
    required_retrieval_columns = {
        "question_id", "question", "chunk_id", "judgment_id",
        "case_title", "chunk_text", "retrieval_rank", "retrieval_score"
    }

    missing_q_cols = required_question_columns - set(pilot_questions.columns)
    missing_r_cols = required_retrieval_columns - set(hybrid_retrieval.columns)

    assert not missing_q_cols, f"Missing question columns: {sorted(missing_q_cols)}"
    assert not missing_r_cols, f"Missing retrieval columns: {sorted(missing_r_cols)}"

    assert len(pilot_questions) == 10, f"Expected 10 pilot questions, found {len(pilot_questions)}"
    assert pilot_questions["question_id"].is_unique, "Duplicate question IDs found"
    assert len(hybrid_retrieval) == 50, f"Expected 50 retrieval rows, found {len(hybrid_retrieval)}"

    chunks_per_q = hybrid_retrieval.groupby("question_id").size()
    assert chunks_per_q.eq(5).all(), "Every pilot question must have exactly 5 retrieved chunks"

    benchmark_ids = set(pilot_questions["question_id"])
    retrieval_ids = set(hybrid_retrieval["question_id"])
    assert benchmark_ids == retrieval_ids, "Question IDs in benchmark and retrieval files do not match"

    empty_questions = pilot_questions["question"].fillna("").str.strip().eq("").sum()
    empty_chunks = hybrid_retrieval["chunk_text"].fillna("").str.strip().eq("").sum()

    assert empty_questions == 0, f"Found {empty_questions} empty questions"
    assert empty_chunks == 0, f"Found {empty_chunks} empty chunks"

    summary = {
        "pilot_questions_count": len(pilot_questions),
        "unique_questions": pilot_questions["question_id"].nunique(),
        "retrieval_rows": len(hybrid_retrieval),
        "chunks_per_question": f"{chunks_per_q.min()}-{chunks_per_q.max()}",
        "empty_questions": empty_questions,
        "empty_chunks": empty_chunks,
        "ids_match": True,
        "status": "PASSED"
    }

    table = Table(
        title="Pilot Benchmark & Hybrid Retrieval Data Validation",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Validation Check", style="bold cyan")
    table.add_column("Observed Value", style="bold white", justify="right")
    table.add_column("Status", justify="center")

    table.add_row("Pilot Questions Count", str(summary["pilot_questions_count"]), "[bold green]VALID[/bold green]")
    table.add_row("Unique Question IDs", str(summary["unique_questions"]), "[bold green]VALID[/bold green]")
    table.add_row("Hybrid Retrieval Chunks", str(summary["retrieval_rows"]), "[bold green]VALID[/bold green]")
    table.add_row("Chunks per Question", str(summary["chunks_per_question"]), "[bold green]EXACT (5)[/bold green]")
    table.add_row("Empty Questions", str(summary["empty_questions"]), "[bold green]NONE (0)[/bold green]")
    table.add_row("Empty Retrieved Chunks", str(summary["empty_chunks"]), "[bold green]NONE (0)[/bold green]")
    table.add_row("Question ID Set Alignment", "10 / 10 Matching", "[bold green]PASSED[/bold green]")

    console.print(table)
    return summary


def display_pilot_questions(pilot_questions: pd.DataFrame) -> None:
    """Display all pilot questions in a clean Rich table."""
    table = Table(title="Pilot Questions Overview (10 Questions)", box=box.ROUNDED)
    table.add_column("Question ID", style="bold cyan", no_wrap=True)
    table.add_column("Legal Question Text", style="white")

    for _, row in pilot_questions.iterrows():
        table.add_row(row["question_id"], row["question"])

    console.print(table)


# =============================================================================
# 5. Main Execution Block
# =============================================================================

def main():
    console.print(Panel(
        "[bold bright_white]NYAYARAG GENERATION & EVALUATION — SCRIPT 01[/bold bright_white]\n"
        "[dim cyan]Environment Setup, Pilot Benchmark & Hybrid Retrieval Validation[/dim cyan]",
        box=box.DOUBLE,
        border_style="bright_blue"
    ))

    # Configure seeds & check environment
    configure_reproducibility()

    # Load data
    console.print("\n[cyan]Loading pilot questions and hybrid retrieval outputs...[/cyan]")
    pilot_questions, hybrid_retrieval = load_pilot_benchmark_and_retrieval()

    # Validate integrity
    console.print("\n[cyan]Validating pilot dataset schemas and evidence alignment...[/cyan]")
    validate_pilot_data(pilot_questions, hybrid_retrieval)

    # Display pilot questions
    console.print()
    display_pilot_questions(pilot_questions)

    console.print(Panel(
        "[bold green][OK] Script 01 finished successfully: Pilot benchmark and retrieval evidence validated.[/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
