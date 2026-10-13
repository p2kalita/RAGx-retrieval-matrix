"""
=============================================================================
NyayaRAG Pilot Evaluation — Script 11: Master Pipeline Orchestrator
=============================================================================

This master orchestrator runs and validates the complete NyayaRAG pilot evaluation
pipeline across all 10 stages:
  Stage 01: Environment, Benchmark, and Corpus Setup
  Stage 02: TF-IDF Lexical Retrieval
  Stage 03: TF-IDF Metric Computation & Evaluation
  Stage 04: BM25 Probabilistic Lexical Retrieval & Evaluation
  Stage 05: E5 Dense Semantic Retrieval & Evaluation
  Stage 06: BGE Dense Semantic Retrieval & Evaluation
  Stage 07: InLegalBERT Domain-Specific Dense Retrieval & Evaluation
  Stage 08: Hybrid (BM25 + E5) RRF Retrieval & Final 6-Model Benchmark
  Stage 09: Research Question 1 (RQ1) Wilcoxon Signed-Rank Test (Semantic vs Lexical)
  Stage 10: Research Question 2 (RQ2) Friedman Omnibus Test & Post-Hoc Analysis

Usage:
  python notebook_4/11_run_all_pilot_evaluations.py
=============================================================================
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

# Ensure UTF-8 output on Windows if available
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(legacy_windows=False)

SCRIPTS = [
    ("01_environment_and_pilot_benchmark_setup.py", "Stage 01: Benchmark & Corpus Validation"),
    ("02_tfidf_retrieval_and_pilot_evaluation.py", "Stage 02: TF-IDF Query Retrieval & Templates"),
    ("03_tfidf_metrics_and_evaluation.py", "Stage 03: TF-IDF Performance Metric Evaluation"),
    ("04_bm25_retrieval_and_evaluation.py", "Stage 04: BM25 Retrieval & Cumulative Lexical"),
    ("05_e5_dense_retrieval_and_evaluation.py", "Stage 05: E5 Dense Semantic Retrieval & Metrics"),
    ("06_bge_dense_retrieval_and_evaluation.py", "Stage 06: BGE Dense Semantic Retrieval & Metrics"),
    ("07_inlegalbert_retrieval_and_evaluation.py", "Stage 07: InLegalBERT Dense Domain Retrieval"),
    ("08_hybrid_rrf_retrieval_and_evaluation.py", "Stage 08: Hybrid BM25+E5 (RRF) & 6-Model Benchmark"),
    ("09_rq1_statistical_analysis_wilcoxon.py", "Stage 09: RQ1 Statistical Analysis (Wilcoxon)"),
    ("10_rq2_statistical_analysis_friedman.py", "Stage 10: RQ2 Statistical Analysis (Friedman)"),
]


def run_stage(script_name: str, stage_description: str, script_dir: Path) -> Dict[str, any]:
    """Run a single pilot evaluation script as a subprocess."""
    script_path = script_dir / script_name
    cmd = [sys.executable, str(script_path)]

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=script_dir.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    duration = time.time() - start_time

    return {
        "script": script_name,
        "description": stage_description,
        "exit_code": result.returncode,
        "duration": duration,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def main():
    console.print(Panel(
        "[bold bright_white]NYAYARAG PILOT EVALUATION PIPELINE[/bold bright_white]\n"
        "[dim cyan]Master Pipeline Orchestration Across All 10 Evaluation Modules[/dim cyan]",
        box=box.DOUBLE,
        border_style="bright_blue"
    ))

    script_dir = Path(__file__).resolve().parent
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[bold yellow]Executing Pipeline Stages...", total=len(SCRIPTS))

        for script_name, description in SCRIPTS:
            stage_task = progress.add_task(f"Running {script_name}...", total=None)
            res = run_stage(script_name, description, script_dir)
            results.append(res)
            progress.remove_task(stage_task)
            progress.advance(overall_task)

            status_style = "bold green" if res["exit_code"] == 0 else "bold red"
            status_text = "[OK]" if res["exit_code"] == 0 else "[FAILED]"
            console.print(f"[{status_style}]{status_text}[/{status_style}] {description} ({res['duration']:.2f}s)")

            if res["exit_code"] != 0:
                console.print(f"[bold red]Error in {script_name}:[/bold red]\n{res['stderr'][-500:]}")
                break

    # Summary Table
    console.print()
    table = Table(
        title="NyayaRAG Pilot Pipeline Execution Summary Dashboard",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Stage", style="bold cyan")
    table.add_column("Script Name", style="white")
    table.add_column("Execution Time", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    total_time = sum(r["duration"] for r in results)
    all_success = all(r["exit_code"] == 0 for r in results)

    for r in results:
        status_cell = "[bold green]PASS[/bold green]" if r["exit_code"] == 0 else "[bold red]FAIL[/bold red]"
        table.add_row(
            r["description"],
            r["script"],
            f"{r['duration']:.2f}s",
            status_cell
        )

    console.print(table)
    console.print(f"[bold cyan]Total Pipeline Execution Time:[/bold cyan] [bold yellow]{total_time:.2f} seconds[/bold yellow]")

    if all_success:
        console.print(Panel(
            "[bold bright_green]✓ ALL 10 PILOT EVALUATION MODULES EXECUTED SUCCESSFULLY![/bold bright_green]\n\n"
            "• Lexical Retrievers:  TF-IDF & BM25 validated.\n"
            "• Dense Retrievers:    E5, BGE, and InLegalBERT validated.\n"
            "• Hybrid Retriever:    BM25 + E5 (Reciprocal Rank Fusion k=60) validated.\n"
            "• RQ1 Wilcoxon Test:   Evaluated Semantic vs. Lexical paradigms.\n"
            "• RQ2 Friedman Test:   Omnibus test across 6 models and post-hoc Holm comparisons verified.",
            title="[bold green]Pipeline Certification[/bold green]",
            box=box.ROUNDED
        ))
    else:
        console.print(Panel(
            "[bold red]Pipeline encountered errors during execution.[/bold red]",
            box=box.ROUNDED
        ))


if __name__ == "__main__":
    main()
