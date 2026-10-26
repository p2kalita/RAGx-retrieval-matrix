"""
NYAYaRAG: A Legal Research Assistant
Script 09: Master Pipeline Runner — LLM-as-a-Judge Evaluation Suite (Notebook 3 Simulated)

This script serves as the master automation orchestrator for the complete
LLM-as-a-Judge evaluation workflow (Notebook 3 series). It executes scripts
01 through 08 sequentially, tracking runtime statistics, execution status,
and producing a comprehensive master execution dashboard.

Workflow Stages Executed:
- Stage 01: Dataset Acquisition, Loading & Schema Inspection
- Stage 02: Gemini API Client Initialization & Judge Configuration
- Stage 03: Multi-Dimensional Evaluation Rubric & Prompt Builder
- Stage 04: Preferred Reconstruction Batch Evaluation Pipeline
- Stage 05: Evaluation Results Validation, Integrity Check & Export
- Stage 06: Statistical Analysis & Quality Metric Visualization
- Stage 07: Manual Review Screening, Risk Flagging & Audit Manifest
- Stage 08: Multi-Candidate Comparative Evaluation & Preference Validation
"""

from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any

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

PIPELINE_STAGES = [
    {
        "id": "01",
        "script": "01_dataset_loading_and_inspection.py",
        "name": "Dataset Acquisition & Inspection",
        "desc": "Resolves reconstructed queries & validates tabular schema",
    },
    {
        "id": "02",
        "script": "02_gemini_judge_client_setup.py",
        "name": "Gemini Judge Configuration",
        "desc": "Initializes Google GenAI client & verifies JSON capability",
    },
    {
        "id": "03",
        "script": "03_evaluation_rubric_and_prompt_builder.py",
        "name": "Evaluation Rubric & Prompt Builder",
        "desc": "Establishes 7-dimension weights & structured JSON schema",
    },
    {
        "id": "04",
        "script": "04_preferred_reconstruction_evaluation.py",
        "name": "Preferred Question Evaluation",
        "desc": "Executes LLM Judge evaluation on preferred reconstructions",
    },
    {
        "id": "05",
        "script": "05_results_validation_and_export.py",
        "name": "Results Validation & CSV Export",
        "desc": "Verifies score bounds & exports official evaluation CSV",
    },
    {
        "id": "06",
        "script": "06_statistical_analysis_and_visualization.py",
        "name": "Statistical Analysis & Visualization",
        "desc": "Computes dimension statistics & renders publication plot",
    },
    {
        "id": "07",
        "script": "07_manual_review_screening_and_risk_flagging.py",
        "name": "Manual Review Screening & Risk Flagging",
        "desc": "Applies threshold screening & generates human audit list",
    },
    {
        "id": "08",
        "script": "08_candidate_comparison_evaluation.py",
        "name": "Candidate Comparison & Preference Audit",
        "desc": "Evaluates all 3 candidates & validates top preference",
    },
]


def run_stage(stage_info: Dict[str, str], python_bin: str) -> Dict[str, Any]:
    """
    Execute a single pipeline script via subprocess and record metrics.
    """
    script_path = NOTEBOOK_3_DIR / stage_info["script"]

    if not script_path.exists():
        return {
            "id": stage_info["id"],
            "name": stage_info["name"],
            "script": stage_info["script"],
            "status": "FAIL",
            "exit_code": -1,
            "duration": 0.0,
            "error": f"Script not found at: {script_path}",
        }

    start_time = time.time()
    try:
        proc = subprocess.run(
            [python_bin, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        duration = time.time() - start_time
        status = "PASS" if proc.returncode == 0 else "FAIL"

        return {
            "id": stage_info["id"],
            "name": stage_info["name"],
            "script": stage_info["script"],
            "status": status,
            "exit_code": proc.returncode,
            "duration": duration,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "id": stage_info["id"],
            "name": stage_info["name"],
            "script": stage_info["script"],
            "status": "FAIL",
            "exit_code": -1,
            "duration": duration,
            "error": str(e),
        }


def run_master_pipeline() -> None:
    """
    Execute all 8 evaluation pipeline stages sequentially.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 09[/bold bright_white]\n"
            "[bright_yellow]Master End-to-End Evaluation Orchestrator (Scripts 01 through 08)[/bright_yellow]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    python_bin = sys.executable
    console.print(f"[dim]Executing with Python Interpreter: {python_bin}[/dim]")
    console.print(f"[dim]Working Directory: {PROJECT_ROOT}[/dim]\n")

    results: List[Dict[str, Any]] = []
    total_start = time.time()

    for idx, stage in enumerate(PIPELINE_STAGES, start=1):
        console.print(
            f"[bold cyan]Running [{idx}/{len(PIPELINE_STAGES)}]:[/bold cyan] "
            f"[bold white]{stage['name']}[/bold white] ([dim]{stage['script']}[/dim])...",
            end=" ",
        )

        res = run_stage(stage, python_bin)
        results.append(res)

        if res["status"] == "PASS":
            console.print(
                f"[bold green]PASSED[/bold green] "
                f"[dim]({res['duration']:.2f}s)[/dim]"
            )
        else:
            console.print(
                f"[bold red]FAILED[/bold red] "
                f"[dim](Exit code {res['exit_code']}, {res['duration']:.2f}s)[/dim]"
            )
            if "stderr" in res and res["stderr"]:
                console.print(Panel(res["stderr"][:500], title="Error Details", border_style="red"))

    total_duration = time.time() - total_start

    # Render Summary Dashboard
    console.print("\n" + "=" * 80)
    console.print("[bold bright_white]NYAYARAG LLM JUDGE PIPELINE EXECUTION SUMMARY REPORT[/bold bright_white]")
    console.print("=" * 80 + "\n")

    summary_table = Table(
        title="Master Pipeline Execution Manifest",
        box=box.ROUNDED,
        header_style="bold bright_cyan",
    )
    summary_table.add_column("Stage", style="bold cyan", justify="center")
    summary_table.add_column("Script Name", style="white")
    summary_table.add_column("Pipeline Stage Description", style="dim", max_width=42)
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Runtime", justify="right", style="bold yellow")

    all_passed = True
    for s_info, res in zip(PIPELINE_STAGES, results):
        if res["status"] != "PASS":
            all_passed = False
            status_style = "[bold red]FAIL[/bold red]"
        else:
            status_style = "[bold green]PASS[/bold green]"

        summary_table.add_row(
            s_info["id"],
            s_info["script"],
            s_info["desc"],
            status_style,
            f"{res['duration']:.2f}s",
        )
    console.print(summary_table)

    # Master Status Banner
    if all_passed:
        console.print(
            Panel(
                f"[bold bright_green]SUCCESS: All 8 LLM Judge Evaluation stages executed cleanly![/bold bright_green]\n\n"
                f"  * Total Execution Time : {total_duration:.2f} seconds\n"
                f"  * Total Scripts Run    : {len(PIPELINE_STAGES)}\n"
                f"  * Pipeline Artifacts   : All CSV evaluation datasets, visual plots & manual review manifests verified.\n"
                f"  * Official Evaluation  : Benchmarked against Gemini 2.5 Flash LLM Judge.",
                title="[bold green]Master Pipeline Status: FULLY OPERATIONAL[/bold green]",
                border_style="bright_green",
                box=box.DOUBLE,
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]WARNING: One or more pipeline stages encountered failures.[/bold red]\n"
                "Please inspect the detailed logs above for failed stages.",
                title="[bold red]Master Pipeline Status: INCOMPLETE[/bold red]",
                border_style="red",
                box=box.DOUBLE,
            )
        )


def main() -> None:
    """
    Main entry point for Script 09.
    """
    run_master_pipeline()


if __name__ == "__main__":
    main()
