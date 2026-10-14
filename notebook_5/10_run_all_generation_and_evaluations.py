"""
NYAYaRAG: A Legal Research Assistant
Script 10: Master Pipeline Runner — Generation and Evaluation Orchestrator

This script serves as the master automation runner for the complete NYAYaRAG
Generation and Evaluation workflow (Notebook 05 series). It executes scripts
01 through 09 sequentially, capturing runtime statistics, execution status,
and producing a comprehensive master execution dashboard.

Workflow Stages Executed:
- Stage 01: Environment & Benchmark Setup (Data loading & schema validation)
- Stage 02: Qwen Architecture Setup (Colab GPU & vLLM fallback configuration)
- Stage 03: Expert Ground Truth Definition (Answerable vs Insufficient Evidence)
- Stage 04: Evidence Packaging & Prompt Engineering (Top-5 chunks + System Prompt)
- Stage 05: Evidence-Grounded Generation Pipeline (Gemini 3.5 Flash-Lite / Cache)
- Stage 06: Generation Validation & Decision Classification (10/10 format check)
- Stage 07: RQ3 Evaluation (Correctness, Faithfulness, Citations, Completeness)
- Stage 08: RQ4 Evaluation (Safe Abstention, 2x2 Confusion Matrix, Precision/Recall)
- Stage 09: Pilot Synthesis & Conclusions (Final results export & certification)
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

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_5_DIR = PROJECT_ROOT / "notebook_5"

PIPELINE_STAGES = [
    {
        "id": "01",
        "script": "01_environment_and_benchmark_setup.py",
        "name": "Environment & Benchmark Setup",
        "desc": "Validates project root, loads pilot queries & retrieval manifest",
    },
    {
        "id": "02",
        "script": "02_qwen_generation_setup_optional.py",
        "name": "Qwen Generation Architecture",
        "desc": "Checks Qwen3-4B quantisation & Colab/vLLM fallback parameters",
    },
    {
        "id": "03",
        "script": "03_expert_answerability_ground_truth.py",
        "name": "Expert Answerability Ground Truth",
        "desc": "Validates 7:3 distribution & documents RAW015 evidence revision",
    },
    {
        "id": "04",
        "script": "04_evidence_packaging_and_prompt_engineering.py",
        "name": "Evidence Packaging & Prompt Engineering",
        "desc": "Packages Top-5 hybrid chunks & applies frozen system prompt",
    },
    {
        "id": "05",
        "script": "05_evidence_grounded_generation_pipeline.py",
        "name": "Evidence-Grounded Generation Pipeline",
        "desc": "Executes Gemini 3.5 Flash-Lite generation & verifies cache",
    },
    {
        "id": "06",
        "script": "06_generation_validation_and_decision_classification.py",
        "name": "Generation Validation & Classification",
        "desc": "Verifies 10/10 format validity & maps decision outcomes",
    },
    {
        "id": "07",
        "script": "07_rq3_evidence_grounded_generation_evaluation.py",
        "name": "RQ3 Generation Quality Evaluation",
        "desc": "Computes Correctness, Faithfulness, Citations & Completeness",
    },
    {
        "id": "08",
        "script": "08_rq4_safe_abstention_and_confusion_matrix.py",
        "name": "RQ4 Safe Abstention & Confusion Matrix",
        "desc": "Calculates 2x2 confusion matrix, precision, recall & F1",
    },
    {
        "id": "09",
        "script": "09_pilot_evaluation_summary_and_conclusions.py",
        "name": "Pilot Synthesis & Academic Conclusions",
        "desc": "Exports pilot_final_results.csv & generates validation manifest",
    },
]


def run_stage(stage_info: Dict[str, str], python_bin: str) -> Dict[str, Any]:
    """
    Execute a single pipeline script via subprocess and record metrics.
    """
    script_path = NOTEBOOK_5_DIR / stage_info["script"]

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
    Run all 9 generation & evaluation scripts sequentially with live dashboard.
    """
    # Header Banner
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG GENERATION & EVALUATION — SCRIPT 10[/bold bright_white]\n"
            "[bright_yellow]Master End-to-End Orchestrator (Scripts 01 through 09)[/bright_yellow]",
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
    console.print("[bold bright_white]NYAYARAG PIPELINE EXECUTION SUMMARY REPORT[/bold bright_white]")
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
                f"[bold bright_green]SUCCESS: All 9 Generation & Evaluation stages executed cleanly![/bold bright_green]\n\n"
                f"  * Total Execution Time : {total_duration:.2f} seconds\n"
                f"  * Total Scripts Run    : {len(PIPELINE_STAGES)}\n"
                f"  * Pipeline Artifacts   : All CSV reports, evaluations, and ground-truth verified.\n"
                f"  * Official Conclusions : Validated for RQ1, RQ2, RQ3, and RQ4 benchmarks.",
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
    Main entry point for Script 10.
    """
    run_master_pipeline()


if __name__ == "__main__":
    main()
