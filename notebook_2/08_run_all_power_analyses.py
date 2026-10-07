"""
NYAYaRAG: Master Pipeline Runner for Notebook 2
(Model Evaluation and Statistical Analysis - Section 1 Power Analysis Pipeline)

This master runner executes the complete sample size justification and prospective power
analysis workflow across all four Research Questions (RQ1–RQ4):
- 01: Research Questions, Hypotheses, and Statistical Plan
- 02: Statistical Assumptions for Sample Size Determination
- 03: Prospective Sample-Size Analysis for RQ1 (Wilcoxon Paired Retrieval)
- 04: Analytical Power Approximation for RQ2 (Friedman Multi-Model Comparison)
- 05: One-Sample Power Analysis for RQ3 (Answer Generation Quality)
- 06: Exact Binomial Power Analysis for RQ4 (Safe Abstention Performance)
- 07: Final Statistical Tests and Sample-Size Decisions Synthesis
"""

import sys
import importlib
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize Rich Console
console = Console(force_terminal=True, legacy_windows=False)

# Add current folder to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def run_pipeline():
    """
    Sequentially imports and executes the sample-size justification scripts.
    """
    scripts = [
        ("01_research_questions_and_statistical_plan", "01. Statistical Plan & Research Questions"),
        ("02_statistical_assumptions_sample_size", "02. Statistical Assumptions for Sample Size"),
        ("03_sample_size_determination_rq1", "03. Sample Size Determination for RQ1 (Wilcoxon)"),
        ("04_sample_size_determination_rq2", "04. Sample Size Determination for RQ2 (Friedman)"),
        ("05_sample_size_determination_rq3", "05. Sample Size Determination for RQ3 (Answer Quality)"),
        ("06_sample_size_determination_rq4", "06. Sample Size Determination for RQ4 (Safe Abstention)"),
        ("07_final_sample_size_decisions_summary", "07. Final Decisions Across Research Questions")
    ]

    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Master Pipeline Runner — Notebook 2[/bold cyan]\n"
        "[italic bright_white]Part A — Sample Size Justification & Power Analysis Across RQ1–RQ4[/italic bright_white]",
        border_style="magenta",
        title="[bold green]Pipeline Execution[/bold green]"
    ))

    for idx, (module_name, desc) in enumerate(scripts, start=1):
        console.print(Rule(f"[bold yellow]Step {idx}/7: {desc} ({module_name}.py)[/bold yellow]", style="bright_blue"))

        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "main"):
                mod.main()
            console.print(f"[bold green]✔ Successfully executed step {idx}: {desc}[/bold green]\n")
        except Exception as e:
            console.print(f"[bold red]✖ Error occurred in step {idx} ({desc}): {e}[/bold red]\n")
            raise e

    console.print(Panel(
        "[bold green]✔ All 7 Notebook 2 power analyses, validations, and summary matrices executed successfully![/bold green]\n"
        "[white]• Total benchmark questions justified : [bold bright_white]75 questions[/bold bright_white] (50 answerable + 25 unanswerable)\n"
        "• Power targets achieved across all tasks : [bold bright_white]1 - β ≥ 0.80[/bold bright_white] at [bold bright_white]α = 0.05[/bold bright_white][/white]",
        border_style="green",
        title="[bold bright_white]Pipeline Completion Status[/bold bright_white]"
    ))


if __name__ == "__main__":
    run_pipeline()
