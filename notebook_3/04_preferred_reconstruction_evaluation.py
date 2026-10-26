"""
NYAYaRAG: A Legal Research Assistant
Script 04: Preferred Reconstruction Evaluation Pipeline (Notebook 3 Simulated)

This script implements Step 5 of the LLM-as-a-Judge evaluation workflow:
- Iterates over each reconstructed judicial research question.
- Executes the structured LLM evaluation call via Google GenAI (`gemini-2.5-flash`).
- Employs 3-attempt exponential backoff retry logic.
- Enforces strict score bounds [0, 100] and deterministic weighted score recalculation.
- Supports caching: saves results to `evaluation/nyayarag_llm_judge_evaluation.csv`.
- Displays real-time Rich evaluation progress and audit metrics.
"""

from __future__ import annotations

import sys
import time
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
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
OUTPUT_CACHE_FILE = PROJECT_ROOT / "evaluation" / "nyayarag_llm_judge_evaluation.csv"

# Weights from Script 03 / Notebook Step 5
WEIGHTS: Dict[str, float] = {
    "humanization_score": 0.25,
    "indian_english_authenticity_score": 0.20,
    "conversational_naturalness_score": 0.15,
    "experience_level_alignment_score": 0.10,
    "clarity_score": 0.10,
    "legal_terminology_context_score": 0.10,
    "reconstruction_plausibility_score": 0.10,
}


def load_dependencies():
    """Dynamically import helper functions from scripts 01, 02, and 03."""
    s1_path = NOTEBOOK_3_DIR / "01_dataset_loading_and_inspection.py"
    spec1 = importlib.util.spec_from_file_location("s1", s1_path)
    s1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(s1)

    s2_path = NOTEBOOK_3_DIR / "02_gemini_judge_client_setup.py"
    spec2 = importlib.util.spec_from_file_location("s2", s2_path)
    s2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(s2)

    s3_path = NOTEBOOK_3_DIR / "03_evaluation_rubric_and_prompt_builder.py"
    spec3 = importlib.util.spec_from_file_location("s3", s3_path)
    s3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(s3)

    return s1, s2, s3


def safe_text(value: Any) -> str:
    """Safely convert NaN or missing pandas values to string."""
    if pd.isna(value):
        return ""
    return str(value)


def evaluate_row(
    row: pd.Series | Dict[str, Any],
    client: Any,
    model_name: str,
    prompt_template: str,
    weights: Dict[str, float] = WEIGHTS,
) -> Dict[str, Any]:
    """
    Evaluate a single question row using the Gemini LLM Judge.
    Preserves exact Notebook 3 Cell 12 retry logic and score calculation.
    """
    context_columns = [
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

    row_context = {
        col: safe_text(row[col])
        for col in context_columns
        if col in (row.index if hasattr(row, "index") else row)
    }

    if not row_context.get("Preferred_Reconstruction", "").strip():
        return {
            "Question_ID": row_context.get("Question_ID", ""),
            "Evaluated_Question": "",
            "judge_model": model_name,
            "error": "Missing preferred reconstruction",
        }

    # Inject context without tripping format curly brackets
    prompt = prompt_template.replace(
        "{row_context}", json.dumps(row_context, ensure_ascii=False)
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )

            result = json.loads(response.text)

            # Validate score fields and calculate weighted score
            scores = []
            valid = True

            for field, weight in weights.items():
                value = result.get(field)

                if value is None:
                    valid = False
                    continue

                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid score type for {field}")

                if not 0 <= value <= 100:
                    raise ValueError(f"Out-of-range score for {field}")

                scores.append((value, weight))

            if valid:
                result["overall_quality_score"] = round(
                    sum(val * wt for val, wt in scores)
                )
            else:
                result["overall_quality_score"] = None

            result["Question_ID"] = row_context.get("Question_ID", "")
            result["Evaluated_Question"] = row_context.get(
                "Preferred_Reconstruction", ""
            )
            result["judge_model"] = model_name
            result["error"] = ""

            return result

        except Exception as exc:
            if attempt == 2:
                return {
                    "Question_ID": row_context.get("Question_ID", ""),
                    "Evaluated_Question": row_context.get(
                        "Preferred_Reconstruction", ""
                    ),
                    "judge_model": model_name,
                    "error": str(exc)[:500],
                }
            time.sleep(2 ** attempt)

    return {
        "Question_ID": row_context.get("Question_ID", ""),
        "Evaluated_Question": row_context.get("Preferred_Reconstruction", ""),
        "judge_model": model_name,
        "error": "Exhausted retry attempts",
    }


def run_batch_evaluation(
    df: pd.DataFrame,
    client: Any,
    model_name: str,
    prompt_template: str,
    delay_between_requests: float = 1.0,
    force_live: bool = False,
) -> pd.DataFrame:
    """
    Execute batch evaluation over all rows in the dataset.
    Loads from cache if available and force_live is False.
    """
    if not force_live and OUTPUT_CACHE_FILE.exists():
        console.print(f"[bold green][OK] Found existing evaluation cache at {OUTPUT_CACHE_FILE}. Loading...[/bold green]")
        cached_df = pd.read_csv(OUTPUT_CACHE_FILE, encoding="utf-8-sig")
        if len(cached_df) == len(df):
            return cached_df

    console.print(f"[bold cyan]Initiating live LLM Judge batch evaluation for {len(df)} questions...[/bold cyan]")
    results: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating reconstructions...", total=len(df))

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            qid = row.get("Question_ID", f"ROW_{idx}")
            progress.update(task, description=f"[cyan]Evaluating {qid} ({idx}/{len(df)})...")
            res = evaluate_row(row, client, model_name, prompt_template)
            results.append(res)
            progress.advance(task)
            if delay_between_requests > 0 and idx < len(df):
                time.sleep(delay_between_requests)

    scores_df = pd.DataFrame(results)
    OUTPUT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(OUTPUT_CACHE_FILE, index=False, encoding="utf-8-sig")
    console.print(f"[bold green][OK] Evaluation completed and cached to {OUTPUT_CACHE_FILE}[/bold green]")
    return scores_df


def display_evaluation_results(scores_df: pd.DataFrame) -> None:
    """
    Render evaluation results in a formatted Rich table.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 04[/bold bright_white]\n"
            "[bright_cyan]Preferred Reconstruction Evaluation Results Summary[/bright_cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    table = Table(
        title="Evaluated Preferred Reconstructions (Sample Manifest)",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("QID", style="bold cyan", justify="center")
    table.add_column("Human.", justify="right", style="magenta")
    table.add_column("Authentic.", justify="right", style="cyan")
    table.add_column("Natural.", justify="right", style="yellow")
    table.add_column("Clarity", justify="right", style="green")
    table.add_column("Legal", justify="right", style="blue")
    table.add_column("Plausib.", justify="right", style="white")
    table.add_column("Overall", justify="right", style="bold bright_yellow")
    table.add_column("Status / Error", style="bold green", max_width=25)

    for _, r in scores_df.iterrows():
        err = r.get("error", "")
        status = "[green]SUCCESS[/green]" if not err or pd.isna(err) or str(err).strip() == "" else f"[red]{str(err)[:20]}[/red]"
        table.add_row(
            str(r.get("Question_ID")),
            str(r.get("humanization_score")),
            str(r.get("indian_english_authenticity_score")),
            str(r.get("conversational_naturalness_score")),
            str(r.get("clarity_score")),
            str(r.get("legal_terminology_context_score")),
            str(r.get("reconstruction_plausibility_score")),
            f"[bold yellow]{str(r.get('overall_quality_score'))}[/bold yellow]",
            status,
        )
    console.print(table)


def main(force_live: bool = False) -> pd.DataFrame:
    """
    Main execution workflow for Script 04.
    """
    s1, s2, s3 = load_dependencies()

    data_path = s1.resolve_dataset_path()
    df = s1.load_and_validate_dataset(data_path)

    api_key = s2.load_api_key()
    client = s2.initialize_judge_client(api_key)
    model_name = s2.DEFAULT_MODEL_NAME

    prompt_template = s3.EVALUATION_PROMPT

    scores_df = run_batch_evaluation(
        df=df,
        client=client,
        model_name=model_name,
        prompt_template=prompt_template,
        delay_between_requests=1.0,
        force_live=force_live,
    )

    display_evaluation_results(scores_df)
    return scores_df


if __name__ == "__main__":
    # Check if CLI passed --live or --force-live
    force_live_flag = "--force-live" in sys.argv or "--live" in sys.argv
    main(force_live=force_live_flag)
