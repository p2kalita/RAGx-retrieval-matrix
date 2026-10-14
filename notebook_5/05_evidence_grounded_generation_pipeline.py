"""
=============================================================================
NyayaRAG Generation & Evaluation — Script 05: Evidence-Grounded Batch Generation
=============================================================================

This script implements the generation pipeline from Sections 2, 7, and 7.1
(Notebook Cells 29-33, 45, 51-54, 59):
  - Model: models/gemini-3.5-flash-lite
  - Configuration:
      * max_output_tokens: 3000 (accommodates full visible answer)
      * temperature: 0.0 (deterministic, reproducible research runs)
      * thinking_level: minimal (preserves output budget for substantive legal prose)
      * max_retries: 3 with exponential backoff
      * delay: 2.0s between requests
  - Batch generation across all 10 pilot benchmark questions.
  - Dual Execution Mode:
      * Loads verified benchmark results from gemini35_pilot_generation_results.csv
      * Supports full live regeneration via '--regenerate' CLI flag.
  - Output: evaluation/pilot/generation_gemini/gemini35_pilot_generation_results.csv
=============================================================================
"""

import os
import sys
import time
import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv

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

PROJECT_ROOT = Path(r"D:\_ResearchNotebooks\NyayaRAG")
GENERATION_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "pilot" / "generation_gemini"
GENERATION_RESULTS_FILE = GENERATION_OUTPUT_DIR / "gemini35_pilot_generation_results.csv"

MODEL_NAME = "models/gemini-3.5-flash-lite"
MAX_OUTPUT_TOKENS = 3000
GENERATION_TEMPERATURE = 0.0
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 2.0


# =============================================================================
# 1. Gemini Client Initialization
# =============================================================================

def initialize_gemini_client():
    """Load API key from .env and initialize the Google GenAI client."""
    ENV_PATH = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        console.print("[dim yellow]Notice: GEMINI_API_KEY not found in environment or .env.[/dim yellow]")
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        console.print(f"[dim yellow]Warning initializing Gemini client: {e}[/dim yellow]")
        return None


# =============================================================================
# 2. Single Question Generation with Retry Logic
# =============================================================================

def generate_one_answer(
    question_id: str,
    client,
    pilot_questions: pd.DataFrame,
    hybrid_retrieval: pd.DataFrame,
    final_system_prompt: str,
    build_user_prompt_fn,
    parse_response_fn,
    max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    Generate one evidence-grounded answer using the frozen NYAYaRAG prompt.
    (Notebook Cell 59)
    """
    from google.genai import types

    user_prompt = build_user_prompt_fn(question_id, pilot_questions, hybrid_retrieval)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            started = time.perf_counter()

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=final_system_prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    response_mime_type="text/plain",
                    thinking_config=types.ThinkingConfig(thinking_level="minimal")
                )
            )

            elapsed = time.perf_counter() - started
            raw_response = str(response.text or "").strip()
            finish_reason = str(response.candidates[0].finish_reason)
            parsed = parse_response_fn(raw_response)

            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            answer_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
            thinking_tokens = getattr(response.usage_metadata, "thoughts_token_count", None)

            complete = (
                "STOP" in finish_reason.upper()
                and parsed["format_valid"]
                and bool(parsed["generated_answer"].strip())
            )

            if not complete:
                raise ValueError(
                    f"Incomplete/invalid model response: finish_reason={finish_reason}, "
                    f"format_valid={parsed['format_valid']}"
                )

            return {
                "generation_status": "success",
                "question_id": question_id,
                "model_name": MODEL_NAME,
                "temperature": GENERATION_TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "thinking_level": "minimal",
                "raw_model_response": raw_response,
                "model_decision": parsed["model_decision"],
                "generated_answer": parsed["generated_answer"],
                "authorities_text": parsed["authorities_text"],
                "format_valid": parsed["format_valid"],
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "thinking_tokens": thinking_tokens,
                "answer_tokens": answer_tokens,
                "elapsed_seconds": round(elapsed, 3),
                "attempts": attempt,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_message": ""
            }

        except Exception as error:
            last_error = error
            console.print(f"[yellow]{question_id}: attempt {attempt} failed ({error})[/yellow]")
            if attempt < max_retries:
                time.sleep(3 * attempt)

    return {
        "generation_status": "failure",
        "question_id": question_id,
        "model_name": MODEL_NAME,
        "temperature": GENERATION_TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "thinking_level": "minimal",
        "raw_model_response": "",
        "model_decision": "",
        "generated_answer": "",
        "authorities_text": "",
        "format_valid": False,
        "finish_reason": "",
        "prompt_tokens": None,
        "thinking_tokens": None,
        "answer_tokens": None,
        "elapsed_seconds": None,
        "attempts": max_retries,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_message": str(last_error)
    }


# =============================================================================
# 3. Batch Generation Pipeline
# =============================================================================

def run_batch_generation(
    pilot_questions: pd.DataFrame,
    hybrid_retrieval: pd.DataFrame,
    final_system_prompt: str,
    build_user_prompt_fn,
    parse_response_fn,
    output_file: Path = GENERATION_RESULTS_FILE
) -> pd.DataFrame:
    """Execute live batch generation across all 10 pilot questions."""
    client = initialize_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client could not be initialized. Check GEMINI_API_KEY.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating pilot answers via Gemini...", total=len(pilot_questions))

        for q_id in pilot_questions["question_id"].tolist():
            rec = generate_one_answer(
                q_id, client, pilot_questions, hybrid_retrieval,
                final_system_prompt, build_user_prompt_fn, parse_response_fn
            )
            records.append(rec)
            # Incremental save
            pd.DataFrame(records).to_csv(output_file, index=False, encoding="utf-8-sig")
            progress.advance(task)
            time.sleep(DELAY_BETWEEN_REQUESTS)

    results_df = pd.DataFrame(records)
    return results_df


def load_or_run_generation(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load pre-generated results if available and valid; otherwise run generation.
    """
    script_dir = Path(__file__).resolve().parent

    # Dynamically import script 01 and script 04
    s1_path = script_dir / "01_environment_and_benchmark_setup.py"
    spec1 = importlib.util.spec_from_file_location("s1", s1_path)
    s1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(s1)
    pilot_questions, hybrid_retrieval = s1.load_pilot_benchmark_and_retrieval()

    s4_path = script_dir / "04_evidence_packaging_and_prompt_engineering.py"
    spec4 = importlib.util.spec_from_file_location("s4", s4_path)
    s4 = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(s4)

    if GENERATION_RESULTS_FILE.exists() and not force_regenerate:
        console.print(f"[cyan]Loading verified pilot generation results from {GENERATION_RESULTS_FILE}...[/cyan]")
        df_results = pd.read_csv(GENERATION_RESULTS_FILE, encoding="utf-8-sig")
        assert len(df_results) == 10, f"Expected 10 rows in {GENERATION_RESULTS_FILE}"
        assert df_results["generation_status"].eq("success").all(), "Non-successful records present"
        console.print("[bold green][OK] Loaded 10 verified generation records.[/bold green]")
        return df_results

    console.print("[bold yellow]Executing live batch generation via Gemini API...[/bold yellow]")
    df_results = run_batch_generation(
        pilot_questions,
        hybrid_retrieval,
        s4.FINAL_SYSTEM_PROMPT,
        s4.build_final_user_prompt,
        s4.parse_generation_response,
        output_file=GENERATION_RESULTS_FILE
    )
    return df_results


# =============================================================================
# 4. Main Pipeline Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NyayaRAG Batch Generation Pipeline")
    parser.add_argument("--regenerate", action="store_true", help="Force live API regeneration")
    args = parser.parse_args()

    console.print(Panel(
        "[bold bright_white]NYAYARAG GENERATION & EVALUATION — SCRIPT 05[/bold bright_white]\n"
        "[dim cyan]Evidence-Grounded Batch Answer Generation (Gemini 3.5 Flash Lite)[/dim cyan]",
        box=box.DOUBLE,
        border_style="bright_blue"
    ))

    # Display generation config
    table = Table(title="Batch Generation Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", MODEL_NAME)
    table.add_row("Max Output Tokens", str(MAX_OUTPUT_TOKENS))
    table.add_row("Temperature", str(GENERATION_TEMPERATURE))
    table.add_row("Thinking Level", "minimal")
    table.add_row("Delay Between Requests", f"{DELAY_BETWEEN_REQUESTS}s")
    table.add_row("Destination File", str(GENERATION_RESULTS_FILE))
    console.print(table)

    # Load or run generation
    df_results = load_or_run_generation(force_regenerate=args.regenerate)

    # Display generation status overview
    summary_table = Table(
        title="Batch Generation Results Overview (10 Questions)",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_blue"
    )
    summary_table.add_column("Question ID", style="bold cyan")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Decision", style="bold yellow")
    summary_table.add_column("Format Valid", justify="center")
    summary_table.add_column("Latency (s)", justify="right")
    summary_table.add_column("Finish Reason", style="dim")

    for _, row in df_results.iterrows():
        status_str = "[bold green]success[/bold green]" if row["generation_status"] == "success" else "[bold red]failure[/bold red]"
        fmt_str = "[bold green]True[/bold green]" if row["format_valid"] else "[bold red]False[/bold red]"
        summary_table.add_row(
            row["question_id"],
            status_str,
            row["model_decision"],
            fmt_str,
            f"{row['elapsed_seconds']:.2f}",
            str(row["finish_reason"])
        )

    console.print(summary_table)

    console.print(Panel(
        "[bold green][OK] Script 05 finished: Evidence-grounded generation pipeline verified.[/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
