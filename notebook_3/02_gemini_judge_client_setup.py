"""
NYAYaRAG: A Legal Research Assistant
Script 02: Gemini LLM Judge Client Configuration & Model Setup (Notebook 3 Simulated)

This script implements Steps 2 & 3 of the LLM-as-a-Judge workflow:
- Securely configures the Gemini API client via .env or environment variable.
- Verifies model availability (`gemini-2.5-flash` or alternative judge candidates).
- Validates API communication and JSON generation capabilities.
- Provides Rich terminal diagnostics on active model parameters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
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

# Default Judge model specification from Notebook 3 Step 3
DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def load_api_key(project_root: Path = PROJECT_ROOT) -> str:
    """
    Retrieve Gemini API key from .env file or environment variables.
    """
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment or .env file.\n"
            "Please ensure GEMINI_API_KEY is configured in your project root .env file."
        )

    return api_key


def initialize_judge_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Instantiate and return the Google GenAI Client.
    """
    if not api_key:
        api_key = load_api_key()
    client = genai.Client(api_key=api_key)
    return client


def list_available_models(client: genai.Client, limit: int = 10) -> List[Dict[str, str]]:
    """
    Retrieve a list of available Gemini models for evaluation inspection.
    """
    try:
        models = []
        for m in client.models.list():
            models.append({
                "name": m.name,
                "display_name": getattr(m, "display_name", m.name),
            })
            if len(models) >= limit:
                break
        return models
    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not list models ({e})[/bold yellow]")
        return [{"name": DEFAULT_MODEL_NAME, "display_name": "Gemini 2.5 Flash (Default)"}]


def verify_model_capability(
    client: genai.Client, model_name: str = DEFAULT_MODEL_NAME
) -> bool:
    """
    Perform a minimal health check ensuring the judge model supports JSON generation.
    """
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents="Respond with a valid JSON object: {\"status\": \"active\", \"role\": \"llm_judge\"}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        return "status" in resp.text
    except Exception as e:
        console.print(f"[bold red]Verification error on {model_name}: {e}[/bold red]")
        return False


def display_judge_setup(model_name: str, models_list: List[Dict[str, str]], is_healthy: bool) -> None:
    """
    Render Rich diagnostic panel and model catalog.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 02[/bold bright_white]\n"
            "[bright_cyan]Gemini API Client Initialization & Judge Configuration[/bright_cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    status_str = "[bold green]OPERATIONAL (JSON Ready)[/bold green]" if is_healthy else "[bold red]CHECK FAILED[/bold red]"

    diag_table = Table(title="LLM Judge Configuration Parameters", box=box.ROUNDED, header_style="bold magenta")
    diag_table.add_column("Parameter", style="white")
    diag_table.add_column("Configured Value", style="bold yellow")

    diag_table.add_row("Judge Model ID", model_name)
    diag_table.add_row("Client Library", "google-genai (v1.0+)")
    diag_table.add_row("Response Format", "application/json (Structured Output)")
    diag_table.add_row("Sampling Temperature", "0.0 (Deterministic Greedy)")
    diag_table.add_row("Health Verification", status_str)
    console.print(diag_table)

    catalog_table = Table(title="Sample Available Foundation Models", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    catalog_table.add_column("Model ID", style="dim")
    catalog_table.add_column("Display Name", style="white")

    for m in models_list:
        highlight = "[bold green]* SELECTED[/bold green]" if model_name in m["name"] else ""
        catalog_table.add_row(m["name"], f"{m['display_name']} {highlight}")
    console.print(catalog_table)


def main() -> Dict[str, Any]:
    """
    Main execution workflow for Script 02.
    """
    api_key = load_api_key()
    client = initialize_judge_client(api_key)
    model_name = DEFAULT_MODEL_NAME

    models_list = list_available_models(client, limit=6)
    is_healthy = verify_model_capability(client, model_name)

    display_judge_setup(model_name, models_list, is_healthy)

    return {
        "client": client,
        "model_name": model_name,
        "is_healthy": is_healthy,
    }


if __name__ == "__main__":
    main()
