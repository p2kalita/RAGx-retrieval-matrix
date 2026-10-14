"""
=============================================================================
NyayaRAG Generation & Evaluation — Script 02: Qwen3 Generation Setup (Optional)
=============================================================================

This script preserves Sections 2 and 2.1 of the generation notebook (Cells 07-21):
  - Model: Qwen/Qwen3-4B-Instruct-2507
  - Configuration: 4-bit quantisation (NF4/float16), temperature=0.0, max_tokens=600.
  - BitsAndBytesConfig specification for Colab/local GPU deployment.
  - Alternative remote vLLM + ngrok endpoint architecture with OpenAI-compatible client.
  - Safe, environment-aware execution: validates hardware and endpoint availability
    gracefully without blocking execution on non-GPU/offline systems.
=============================================================================
"""

import os
import sys
import time
from typing import Dict, Any, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
# 1. Qwen Generation Model Configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
GENERATION_TEMPERATURE = 0.0
GENERATION_SEED = 42
MAX_NEW_TOKENS = 600
USE_4BIT_QUANTISATION = True

NGROK_BASE_URL = "https://dried-moonscape-violet.ngrok-free.dev/v1"
VLLM_API_KEY = "local-dev-key"


def get_qwen_configuration() -> Dict[str, Any]:
    """Return the frozen Qwen model generation settings (Notebook Cell 08)."""
    return {
        "model_id": MODEL_ID,
        "four_bit_quantisation": USE_4BIT_QUANTISATION,
        "temperature": GENERATION_TEMPERATURE,
        "seed": GENERATION_SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "ngrok_base_url": NGROK_BASE_URL,
        "vllm_api_key": VLLM_API_KEY
    }


def display_qwen_configuration(config: Dict[str, Any]) -> None:
    """Display Qwen generation parameters in a Rich table."""
    table = Table(title="Qwen3-4B-Instruct-2507 Configuration (Colab/vLLM)", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Configured Value", style="green")

    table.add_row("Model Identifier", config["model_id"])
    table.add_row("Four-bit Quantisation", str(config["four_bit_quantisation"]))
    table.add_row("Sampling Temperature", str(config["temperature"]))
    table.add_row("Random Seed", str(config["seed"]))
    table.add_row("Maximum New Tokens", str(config["max_new_tokens"]))
    table.add_row("vLLM Remote Endpoint", config["ngrok_base_url"])

    console.print(table)


# =============================================================================
# 2. Local 4-Bit BitsAndBytes Setup (GPU Hardware-Aware)
# =============================================================================

def build_quantisation_config():
    """
    Build BitsAndBytes 4-bit NF4 configuration.
    (Notebook Cell 10)
    """
    try:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        return quant_config
    except Exception as e:
        console.print(f"[dim yellow]BitsAndBytesConfig notice: {e}[/dim yellow]")
        return None


def validate_gpu_environment():
    """Check GPU availability and memory properties (Notebook Cells 06 & 12)."""
    table = Table(title="GPU Hardware & Execution Environment", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Status / Measurement", style="white")

    cuda_available = torch.cuda.is_available()
    table.add_row("CUDA Acceleration", "[bold green]Available[/bold green]" if cuda_available else "[yellow]Not Available (CPU Runtime)[/yellow]")

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        table.add_row("Device Name", props.name)
        table.add_row("Total VRAM", f"{props.total_memory / (1024**3):.2f} GB")
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        table.add_row("Allocated VRAM", f"{allocated:.2f} GB")
        table.add_row("Reserved VRAM", f"{reserved:.2f} GB")
    else:
        table.add_row("Execution Mode", "Optional Local/Remote Fallback Mode")

    console.print(table)
    return cuda_available


# =============================================================================
# 3. vLLM / OpenAI Remote Client Setup
# =============================================================================

def setup_vllm_client(base_url: str = NGROK_BASE_URL, api_key: str = VLLM_API_KEY):
    """
    Initialize OpenAI client pointing to the remote vLLM endpoint.
    (Notebook Cells 14-15)
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=10.0,
            max_retries=1
        )
        return client
    except ImportError:
        console.print("[dim yellow]OpenAI package not installed. Skipping remote client initialization.[/dim yellow]")
        return None


def test_vllm_endpoint(client) -> bool:
    """Test remote vLLM endpoint reachability."""
    if client is None:
        return False
    try:
        available_models = [item.id for item in client.models.list().data]
        console.print(f"[bold green][OK] vLLM Endpoint reachable. Models: {available_models}[/bold green]")
        return True
    except Exception as e:
        console.print(f"[dim yellow]Remote vLLM endpoint offline or unreachable ({e.__class__.__name__}).[/dim yellow]")
        return False


# =============================================================================
# 4. Main Execution Block
# =============================================================================

def main():
    console.print(Panel(
        "[bold bright_white]NYAYARAG GENERATION & EVALUATION — SCRIPT 02[/bold bright_white]\n"
        "[dim cyan]Qwen3-4B-Instruct-2507 Model Configuration & Environment Setup (Optional/Archive)[/dim cyan]",
        box=box.DOUBLE,
        border_style="bright_blue"
    ))

    # Display model configuration
    config = get_qwen_configuration()
    display_qwen_configuration(config)

    # Validate GPU environment
    console.print("\n[cyan]Checking hardware execution environment...[/cyan]")
    gpu_available = validate_gpu_environment()

    # Attempt BitsAndBytes setup if GPU is present
    if gpu_available:
        quant_config = build_quantisation_config()
        if quant_config:
            console.print("[bold green][OK] BitsAndBytes NF4 4-bit quantisation configured.[/bold green]")

    # Check remote endpoint status
    console.print("\n[cyan]Inspecting remote vLLM + ngrok endpoint configuration...[/cyan]")
    client = setup_vllm_client()
    test_vllm_endpoint(client)

    console.print(Panel(
        "[bold green][OK] Script 02 finished: Qwen model configuration and execution options documented.[/bold green]\n"
        "[dim]Note: In accordance with Notebook 5's execution flow, the primary generation pipeline uses Gemini 3.5 Flash,\n"
        "which is validated and orchestrated in subsequent scripts.[/dim]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
