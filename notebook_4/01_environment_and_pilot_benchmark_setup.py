"""
Section 1: Environment and Pilot Benchmark Setup
=================================================

Covers Part A (Environment Setup & Project Directories), Part B (Pilot Benchmark Dataset),
Part C (Retrieval Resources & Model Validation), and Part D (Retrieval Configuration)
from Notebook 4 (Cells 001 to 035).

This module configures the project filesystem paths, validates directory access, loads and
validates the 10-question pilot benchmark dataset, inspects required retrieval models and
embeddings on disk, loads the processed corpus chunks, and establishes experimental parameters.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure UTF-8 output encoding across consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Default Project Paths
DEFAULT_PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")


def print_banner(title: str, subtitle: Optional[str] = None, console: Optional[Console] = None) -> None:
    """Print an eye-catching Rich section banner."""
    if console is None:
        console = Console(legacy_windows=False)
    content = f"[bold bright_white]{title}[/bold bright_white]"
    if subtitle:
        content += f"\n[dim cyan]{subtitle}[/dim cyan]"
    console.print()
    console.print(
        Panel(
            content,
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(0, 2),
        )
    )


def setup_project_directories(project_dir: Path = DEFAULT_PROJECT_DIR) -> Dict[str, Path]:
    """
    Configure and create all required directories for the pilot evaluation pipeline.
    (Notebook Cells 013 to 014)
    """
    dirs = {
        "PROJECT_DIR": project_dir,
        "BENCHMARK_DIR": project_dir / "benchmark",
        "DATA_DIR": project_dir / "data",
        "EMBEDDINGS_DIR": project_dir / "embeddings",
        "MODELS_DIR": project_dir / "models",
        "EVALUATION_DIR": project_dir / "evaluation",
        "OUTPUTS_DIR": project_dir / "outputs",
        "RESULTS_DIR": project_dir / "results",
        "PILOT_BENCHMARK_DIR": project_dir / "benchmark" / "final",
        "PILOT_EVALUATION_DIR": project_dir / "evaluation" / "pilot",
        "PILOT_RETRIEVAL_DIR": project_dir / "evaluation" / "pilot" / "retrieval",
        "PILOT_ANNOTATION_DIR": project_dir / "evaluation" / "pilot" / "annotation",
        "PILOT_GENERATION_DIR": project_dir / "evaluation" / "pilot" / "generation",
        "PILOT_RESULTS_DIR": project_dir / "evaluation" / "pilot" / "results",
    }

    # Ensure evaluation directories exist
    for dir_key in [
        "PILOT_EVALUATION_DIR",
        "PILOT_RETRIEVAL_DIR",
        "PILOT_ANNOTATION_DIR",
        "PILOT_GENERATION_DIR",
        "PILOT_RESULTS_DIR",
    ]:
        dirs[dir_key].mkdir(parents=True, exist_ok=True)

    return dirs


def validate_project_directories(dirs: Dict[str, Path], console: Optional[Console] = None) -> None:
    """
    Validate that essential project directories exist and are accessible.
    (Notebook Cells 015 to 016)
    """
    if console is None:
        console = Console(legacy_windows=False)

    table = Table(
        title="[bold bright_cyan]Project Directory Status[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Directory Role", style="bold cyan")
    table.add_column("Path", style="dim white")
    table.add_column("Status", justify="center")

    check_keys = [
        "PROJECT_DIR",
        "BENCHMARK_DIR",
        "DATA_DIR",
        "MODELS_DIR",
        "EVALUATION_DIR",
        "PILOT_RETRIEVAL_DIR",
        "PILOT_ANNOTATION_DIR",
    ]

    for key in check_keys:
        p = dirs[key]
        exists = p.exists()
        status_text = "[bold green][OK] FOUND[/bold green]" if exists else "[bold red][MISSING][/bold red]"
        table.add_row(key, str(p), status_text)

    console.print(table)


def load_and_validate_pilot_benchmark(
    pilot_file: Path,
    expected_size: int = 10,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Load and rigorously validate the 10 legal research questions in the pilot benchmark.
    (Notebook Cells 017 to 024)
    """
    if console is None:
        console = Console(legacy_windows=False)

    if not pilot_file.exists():
        raise FileNotFoundError(f"Pilot questions file missing at: {pilot_file}")

    df_pilot = pd.read_csv(pilot_file)

    # Validations from Cell 023
    actual_size = len(df_pilot)
    unique_ids = df_pilot["Question_ID"].nunique()

    assert actual_size == expected_size, (
        f"Expected {expected_size} pilot questions, but found {actual_size}."
    )
    assert unique_ids == actual_size, (
        "Duplicate Question_ID values were detected in pilot dataset."
    )
    assert df_pilot["Question"].str.strip().ne("").all(), (
        "One or more pilot questions are blank."
    )

    table = Table(
        title=f"[bold bright_magenta]Pilot Benchmark Dataset ({actual_size} Questions)[/bold bright_magenta]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("Q ID", style="bold cyan", justify="center", width=8)
    table.add_column("Research Question", style="white")

    for _, row in df_pilot.iterrows():
        q_id = str(row["Question_ID"])
        q_text = str(row["Question"])
        if len(q_text) > 85:
            q_text = q_text[:82] + "..."
        table.add_row(q_id, q_text)

    console.print(table)
    return df_pilot


def list_directory_tree(directory: Path, max_depth: int = 3, console: Optional[Console] = None) -> None:
    """
    Display directory tree up to max_depth.
    (Notebook Cell 026)
    """
    if console is None:
        console = Console(legacy_windows=False)

    directory = Path(directory)
    if not directory.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        return

    tree_table = Table(
        title=f"[bold yellow]Directory Tree: {directory.name}[/bold yellow]",
        box=box.SIMPLE_HEAVY,
        show_header=False,
    )
    tree_table.add_column("Entry")

    for path in sorted(directory.rglob("*")):
        rel = path.relative_to(directory)
        if len(rel.parts) <= max_depth:
            indent = "  " * (len(rel.parts) - 1)
            marker = "[DIR] " if path.is_dir() else "[FILE]"
            color = "cyan" if path.is_dir() else "white"
            tree_table.add_row(f"{indent}[{color}]{marker} {rel.name}[/{color}]")

    console.print(tree_table)


def get_retrieval_resource_paths(dirs: Dict[str, Path]) -> Dict[str, Path]:
    """
    Define all retrieval resource paths aligned with Notebook Cell 028.
    """
    data_dir = dirs["DATA_DIR"]
    models_dir = dirs["MODELS_DIR"]

    return {
        "CHUNKS_FILE": data_dir / "processed" / "chunks_384_40.parquet",
        "CLEAN_CORPUS_FILE": data_dir / "processed" / "clean_corpus.parquet",
        "TFIDF_DIR": models_dir / "tfidf",
        "TFIDF_VECTORIZER_FILE": models_dir / "tfidf" / "tfidf_vectorizer.pkl",
        "TFIDF_MATRIX_FILE": models_dir / "tfidf" / "tfidf_matrix.npz",
        "BM25_DIR": models_dir / "bm25",
        "BM25_MODEL_FILE": models_dir / "bm25" / "bm25.pkl",
        "BM25_TOKENIZED_CORPUS_FILE": models_dir / "bm25" / "tokenized_corpus.pkl",
        "BGE_DIR": models_dir / "bge",
        "BGE_EMBEDDINGS_FILE": models_dir / "bge" / "bge_embeddings.npy",
        "BGE_CONFIG_FILE": models_dir / "bge" / "bge_config.json",
        "BGE_CHROMA_DIR": models_dir / "bge" / "chromadb",
        "E5_DIR": models_dir / "e5",
        "E5_EMBEDDINGS_FILE": models_dir / "e5" / "e5_embeddings.npy",
        "E5_CONFIG_FILE": models_dir / "e5" / "e5_config.json",
        "E5_CHROMA_DIR": models_dir / "e5" / "chromadb",
        "INLEGALBERT_DIR": models_dir / "inlegalbert",
        "INLEGALBERT_EMBEDDINGS_FILE": models_dir / "inlegalbert" / "inlegalbert_embeddings.npy",
        "INLEGALBERT_CONFIG_FILE": models_dir / "inlegalbert" / "inlegalbert_config.json",
        "INLEGALBERT_CHROMA_DIR": models_dir / "inlegalbert" / "chromadb",
        "RETRIEVAL_METADATA_FILE": models_dir / "retrieval_index_metadata.json",
    }


def validate_required_retrieval_resources(
    res_paths: Dict[str, Path],
    console: Optional[Console] = None,
) -> None:
    """
    Verify all required corpus files, indices, and embeddings exist.
    (Notebook Cells 029 to 030)
    """
    if console is None:
        console = Console(legacy_windows=False)

    table = Table(
        title="[bold bright_green]Retrieval Resource Inventory & Integrity Validation[/bold bright_green]",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_green",
    )
    table.add_column("Resource Identifier", style="bold cyan")
    table.add_column("Disk Location", style="dim white")
    table.add_column("Integrity Gate", justify="center")

    missing = []
    for name, p in res_paths.items():
        exists = p.exists()
        gate = "[bold green][OK] FOUND[/bold green]" if exists else "[bold red][FAIL] MISSING[/bold red]"
        table.add_row(name, str(p), gate)
        if not exists:
            missing.append(name)

    console.print(table)
    if missing:
        raise FileNotFoundError(f"Missing required retrieval resources: {', '.join(missing)}")


def load_and_validate_chunk_dataset(
    chunks_file: Path,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    Load and inspect the 384-word chunk corpus parquet dataset.
    (Notebook Cells 031 to 032)
    """
    if console is None:
        console = Console(legacy_windows=False)

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks dataset missing at: {chunks_file}")

    df_chunks = pd.read_parquet(chunks_file)

    table = Table(box=box.ROUNDED, show_header=False, border_style="bright_blue")
    table.add_column("Property", style="bold cyan", width=25)
    table.add_column("Value", style="white")

    table.add_row("Total Corpus Chunks", f"{len(df_chunks):,}")
    table.add_row("Total Metadata Columns", str(len(df_chunks.columns)))
    table.add_row("Columns Present", ", ".join(df_chunks.columns))
    table.add_row("Memory Consumption", f"{df_chunks.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    console.print(
        Panel(
            table,
            title="[bold bright_cyan] Corpus Chunk Dataset Summary [/bold bright_cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return df_chunks


def configure_retrieval_parameters(
    top_k: int = 5,
    display_k: int = 3,
    console: Optional[Console] = None,
) -> Tuple[int, int]:
    """
    Establish uniform retrieval depths across all models.
    (Notebook Cells 034 to 035)
    """
    if console is None:
        console = Console(legacy_windows=False)

    assert display_k <= top_k, "display_k must be <= top_k"

    console.print(
        Panel(
            f"[bold white]Top-k Evaluation Depth:[/bold white] [bold green]{top_k}[/bold green]\n"
            f"[bold white]Display Inspection Depth:[/bold white] [bold yellow]{display_k}[/bold yellow]\n"
            f"[dim]All sparse, dense, and hybrid models will be evaluated under identical Top-{top_k} constraints.[/dim]",
            title="[bold bright_white on dark_blue] Experimental Retrieval Configuration [/bold bright_white on dark_blue]",
            border_style="bright_blue",
            box=box.ROUNDED,
        )
    )
    return top_k, display_k


def run_setup() -> Dict[str, Any]:
    """Execute complete Environment and Setup pipeline."""
    console = Console(legacy_windows=False)
    print_banner(
        "NYAYaRAG: Pilot Evaluation and Pipeline Validation",
        "Part A-D: Environment, Benchmark Dataset, and Resource Validation",
        console=console,
    )

    dirs = setup_project_directories()
    validate_project_directories(dirs, console=console)

    pilot_file = dirs["PILOT_BENCHMARK_DIR"] / "pilot_questions.csv"
    df_pilot = load_and_validate_pilot_benchmark(pilot_file, console=console)

    res_paths = get_retrieval_resource_paths(dirs)
    validate_required_retrieval_resources(res_paths, console=console)

    df_chunks = load_and_validate_chunk_dataset(res_paths["CHUNKS_FILE"], console=console)
    top_k, display_k = configure_retrieval_parameters(top_k=5, display_k=3, console=console)

    return {
        "directories": dirs,
        "pilot_df": df_pilot,
        "resources": res_paths,
        "chunks_df": df_chunks,
        "top_k": top_k,
        "display_k": display_k,
    }


if __name__ == "__main__":
    run_setup()
