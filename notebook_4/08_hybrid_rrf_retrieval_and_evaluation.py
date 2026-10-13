"""
=============================================================================
NyayaRAG Pilot Evaluation — Script 08: Hybrid BM25 + E5 Retrieval (RRF) & Evaluation
=============================================================================

This script executes Part S, Part T, and Part U of the NyayaRAG pilot evaluation:
  1. Configures parameters: candidate depth k=50, RRF constant k=60, top_k=5.
  2. Implements Reciprocal Rank Fusion (RRF):
       RRF(d) = 1 / (60 + r_BM25(d)) + 1 / (60 + r_E5(d))
  3. Fuses BM25 (lexical) and E5 (semantic) rankings for sample question.
  4. Retrieves hybrid candidates across all 10 pilot questions.
  5. Saves hybrid retrieval results and generates the annotation template.
  6. Loads and validates annotated hybrid dataset (hybrid_pilot_annotations.csv).
  7. Computes Precision@5, Hit@5, MRR, and nDCG@5.
  8. Displays the full 6-model cumulative performance comparison table:
     TF-IDF, BM25, E5, BGE, InLegalBERT, and Hybrid (BM25 + E5).
=============================================================================
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

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
# 1. Directory and Path Configurations
# =============================================================================

BASE_DIR = Path("d:/_ResearchNotebooks/NyayaRAG")
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation" / "pilot"
PILOT_RETRIEVAL_DIR = EVAL_DIR / "retrieval"
PILOT_ANNOTATION_DIR = EVAL_DIR / "annotation"

CHUNKS_FILE = DATA_DIR / "chunks_384_40.parquet"
PILOT_QUESTIONS_FILE = BASE_DIR / "benchmark" / "final" / "pilot_questions.csv"

# Models
BM25_MODEL_FILE = MODELS_DIR / "bm25" / "bm25.pkl"
E5_CONFIG_FILE = MODELS_DIR / "e5" / "e5_config.json"
E5_EMBEDDINGS_FILE = MODELS_DIR / "e5" / "e5_embeddings.npy"

# Outputs
HYBRID_OUTPUT_FILE = PILOT_RETRIEVAL_DIR / "hybrid_pilot_retrieval.csv"
HYBRID_TEMPLATE_FILE = PILOT_ANNOTATION_DIR / "hybrid_annotation_template.csv"
HYBRID_ANNOTATION_FILE = PILOT_ANNOTATION_DIR / "hybrid_pilot_annotations.csv"

RRF_K = 60
HYBRID_CANDIDATES = 50
HYBRID_TOP_K = 5


# =============================================================================
# 2. Resource Loaders (BM25 & E5)
# =============================================================================

def load_bm25_retriever(bm25_path: Path = BM25_MODEL_FILE):
    """Load precomputed BM25Okapi model."""
    import joblib
    console.print(f"[cyan]Loading BM25 model from {bm25_path}...[/cyan]")
    try:
        bm25 = joblib.load(bm25_path)
    except Exception:
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
    console.print("[bold green][OK] BM25 model loaded successfully.[/bold green]")
    return bm25


def load_e5_retriever(
    config_path: Path = E5_CONFIG_FILE,
    embeddings_path: Path = E5_EMBEDDINGS_FILE,
    df_chunks: pd.DataFrame = None
):
    """Load E5 query encoder and embeddings matrix."""
    from sentence_transformers import SentenceTransformer

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    console.print(f"[cyan]Loading E5 embeddings from {embeddings_path}...[/cyan]")
    embeddings = np.load(embeddings_path, mmap_mode="r")

    if df_chunks is not None:
        assert embeddings.shape[0] == len(df_chunks), "Row mismatch in E5 embeddings"

    console.print(f"[cyan]Loading E5 encoder: {config['model_name']}...[/cyan]")
    model = SentenceTransformer(config["model_name"])
    model.max_seq_length = config.get("max_sequence_length", 512)

    console.print("[bold green][OK] E5 model and embeddings loaded successfully.[/bold green]")
    return model, embeddings, config


# =============================================================================
# 3. Base Retrievers: BM25 & E5
# =============================================================================

def retrieve_bm25(query: str, bm25, df_chunks: pd.DataFrame, top_k: int = HYBRID_CANDIDATES) -> pd.DataFrame:
    """Retrieve top candidates using BM25Okapi."""
    tokens = str(query).lower().split()
    scores = np.asarray(bm25.get_scores(tokens), dtype=float)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = df_chunks.iloc[top_indices].copy()
    results["BM25_Score"] = scores[top_indices]
    return results[[
        "global_chunk_id", "judgment_id", "case_title",
        "local_chunk_id", "BM25_Score", "chunk_text"
    ]].reset_index(drop=True)


def retrieve_e5(
    query: str,
    e5_model,
    e5_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    config: dict,
    top_k: int = HYBRID_CANDIDATES
) -> pd.DataFrame:
    """Retrieve top candidates using normalized E5 cosine similarity."""
    formatted_query = f"{config.get('query_prefix', 'query: ')}{str(query).strip()}"
    query_emb = e5_model.encode(
        formatted_query,
        normalize_embeddings=config.get("normalize_embeddings", True),
        convert_to_numpy=True,
        show_progress_bar=False
    )
    query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)

    scores = np.asarray(e5_embeddings @ query_emb, dtype=float)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = df_chunks.iloc[top_indices].copy()
    results["E5_Score"] = scores[top_indices]
    return results[[
        "global_chunk_id", "judgment_id", "case_title",
        "local_chunk_id", "E5_Score", "chunk_text"
    ]].reset_index(drop=True)


# =============================================================================
# 4. Reciprocal Rank Fusion & Hybrid Retrieval
# =============================================================================

def reciprocal_rank_fusion(
    bm25_results: pd.DataFrame,
    e5_results: pd.DataFrame,
    rrf_k: int = RRF_K
) -> Dict[int, float]:
    """
    Combine BM25 and E5 ranked candidates using Reciprocal Rank Fusion.
    Score(d) = sum(1 / (k + rank_m(d)))
    """
    fused_scores = {}

    for rank, row in enumerate(bm25_results.itertuples(index=False), start=1):
        c_id = row.global_chunk_id
        fused_scores.setdefault(c_id, 0.0)
        fused_scores[c_id] += 1.0 / (rrf_k + rank)

    for rank, row in enumerate(e5_results.itertuples(index=False), start=1):
        c_id = row.global_chunk_id
        fused_scores.setdefault(c_id, 0.0)
        fused_scores[c_id] += 1.0 / (rrf_k + rank)

    return fused_scores


def retrieve_hybrid(
    query: str,
    bm25,
    e5_model,
    e5_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    e5_config: dict,
    candidate_k: int = HYBRID_CANDIDATES,
    top_k: int = HYBRID_TOP_K,
    rrf_k: int = RRF_K
) -> pd.DataFrame:
    """
    Retrieve top-k chunks using BM25 + E5 Reciprocal Rank Fusion.
    """
    bm25_cands = retrieve_bm25(query, bm25, df_chunks, top_k=candidate_k)
    e5_cands = retrieve_e5(query, e5_model, e5_embeddings, df_chunks, e5_config, top_k=candidate_k)

    bm25_cands["BM25_Rank"] = range(1, len(bm25_cands) + 1)
    e5_cands["E5_Rank"] = range(1, len(e5_cands) + 1)

    fused_scores = reciprocal_rank_fusion(bm25_cands, e5_cands, rrf_k=rrf_k)
    candidate_ids = list(fused_scores.keys())

    hybrid_df = df_chunks[df_chunks["global_chunk_id"].isin(candidate_ids)].copy()
    hybrid_df["RRF_Score"] = hybrid_df["global_chunk_id"].map(fused_scores)

    bm25_map = dict(zip(bm25_cands["global_chunk_id"], bm25_cands["BM25_Rank"]))
    e5_map = dict(zip(e5_cands["global_chunk_id"], e5_cands["E5_Rank"]))

    hybrid_df["BM25_Rank"] = hybrid_df["global_chunk_id"].map(bm25_map)
    hybrid_df["E5_Rank"] = hybrid_df["global_chunk_id"].map(e5_map)

    hybrid_df = hybrid_df.sort_values(
        by=["RRF_Score", "BM25_Rank", "E5_Rank"],
        ascending=[False, True, True],
        na_position="last"
    ).head(top_k).reset_index(drop=True)

    return hybrid_df[[
        "global_chunk_id", "judgment_id", "case_title",
        "local_chunk_id", "BM25_Rank", "E5_Rank", "RRF_Score", "chunk_text"
    ]]


# =============================================================================
# 5. Validation and Complete Pilot Retrieval
# =============================================================================

def validate_hybrid_sample_retrieval(
    sample_q: str,
    bm25,
    e5_model,
    e5_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    e5_config: dict
) -> pd.DataFrame:
    """Validate hybrid retrieval on sample question."""
    results = retrieve_hybrid(
        sample_q, bm25, e5_model, e5_embeddings, df_chunks, e5_config,
        candidate_k=HYBRID_CANDIDATES, top_k=HYBRID_TOP_K, rrf_k=RRF_K
    )

    expected_cols = [
        "global_chunk_id", "judgment_id", "case_title", "local_chunk_id",
        "BM25_Rank", "E5_Rank", "RRF_Score", "chunk_text"
    ]
    assert len(results) == HYBRID_TOP_K, f"Expected {HYBRID_TOP_K} results"
    assert results.columns.tolist() == expected_cols, "Columns mismatch"
    assert results["RRF_Score"].notna().all(), "Missing RRF scores"
    assert (results["RRF_Score"].to_numpy()[:-1] >= results["RRF_Score"].to_numpy()[1:]).all(), "Scores not descending"
    assert results["global_chunk_id"].is_unique, "Duplicate chunks found"

    console.print(Panel(
        f"[bold cyan]Validation Question:[/bold cyan]\n{sample_q}",
        title="[bold green]Hybrid BM25 + E5 Retrieval Validation (First Question)[/bold green]",
        box=box.ROUNDED
    ))

    for idx, row in results.iterrows():
        bm25_str = f"Rank {int(row['BM25_Rank'])}" if pd.notna(row['BM25_Rank']) else "Not in Top-50"
        e5_str = f"Rank {int(row['E5_Rank'])}" if pd.notna(row['E5_Rank']) else "Not in Top-50"

        console.print(Panel(
            f"[bold]Hybrid Rank {idx + 1}[/bold] | [cyan]RRF Score: {row['RRF_Score']:.6f}[/cyan] | "
            f"[yellow]BM25: {bm25_str}[/yellow] | [magenta]E5: {e5_str}[/magenta]\n"
            f"[bold white]{row['case_title']}[/bold white] ({row['judgment_id']}) "
            f"Chunk: {row['global_chunk_id']}\n\n"
            f"{row['chunk_text'][:250]}...",
            box=box.MINIMAL
        ))

    return results


def retrieve_all_hybrid_pilot(
    df_pilot: pd.DataFrame,
    bm25,
    e5_model,
    e5_embeddings: np.ndarray,
    df_chunks: pd.DataFrame,
    e5_config: dict
) -> pd.DataFrame:
    """Execute hybrid retrieval for all pilot questions."""
    all_results = []
    for _, row in df_pilot.iterrows():
        q_id = row["Question_ID"]
        q_text = row["Question"]

        retrieved = retrieve_hybrid(
            q_text, bm25, e5_model, e5_embeddings, df_chunks, e5_config,
            candidate_k=HYBRID_CANDIDATES, top_k=HYBRID_TOP_K, rrf_k=RRF_K
        ).copy()

        retrieved.insert(0, "Question", q_text)
        retrieved.insert(0, "Question_ID", q_id)
        retrieved["Rank"] = range(1, len(retrieved) + 1)
        all_results.append(retrieved)

    df_all = pd.concat(all_results, ignore_index=True)

    expected_rows = len(df_pilot) * HYBRID_TOP_K
    assert len(df_all) == expected_rows, f"Expected {expected_rows} rows"
    assert (df_all.groupby("Question_ID").size() == HYBRID_TOP_K).all(), "Incomplete question results"
    assert df_all["Question_ID"].nunique() == len(df_pilot), "Missing question IDs"
    assert df_all["RRF_Score"].notna().all(), "Missing RRF scores"
    assert not df_all.duplicated(subset=["Question_ID", "global_chunk_id"]).any(), "Duplicate chunks within question"

    console.print(f"[bold green][OK] Hybrid pilot retrieval completed ({len(df_all)} chunks total).[/bold green]")
    return df_all


def save_hybrid_results_and_template(
    df_results: pd.DataFrame,
    output_file: Path = HYBRID_OUTPUT_FILE,
    template_file: Path = HYBRID_TEMPLATE_FILE
):
    """Save hybrid outputs and create annotation template."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False)
    console.print(f"[bold green][OK] Saved Hybrid pilot retrieval to {output_file}[/bold green]")

    template_file.parent.mkdir(parents=True, exist_ok=True)
    template_df = df_results.copy()
    template_df["Relevance"] = ""
    template_df["Relevance_Level"] = ""
    template_df["Comments"] = ""
    template_df.to_csv(template_file, index=False)
    console.print(f"[bold green][OK] Created Hybrid annotation template at {template_file}[/bold green]")


# =============================================================================
# 6. Evaluation Metrics Computation
# =============================================================================

def load_and_validate_hybrid_annotations(
    annotation_file: Path = HYBRID_ANNOTATION_FILE,
    df_pilot: pd.DataFrame = None
) -> pd.DataFrame:
    """Load and validate human annotations for hybrid retrieval."""
    annotated_df = pd.read_csv(
        annotation_file,
        encoding="utf-8",
        encoding_errors="replace"
    )

    console.print(f"[cyan]Loaded Hybrid Annotations: {len(annotated_df)} rows[/cyan]")

    annotated_df["Binary_Relevance"] = (annotated_df["Relevance"] > 0).astype(int)

    assert set(annotated_df["Binary_Relevance"].unique()).issubset({0, 1}), "Invalid binary labels"
    if df_pilot is not None:
        assert annotated_df["Question_ID"].nunique() == len(df_pilot), "Missing question IDs"

    return annotated_df


def compute_hybrid_metrics(annotated_df: pd.DataFrame, top_k: int = HYBRID_TOP_K):
    """
    Compute Precision@5, Hit@5, MRR, and nDCG@5 for Hybrid (BM25 + E5).
    """
    # Precision@5
    precision_per_q = (
        annotated_df.groupby("Question_ID")["Binary_Relevance"]
        .mean()
        .reset_index(name="Precision@5")
    )
    mean_precision = precision_per_q["Precision@5"].mean()

    # Hit@5
    hit_per_q = (
        annotated_df.groupby("Question_ID")["Binary_Relevance"]
        .max()
        .reset_index(name="Hit@5")
    )
    mean_hit = hit_per_q["Hit@5"].mean()

    # MRR
    def calc_rr(group):
        rel = group[group["Binary_Relevance"] == 1]
        if rel.empty:
            return 0.0
        return 1.0 / rel["Rank"].min()

    mrr_per_q = (
        annotated_df.groupby("Question_ID")
        .apply(calc_rr, include_groups=False)
        .reset_index(name="Reciprocal_Rank")
    )
    mean_mrr = mrr_per_q["Reciprocal_Rank"].mean()

    # nDCG@5
    ndcg_records = []
    for q_id, group in annotated_df.groupby("Question_ID"):
        grp = group.sort_values("Rank")
        true_rel = grp["Relevance"].to_numpy(dtype=float).reshape(1, -1)
        pred_scores = grp["RRF_Score"].to_numpy(dtype=float).reshape(1, -1)
        val = ndcg_score(true_rel, pred_scores, k=top_k)
        ndcg_records.append({"Question_ID": q_id, "nDCG@5": val})

    ndcg_per_q = pd.DataFrame(ndcg_records)
    mean_ndcg = ndcg_per_q["nDCG@5"].mean()

    # Validations
    assert precision_per_q["Precision@5"].between(0, 1).all()
    assert hit_per_q["Hit@5"].isin([0, 1]).all()
    assert mrr_per_q["Reciprocal_Rank"].between(0, 1).all()
    assert ndcg_per_q["nDCG@5"].between(0, 1).all()

    summary_df = pd.DataFrame({
        "Retriever": ["Hybrid (BM25 + E5)"],
        "Precision@5": [mean_precision],
        "Hit@5": [mean_hit],
        "MRR": [mean_mrr],
        "nDCG@5": [mean_ndcg]
    })

    return precision_per_q, hit_per_q, mrr_per_q, ndcg_per_q, summary_df


def display_final_performance_comparison(
    hybrid_summary: pd.DataFrame,
    tfidf_scores=(0.6400, 0.9000, 0.7167, 0.7744),
    bm25_scores=(0.6800, 0.9000, 0.7167, 0.7521),
    e5_scores=(0.5600, 0.8000, 0.7000, 0.6673),
    bge_scores=(0.6400, 0.8000, 0.7333, 0.7072),
    inleg_scores=(0.4800, 0.8000, 0.5750, 0.6449)
) -> pd.DataFrame:
    """Display final 6-model comparison table (Notebook Cell 286)."""
    hyb_p = hybrid_summary["Precision@5"].iloc[0]
    hyb_h = hybrid_summary["Hit@5"].iloc[0]
    hyb_m = hybrid_summary["MRR"].iloc[0]
    hyb_n = hybrid_summary["nDCG@5"].iloc[0]

    final_df = pd.DataFrame({
        "Retriever": [
            "TF-IDF",
            "BM25",
            "E5",
            "BGE",
            "InLegalBERT",
            "Hybrid (BM25 + E5)"
        ],
        "Type": [
            "Lexical",
            "Lexical",
            "Semantic",
            "Semantic",
            "Semantic",
            "Hybrid"
        ],
        "Precision@5": [
            tfidf_scores[0], bm25_scores[0], e5_scores[0],
            bge_scores[0], inleg_scores[0], hyb_p
        ],
        "Hit@5": [
            tfidf_scores[1], bm25_scores[1], e5_scores[1],
            bge_scores[1], inleg_scores[1], hyb_h
        ],
        "MRR": [
            tfidf_scores[2], bm25_scores[2], e5_scores[2],
            bge_scores[2], inleg_scores[2], hyb_m
        ],
        "nDCG@5": [
            tfidf_scores[3], bm25_scores[3], e5_scores[3],
            bge_scores[3], inleg_scores[3], hyb_n
        ]
    }).round(4)

    table = Table(
        title="Final Pilot Retrieval Performance Comparison (All 6 Models)",
        box=box.DOUBLE_EDGE,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Retriever", style="bold cyan", no_wrap=True)
    table.add_column("Paradigm", style="magenta")
    table.add_column("Precision@5", justify="right", style="bold green")
    table.add_column("Hit@5", justify="right", style="bold yellow")
    table.add_column("MRR", justify="right", style="bold white")
    table.add_column("nDCG@5", justify="right", style="bold cyan")

    for _, row in final_df.iterrows():
        is_hybrid = "Hybrid" in row["Retriever"]
        prefix = "[bold bright_white]" if is_hybrid else ""
        suffix = "[/bold bright_white]" if is_hybrid else ""
        table.add_row(
            f"{prefix}{row['Retriever']}{suffix}",
            row["Type"],
            f"{prefix}{row['Precision@5']:.4f}{suffix}",
            f"{prefix}{row['Hit@5']:.4f}{suffix}",
            f"{prefix}{row['MRR']:.4f}{suffix}",
            f"{prefix}{row['nDCG@5']:.4f}{suffix}"
        )

    console.print()
    console.print(table)
    return final_df


# =============================================================================
# 7. Main Pipeline
# =============================================================================

def main():
    console.print(Panel(
        "[bold cyan]NyayaRAG Pilot Evaluation — Script 08[/bold cyan]\n"
        "[white]Hybrid BM25 + E5 Retrieval (RRF k=60) & Overall 6-Model Benchmark[/white]",
        box=box.DOUBLE
    ))

    # Load corpus chunks & pilot benchmark
    df_chunks = pd.read_parquet(CHUNKS_FILE)
    df_pilot = pd.read_csv(PILOT_QUESTIONS_FILE)
    console.print(f"[cyan]Loaded {len(df_chunks):,} corpus chunks and {len(df_pilot)} pilot questions.[/cyan]")

    # Load BM25 and E5 resources
    bm25 = load_bm25_retriever(BM25_MODEL_FILE)
    e5_model, e5_embeddings, e5_config = load_e5_retriever(
        E5_CONFIG_FILE,
        E5_EMBEDDINGS_FILE,
        df_chunks
    )

    # Validate on sample question
    sample_q = df_pilot.iloc[0]["Question"]
    validate_hybrid_sample_retrieval(
        sample_q,
        bm25,
        e5_model,
        e5_embeddings,
        df_chunks,
        e5_config
    )

    # Retrieve all pilot questions
    df_results = retrieve_all_hybrid_pilot(
        df_pilot,
        bm25,
        e5_model,
        e5_embeddings,
        df_chunks,
        e5_config
    )

    # Save results & annotation template
    save_hybrid_results_and_template(
        df_results,
        HYBRID_OUTPUT_FILE,
        HYBRID_TEMPLATE_FILE
    )

    # Load & evaluate human annotations
    annotated_df = load_and_validate_hybrid_annotations(
        HYBRID_ANNOTATION_FILE,
        df_pilot
    )

    p_per_q, h_per_q, m_per_q, n_per_q, hybrid_summary = compute_hybrid_metrics(
        annotated_df,
        top_k=HYBRID_TOP_K
    )

    # Print summary table
    table = Table(title="Hybrid Pilot Retrieval Metrics", box=box.ROUNDED)
    table.add_column("Retriever", style="cyan")
    table.add_column("Precision@5", justify="right", style="green")
    table.add_column("Hit@5", justify="right", style="green")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("nDCG@5", justify="right", style="green")

    table.add_row(
        "Hybrid (BM25 + E5)",
        f"{hybrid_summary['Precision@5'].iloc[0]:.4f}",
        f"{hybrid_summary['Hit@5'].iloc[0]:.4f}",
        f"{hybrid_summary['MRR'].iloc[0]:.4f}",
        f"{hybrid_summary['nDCG@5'].iloc[0]:.4f}"
    )
    console.print(table)

    # Overall 6-model comparison table
    display_final_performance_comparison(hybrid_summary)

    console.print(Panel(
        "[bold green][OK] Hybrid BM25+E5 retrieval and pilot evaluation finished successfully![/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
