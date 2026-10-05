"""
Section 08: Universal Paragraph-Aware Chunking Strategy — SOLUTION
==================================================================

This module implements the Universal Paragraph-Aware Chunker designed to preserve
coherent paragraph boundaries while enforcing target word windows and overlap
(Notebook Cells 117 to 138).
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    print_header,
    load_processed_corpus,
)

CHUNK_CONFIGS = {
    "chunk_256_30": {"target_size": 256, "overlap_size": 30},
    "chunk_384_40": {"target_size": 384, "overlap_size": 40},
    "chunk_512_50": {"target_size": 512, "overlap_size": 50},
}


def count_words(text: str) -> int:
    """Returns the number of words in a text."""
    return len(re.findall(r"\b\w+\b", str(text)))


def split_into_paragraphs(text: str) -> List[str]:
    """Split judgment text into non-empty paragraphs."""
    paragraphs = re.split(r"\n\s*\n", str(text))
    return [p.strip() for p in paragraphs if p.strip()]


def paragraph_aware_chunker(text: str, target_size: int, overlap_size: int) -> List[Dict[str, Any]]:
    """
    Generate paragraph-aware chunks using word counts.

    Rules:
    1. Preserve complete paragraphs wherever possible.
    2. Allow a chunk to slightly exceed target size when adding the next complete paragraph.
    3. Carry minimum number of trailing paragraphs required to meet desired overlap.
    4. Split only paragraphs that individually exceed target size.
    """
    if overlap_size >= target_size:
        raise ValueError("Overlap must be smaller than target size.")

    original_paragraphs = split_into_paragraphs(text)
    paragraph_units = []

    for paragraph in original_paragraphs:
        words = paragraph.split()
        if len(words) <= target_size:
            paragraph_units.append({
                "text": paragraph,
                "word_count": len(words),
                "was_split": False,
            })
        else:
            start = 0
            while start < len(words):
                segment_words = words[start:start + target_size]
                paragraph_units.append({
                    "text": " ".join(segment_words),
                    "word_count": len(segment_words),
                    "was_split": True,
                })
                start += target_size

    chunks = []
    start_index = 0

    while start_index < len(paragraph_units):
        current_units = []
        current_words = 0
        end_index = start_index

        while end_index < len(paragraph_units):
            unit = paragraph_units[end_index]
            current_units.append(unit)
            current_words += unit["word_count"]
            end_index += 1
            if current_words >= target_size:
                break

        chunk_text = "\n\n".join(unit["text"] for unit in current_units).strip()

        chunks.append({
            "chunk_text": chunk_text,
            "word_count": current_words,
            "paragraph_count": len(current_units),
            "split_paragraph_count": sum(unit["was_split"] for unit in current_units),
            "start_unit": start_index,
            "end_unit": end_index - 1,
        })

        if end_index >= len(paragraph_units):
            break

        overlap_words = 0
        overlap_start = end_index

        while overlap_start > start_index:
            candidate = paragraph_units[overlap_start - 1]
            overlap_words += candidate["word_count"]
            overlap_start -= 1
            if overlap_words >= overlap_size:
                break

        next_start = max(overlap_start, start_index + 1)
        start_index = next_start

    return chunks


def evaluate_sample_configurations(sample_text: str) -> pd.DataFrame:
    """Evaluate candidate chunk configurations on a sample judgment text."""
    results = []
    for name, cfg in CHUNK_CONFIGS.items():
        chunks = paragraph_aware_chunker(sample_text, cfg["target_size"], cfg["overlap_size"])
        word_counts = [c["word_count"] for c in chunks]
        results.append({
            "Configuration": name,
            "Target Size": cfg["target_size"],
            "Overlap": cfg["overlap_size"],
            "Total Chunks": len(chunks),
            "Avg Words/Chunk": round(sum(word_counts) / len(word_counts), 2) if chunks else 0,
            "Min Words": min(word_counts) if chunks else 0,
            "Max Words": max(word_counts) if chunks else 0,
        })
    return pd.DataFrame(results)


def run_section_08() -> Dict[str, Any]:
    """Execute complete Section 08 pipeline."""
    print_header("Section 8: Universal Paragraph-Aware Chunking Strategy")

    corpus_df = load_processed_corpus()
    sample_row = corpus_df.iloc[0]
    sample_text = sample_row["clean_text"]
    sample_words = count_words(sample_text)
    sample_paras = len(split_into_paragraphs(sample_text))

    print(f"Sample Case: '{sample_row['title'][:60]}...'")
    print(f"  - Words: {sample_words:,} | Paragraphs: {sample_paras:,}")

    eval_df = evaluate_sample_configurations(sample_text)
    print("\nCandidate Configuration Comparison on Sample Judgment:")
    print(eval_df.to_string(index=False))

    chunks_384 = paragraph_aware_chunker(sample_text, target_size=384, overlap_size=40)
    print(f"\nInspection of First Generated Chunk (chunk_384_40):")
    print(f"  - Words in chunk : {chunks_384[0]['word_count']}")
    print(f"  - Paragraphs     : {chunks_384[0]['paragraph_count']}")
    preview = chunks_384[0]["chunk_text"][:280].replace("\n", " ")
    print(f"  - Preview        : \"{preview}...\"")

    print("\n Section 8 chunking methodology completed successfully.")
    return {"eval_df": eval_df, "sample_chunks": chunks_384}


if __name__ == "__main__":
    run_section_08()
