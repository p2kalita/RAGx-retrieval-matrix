"""
Section 08: Universal Paragraph-Aware Chunking Strategy — EXERCISE
==================================================================

Overview:
---------
Naively splitting text strictly by character or word count destroys legal continuity:
statutory tests, judicial precedents, and multiphase holdings often span full paragraphs.
In this section, you will develop a Paragraph-Aware Chunker that preserves paragraph
integrity while strictly honoring target window sizes and word-overlap constraints.

Tasks:
------
1. Implement `count_words(text: str) -> int` returning the regex word count (`\\b\\w+\\b`).
2. Implement `split_into_paragraphs(text: str) -> List[str]` splitting on `\\n\\s*\\n`.
3. Implement `paragraph_aware_chunker(text: str, target_size: int, overlap_size: int) -> List[str]`:
   - Accumulate paragraphs until target_size is reached.
   - For oversized paragraphs (> target_size), divide on sentence boundaries.
   - For subsequent chunks, retain an overlap buffer of words from the end of the previous chunk.
4. Implement `evaluate_sample_configurations(sample_text: str) -> pd.DataFrame`
   comparing `chunk_256_30`, `chunk_384_40`, and `chunk_512_50`.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import print_header, load_processed_corpus

CHUNK_CONFIGS = {
    "chunk_256_30": {"target_size": 256, "overlap_size": 30},
    "chunk_384_40": {"target_size": 384, "overlap_size": 40},
    "chunk_512_50": {"target_size": 512, "overlap_size": 50},
}


def count_words(text: str) -> int:
    """
    Count the number of words in a text using word boundary matching.

    Args:
        text: Input string.

    Returns:
        int: Number of words.
    """
    # TODO: Use re.findall(r"\b\w+\b", str(text))
    raise NotImplementedError("Implement count_words")


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into distinct non-empty paragraphs.

    Args:
        text: Cleaned judgment text.

    Returns:
        List[str]: List of trimmed paragraph strings.
    """
    # TODO: Split text using re.split(r"\n\s*\n", str(text)) and filter empty strings
    raise NotImplementedError("Implement split_into_paragraphs")


def paragraph_aware_chunker(text: str, target_size: int, overlap_size: int) -> List[str]:
    """
    Generate paragraph-aware chunks respecting word count limits and overlap.

    Args:
        text: Input text.
        target_size: Desired chunk size in words.
        overlap_size: Overlap size in words between consecutive chunks.

    Returns:
        List[str]: Generated text chunks.
    """
    # TODO: Implement paragraph accumulation with overlap logic:
    # 1. Split text into paragraphs.
    # 2. Accumulate paragraphs while total words < target_size.
    # 3. For oversized paragraphs, split across sentences or word tokens.
    # 4. Form chunk and slide forward retaining overlap_size words for next chunk.
    raise NotImplementedError("Implement paragraph_aware_chunker")


def evaluate_sample_configurations(sample_text: str) -> pd.DataFrame:
    """
    Run candidate chunking configurations on a sample text and summarize statistics.

    Args:
        sample_text: Sample judgment text.

    Returns:
        pd.DataFrame: Table comparing candidate configurations.
    """
    # TODO: Loop over CHUNK_CONFIGS, run paragraph_aware_chunker, and calculate metrics
    raise NotImplementedError("Implement evaluate_sample_configurations")


if __name__ == "__main__":
    print_header("Section 08: Paragraph-Aware Chunking — Exercise Validation")
    try:
        sample_doc = (
            "Paragraph one introduces the appellant.\n\n"
            "Paragraph two contains the substantive legal question under Section 302 IPC. "
            "The High Court erred in rejecting the alibi witness.\n\n"
            "Paragraph three concludes with the acquittal order of the Supreme Court."
        )

        n_words = count_words(sample_doc)
        print(f" [TODO] Counted words in sample doc: {n_words}")
        assert n_words > 20, "Word count should be accurate"

        paras = split_into_paragraphs(sample_doc)
        print(f" [TODO] Split into {len(paras)} paragraphs.")
        assert len(paras) == 3, "Expected exactly 3 paragraphs"

        chunks = paragraph_aware_chunker(sample_doc, target_size=20, overlap_size=5)
        print(f" [TODO] Generated {len(chunks)} chunks with target_size=20.")
        assert len(chunks) >= 2, "Expected at least 2 chunks"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
