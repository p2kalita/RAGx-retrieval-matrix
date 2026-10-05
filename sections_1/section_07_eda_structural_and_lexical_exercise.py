"""
Section 07: EDA — Structural, Lexical, and Statutory Profiling — EXERCISE
=========================================================================

Overview:
---------
In this section, you will extract deep structural and linguistic profiles of the
Supreme Court of India judgment corpus:
1. Sentence and paragraph distributions.
2. Words-per-paragraph metrics.
3. Corpus vocabulary and top legal terminology.
4. Statute citation frequencies across Indian jurisprudence.
5. Lexical diversity (Type-Token Ratio: TTR) per judgment.

Tasks:
------
1. Implement `compute_structural_statistics(df: pd.DataFrame) -> pd.DataFrame`
   calculating sentence count, paragraph count, and avg_words_per_paragraph for each judgment.
2. Implement `extract_top_legal_terms(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame`
   counting the occurrence of standard legal keywords (court, section, act, appellant, etc.).
3. Implement `compute_statute_frequencies(df: pd.DataFrame) -> pd.DataFrame`
   using regex patterns to count mentions of major Indian statutes (Constitution, IPC, CrPC, etc.).
4. Implement `compute_lexical_diversity(text: str) -> float`
   calculating Type-Token Ratio (unique words / total words).

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import print_header, load_processed_corpus


def compute_structural_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentence count, paragraph count, and words-per-paragraph for each judgment.

    Args:
        df: DataFrame with 'clean_text'.

    Returns:
        pd.DataFrame: DataFrame with new columns:
            - 'sentence_count'
            - 'paragraph_count'
            - 'avg_words_per_paragraph'
    """
    # TODO: Split text into sentences using re.split(r'(?<=[.!?])\s+', text)
    # Split text into paragraphs using re.split(r'\n\s*\n', text)
    # Calculate avg_words_per_paragraph = word_count / max(1, paragraph_count)
    raise NotImplementedError("Implement compute_structural_statistics")


def extract_top_legal_terms(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """
    Count the frequency of primary legal terms across the corpus.

    Args:
        df: DataFrame with 'clean_text'.
        top_k: Number of top terms to return.

    Returns:
        pd.DataFrame: Table of ['Legal Term', 'Frequency'].
    """
    # TODO: Tokenize words and count occurrences of domain terms:
    # ['court', 'section', 'act', 'state', 'case', 'order', 'appellant', 'respondent',
    #  'appeal', 'judgment', 'high court', 'law', 'constitution', 'counsel', 'jurisdiction']
    raise NotImplementedError("Implement extract_top_legal_terms")


def compute_statute_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count mentions of major Indian statutes using regular expressions.

    Args:
        df: DataFrame with 'clean_text'.

    Returns:
        pd.DataFrame: Table of ['Statute', 'Total Mentions', 'Judgments Citing'].
    """
    # TODO: Define regex patterns for statutes (Constitution, IPC, CrPC, CPC, Evidence Act, NI Act, etc.)
    # Count total occurrences across all text and the number of documents citing each statute
    raise NotImplementedError("Implement compute_statute_frequencies")


def compute_lexical_diversity(text: str) -> float:
    """
    Calculate Type-Token Ratio (unique alphabetic words / total alphabetic words).

    Args:
        text: Input judgment text.

    Returns:
        float: TTR score between 0.0 and 1.0.
    """
    # TODO: Extract words using re.findall(r"\b[a-zA-Z]+\b", text.lower())
    # Return len(set(words)) / len(words) if words else 0.0
    raise NotImplementedError("Implement compute_lexical_diversity")


if __name__ == "__main__":
    print_header("Section 07: EDA Structural and Lexical — Exercise Validation")
    try:
        sample_text = "The Supreme Court dismissed the appeal under Section 302 of the IPC.\n\nThe judgment was affirmed."
        ttr = compute_lexical_diversity(sample_text)
        print(f" [TODO] Lexical diversity sample: {ttr:.4f}")
        assert 0.0 < ttr <= 1.0, "TTR must be between 0 and 1"

        corpus_df = load_processed_corpus().head(10)
        struct_df = compute_structural_statistics(corpus_df)
        print(f" [TODO] Sentences and paragraphs computed for {len(struct_df)} records.")
        assert "sentence_count" in struct_df.columns, "sentence_count column required"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
