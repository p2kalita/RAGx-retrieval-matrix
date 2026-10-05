"""
Section 05: Text Cleaning and Preprocessing Pipeline — EXERCISE
==============================================================

Overview:
---------
PDF extraction introduces OCR/typesetting artifacts, including broken hyphenated words
across lines, inconsistent newline characters, tab characters, and irregular spacing.
However, legal judgments require paragraph structure, statutory citations, case titles,
and quotes to be strictly preserved.

In this section, you will implement a rule-based legal text cleaner that removes
formatting noise while safeguarding structural and semantic content.

Tasks:
------
1. Implement `clean_legal_text(text: str) -> str` according to the 9-stage legal cleaning
   rules developed in NYAYaRAG:
   - Normalize line breaks (\\r\\n and \\r to \\n).
   - Replace tabs with spaces.
   - Strip leading/trailing whitespace per line.
   - Temporarily preserve paragraph breaks (\\n{2,} -> <PARAGRAPH_BREAK>).
   - Recombine hyphenated line splits (e.g. 'in-\\nlaw' -> 'in-law').
   - Replace single remaining line breaks with a single space.
   - Restore paragraph breaks (<PARAGRAPH_BREAK> -> \\n\\n).
   - Collapse multiple spaces into one.
   - Strip whitespace around paragraph breaks.
2. Implement `build_and_clean_corpus(metadata_df: pd.DataFrame, text_dir: Path) -> pd.DataFrame`
   loading raw texts and adding 'clean_text'.
3. Implement `compute_cleaning_metrics(corpus_df: pd.DataFrame) -> pd.DataFrame`
   calculating raw vs cleaned character and word counts and their reduction percentages.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import TEXT_DIR, print_header, load_raw_metadata


def clean_legal_text(text: str) -> str:
    """
    Standardize extracted legal text while preserving legal meaning,
    paragraph boundaries, citations, statutory references, and punctuation.

    Args:
        text: Raw text extracted from a judgment PDF.

    Returns:
        str: Cleaned, normalized legal text.
    """
    # TODO: Implement the 9-step text cleaning pipeline:
    # 1. if not isinstance(text, str): return ""
    # 2. text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 3. text = text.replace("\t", " ")
    # 4. lines = [line.strip() for line in text.split("\n")]; text = "\n".join(lines)
    # 5. text = re.sub(r"\n{2,}", " <PARAGRAPH_BREAK> ", text)
    # 6. text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", text)
    # 7. text = re.sub(r"\n", " ", text)
    # 8. text = text.replace("<PARAGRAPH_BREAK>", "\n\n")
    # 9. text = re.sub(r"[ ]{2,}", " ", text)
    # 10. text = re.sub(r" *\n\n *", "\n\n", text).strip()
    raise NotImplementedError("Implement clean_legal_text")


def build_and_clean_corpus(
    metadata_df: pd.DataFrame, text_dir: Path = TEXT_DIR, max_docs: int = None
) -> pd.DataFrame:
    """
    Load raw text files, align with metadata, and apply clean_legal_text.

    Args:
        metadata_df: Metadata dataframe.
        text_dir: Path to directory of extracted text files.
        max_docs: Optional document limit for rapid testing.

    Returns:
        pd.DataFrame: Corpus dataframe with raw_text and clean_text.
    """
    # TODO: Iterate through metadata_df rows, read text from (text_dir / f"{row['path']}.txt")
    # Apply clean_legal_text to raw text
    # Return DataFrame with metadata attributes + 'raw_text' + 'clean_text'
    raise NotImplementedError("Implement build_and_clean_corpus")


def compute_cleaning_metrics(corpus_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comparative word and character reduction metrics.

    Args:
        corpus_df: Dataframe with 'raw_text' and 'clean_text'.

    Returns:
        Dict[str, float]: Dictionary with avg_raw_words, avg_clean_words, word_reduction_pct, etc.
    """
    # TODO: Calculate raw and clean word/character counts and compute reduction statistics
    raise NotImplementedError("Implement compute_cleaning_metrics")


if __name__ == "__main__":
    print_header("Section 05: Text Cleaning — Exercise Validation")
    try:
        sample_raw = "This is an in-\nlaw reference.\r\n\r\nSection   302   IPC.\nOrder of the Court."
        cleaned = clean_legal_text(sample_raw)
        print(f" [TODO] Cleaned sample output:\n{repr(cleaned)}")

        assert "in-law" in cleaned, "Hyphenated split word should be rejoined"
        assert "  " not in cleaned, "Multiple spaces should be collapsed"
        assert "\n\n" in cleaned, "Paragraph breaks should be preserved"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
