"""
Section 09: Editorial Front Matter Detection and Removal — EXERCISE
==================================================================

Overview:
---------
Published Indian court judgments often begin with publisher front matter: headnotes,
advocate rosters, case history, SCR volume markings, and editorial summaries.
Indexing this content pollutes the retrieval database with editorial summaries
rather than binding judicial holdings.

In this section, you will identify publisher boundary markers and slice the judgment
so that only the authentic judicial reasoning is retained.

Tasks:
------
1. Implement `remove_editorial_front_matter(text: str, marker: str) -> str`
   locating the marker and stripping preceding editorial headnotes.
2. Implement `apply_front_matter_removal(df: pd.DataFrame) -> pd.DataFrame`
   adding a `judgment_text` column and calculating front matter metrics.
3. Implement `validate_front_matter_removal(df: pd.DataFrame) -> Dict[str, Any]`
   verifying that no documents became empty and checking the average character reduction.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import print_header, load_processed_corpus

PRIMARY_MARKER = "Judgment / Order of the Supreme Court"


def remove_editorial_front_matter(text: str, marker: str = PRIMARY_MARKER) -> str:
    """
    Remove publisher-generated front matter and retain only the judicial reasoning.

    Args:
        text: Cleaned judgment text.
        marker: Primary header marking start of judgment reasoning.

    Returns:
        str: Sliced text starting from the marker (or original text if marker not found).
    """
    # TODO: Find marker in text; if position != -1, return text[position:].strip(), else return text
    raise NotImplementedError("Implement remove_editorial_front_matter")


def apply_front_matter_removal(df: pd.DataFrame, marker: str = PRIMARY_MARKER) -> pd.DataFrame:
    """
    Apply front matter removal across the entire corpus and compute difference columns.

    Args:
        df: Corpus dataframe with 'clean_text'.
        marker: Header marker string.

    Returns:
        pd.DataFrame: DataFrame with 'judgment_text' and character reduction columns.
    """
    # TODO: Apply remove_editorial_front_matter to df['clean_text'] -> df['judgment_text']
    # Calculate df['front_matter_chars_removed'] = df['clean_text'].str.len() - df['judgment_text'].str.len()
    raise NotImplementedError("Implement apply_front_matter_removal")


def validate_front_matter_removal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that no texts are empty and count how many judgments had front matter removed.

    Args:
        df: DataFrame with 'clean_text' and 'judgment_text'.

    Returns:
        Dict[str, Any]: Validation statistics.
    """
    # TODO: Check missing, empty count, number of docs with removed front matter, and avg reduction
    raise NotImplementedError("Implement validate_front_matter_removal")


if __name__ == "__main__":
    print_header("Section 09: Front Matter Removal — Exercise Validation")
    try:
        sample_doc = (
            "Publisher Law Reports: Headnote on Section 302 IPC. Counsel: ABC, XYZ.\n\n"
            "Judgment / Order of the Supreme Court\n"
            "Chandrachud, C.J. - This appeal arises out of..."
        )
        res = remove_editorial_front_matter(sample_doc)
        print(f" [TODO] Cleaned sample doc:\n{repr(res[:60])}")

        assert res.startswith("Judgment / Order of the Supreme Court"), "Text should start with marker"
        assert "Publisher Law Reports" not in res, "Headnote should be stripped"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
