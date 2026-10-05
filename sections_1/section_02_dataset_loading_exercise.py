"""
Section 02: Dataset Acquisition and Corpus Loading — EXERCISE
============================================================

Overview:
---------
This section verifies the presence and integrity of the Supreme Court of India 2024
judgment corpus before any processing. You must load the metadata table, list the
raw PDF files, and verify 1-to-1 correspondence between metadata entries and PDF files.

Tasks:
------
1. Implement `load_raw_metadata(metadata_path: Path) -> pd.DataFrame` to read the
   parquet metadata file.
2. Implement `list_judgment_pdfs(pdf_dir: Path) -> List[Path]` to scan and return
   all `.pdf` judgment files.
3. Implement `validate_corpus_correspondence(metadata_df: pd.DataFrame, pdf_dir: Path) -> Dict[str, Any]`
   which checks whether each row's `path` has a matching `{path}_EN.pdf` file on disk.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import PDF_DIR, METADATA_FILE, print_header


def load_raw_metadata(metadata_path: Path = METADATA_FILE) -> pd.DataFrame:
    """
    Load the Supreme Court metadata Parquet file.

    Args:
        metadata_path: Path to metadata_2024.parquet.

    Returns:
        pd.DataFrame: Loaded metadata dataframe.
    """
    # TODO: Validate that metadata_path exists and read it with pd.read_parquet
    raise NotImplementedError("Implement load_raw_metadata")


def list_judgment_pdfs(pdf_dir: Path = PDF_DIR) -> List[Path]:
    """
    List all PDF files located in the judgment PDF directory.

    Args:
        pdf_dir: Path to directory containing judgment PDFs.

    Returns:
        List[Path]: Sorted list of PDF file paths.
    """
    # TODO: Scan pdf_dir for all files matching '*.pdf' and return sorted list
    raise NotImplementedError("Implement list_judgment_pdfs")


def validate_corpus_correspondence(
    metadata_df: pd.DataFrame, pdf_dir: Path = PDF_DIR
) -> Dict[str, Any]:
    """
    Cross-reference metadata records with PDF files on disk.

    Args:
        metadata_df: Metadata dataframe containing a 'path' column.
        pdf_dir: Path to PDF files directory.

    Returns:
        Dict[str, Any]: Summary metrics:
            - 'total_metadata_records': int
            - 'total_pdf_files': int
            - 'matching_count': int
            - 'missing_pdfs': List[str]
            - 'is_valid': bool (True if all metadata paths match existing PDFs)
    """
    # TODO: Verify that for each row['path'], (pdf_dir / f"{row['path']}_EN.pdf").exists()
    # Compute total matches, missing files, and whether the corpus is valid.
    raise NotImplementedError("Implement validate_corpus_correspondence")


if __name__ == "__main__":
    print_header("Section 02: Dataset Loading — Exercise Validation")
    try:
        df = load_raw_metadata(METADATA_FILE)
        print(f" [TODO] Loaded metadata with {len(df)} rows.")

        pdfs = list_judgment_pdfs(PDF_DIR)
        print(f" [TODO] Found {len(pdfs)} PDF files.")

        val = validate_corpus_correspondence(df, PDF_DIR)
        print(f" [TODO] Correspondence validation: {val}")

        assert val["total_metadata_records"] == 782, "Expected 782 metadata records"
        assert val["total_pdf_files"] == 782, "Expected 782 PDF files"
        assert val["is_valid"] is True, "Expected corpus correspondence to be 100% valid"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
