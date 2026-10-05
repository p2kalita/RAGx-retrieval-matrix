"""
Section 04: PDF Text Extraction with PyMuPDF — EXERCISE
======================================================

Overview:
---------
Raw judicial opinions in India are published as PDF files. To construct a searchable
RAG knowledge base, we must extract text accurately while maintaining layout order.
We utilize PyMuPDF (fitz) with positional sorting enabled (`sort=True`).

Tasks:
------
1. Implement `extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]` to extract all
   pages using `page.get_text("text", sort=True)`, join them with `\\n\\n`, and report
   page count, character count, and word count.
2. Implement `batch_extract_corpus(metadata_df: pd.DataFrame, pdf_dir: Path, output_dir: Path, skip_existing: bool) -> pd.DataFrame`
   to process all documents, save the `.txt` files, and produce an `extraction_log.csv`.
3. Implement `validate_extraction_results(log_df: pd.DataFrame) -> Dict[str, Any]`
   verifying that zero documents failed extraction and all outputs have text.

Instructions:
-------------
Fill in the code under each `TODO` block. Run this script directly to test your solution.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import pymupdf

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    PDF_DIR,
    TEXT_DIR,
    EXTRACTION_LOG_FILE,
    print_header,
    load_raw_metadata,
)


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract structured text from a single judgment PDF using PyMuPDF.

    Args:
        pdf_path: Path to the judgment PDF file.

    Returns:
        Dict[str, Any]:
            - 'text': str (joined page texts with \\n\\n)
            - 'num_pages': int
            - 'char_count': int
            - 'word_count': int
            - 'status': 'SUCCESS' or 'FAILED'
            - 'error': str (empty if success)
    """
    # TODO: Open the PDF using pymupdf.open(pdf_path)
    # Iterate through pages, call page.get_text("text", sort=True)
    # Join page texts with "\n\n"
    # Calculate char_count (len(text)) and word_count (len(text.split()))
    # Catch any exceptions and return status='FAILED' with error string
    raise NotImplementedError("Implement extract_text_from_pdf")


def batch_extract_corpus(
    metadata_df: pd.DataFrame,
    pdf_dir: Path = PDF_DIR,
    output_dir: Path = TEXT_DIR,
    skip_existing: bool = True,
    max_docs: int = None,
) -> pd.DataFrame:
    """
    Extract text across the corpus and record results in an extraction log.

    Args:
        metadata_df: Metadata dataframe.
        pdf_dir: Directory containing input PDFs.
        output_dir: Directory to save extracted .txt files.
        skip_existing: If True, skips files that already exist on disk.
        max_docs: Optional limit for testing.

    Returns:
        pd.DataFrame: Extraction log with columns ['judgment_id', 'path', 'num_pages', 'char_count', 'word_count', 'status', 'error']
    """
    # TODO: Iterate over rows in metadata_df (up to max_docs)
    # Resolve input pdf: pdf_dir / f"{row['path']}_EN.pdf"
    # Resolve output txt: output_dir / f"{row['path']}.txt"
    # If skip_existing and output txt exists, load info from existing file
    # Otherwise call extract_text_from_pdf and save text to output file
    # Accumulate log records and return as DataFrame
    raise NotImplementedError("Implement batch_extract_corpus")


def validate_extraction_results(log_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that all extractions succeeded with non-empty content.

    Args:
        log_df: Extraction log DataFrame.

    Returns:
        Dict[str, Any]: Summary metrics ('total', 'success', 'failed', 'empty_text').
    """
    # TODO: Verify log_df['status'] == 'SUCCESS' and log_df['word_count'] > 0
    raise NotImplementedError("Implement validate_extraction_results")


if __name__ == "__main__":
    print_header("Section 04: PDF Text Extraction — Exercise Validation")
    try:
        raw_df = load_raw_metadata()
        sample_path = PDF_DIR / f"{raw_df.iloc[0]['path']}_EN.pdf"

        sample_res = extract_text_from_pdf(sample_path)
        print(f" [TODO] Extracted sample PDF: {sample_res['num_pages']} pages, {sample_res['word_count']} words.")
        assert sample_res["status"] == "SUCCESS", "Sample extraction should succeed"
        assert sample_res["word_count"] > 100, "Sample should contain substantial text"

        print("\n ALL EXERCISE TESTS PASSED!")
    except NotImplementedError as e:
        print(f"\n Exercise incomplete: {e}")
        print("Please complete the TODOs above and rerun.")
