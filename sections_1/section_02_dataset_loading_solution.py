"""
Section 02: Dataset Acquisition and Corpus Loading — SOLUTION
============================================================

This module loads the raw metadata parquet file, discovers the 782 Supreme Court
judgment PDFs, and formally validates the 1-to-1 correspondence between metadata
records and raw PDF files on disk (Notebook Cells 7 to 15).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    PDF_DIR,
    METADATA_FILE,
    print_header,
)


def load_raw_metadata(metadata_path: Path = METADATA_FILE) -> pd.DataFrame:
    """
    Load the Supreme Court metadata Parquet file.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    df = pd.read_parquet(metadata_path)
    return df


def list_judgment_pdfs(pdf_dir: Path = PDF_DIR) -> List[Path]:
    """
    Discover all `.pdf` judgment files in the specified directory.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    return pdf_files


def validate_corpus_correspondence(
    metadata_df: pd.DataFrame, pdf_dir: Path = PDF_DIR
) -> Dict[str, Any]:
    """
    Validate 1-to-1 correspondence between metadata records and PDF files on disk.
    """
    total_metadata = len(metadata_df)
    pdf_files = list_judgment_pdfs(pdf_dir)
    total_pdfs = len(pdf_files)

    existing_pdf_stems = {p.stem for p in pdf_files}

    missing_pdfs = []
    matched = 0

    for _, row in metadata_df.iterrows():
        expected_stem = f"{row['path']}_EN"
        if expected_stem in existing_pdf_stems:
            matched += 1
        else:
            missing_pdfs.append(expected_stem)

    is_valid = (matched == total_metadata) and (total_pdfs == total_metadata)

    return {
        "total_metadata_records": total_metadata,
        "total_pdf_files": total_pdfs,
        "matching_count": matched,
        "missing_count": len(missing_pdfs),
        "missing_pdfs": missing_pdfs,
        "is_valid": is_valid,
    }


def run_section_02() -> Dict[str, Any]:
    """Execute complete Section 02 pipeline."""
    print_header("Section 2: Dataset Acquisition and Corpus Loading")

    print(f"Loading metadata from : {METADATA_FILE}")
    metadata_df = load_raw_metadata(METADATA_FILE)
    print(f"Metadata records loaded: {len(metadata_df):,}")
    print(f"Metadata columns       : {list(metadata_df.columns)}")

    print(f"\nScanning PDF directory : {PDF_DIR}")
    pdf_files = list_judgment_pdfs(PDF_DIR)
    print(f"Total PDF files found  : {len(pdf_files):,}")

    val = validate_corpus_correspondence(metadata_df, PDF_DIR)
    print("\nCorpus Integrity Validation:")
    print(f"  - Metadata records  : {val['total_metadata_records']:,}")
    print(f"  - PDF files on disk : {val['total_pdf_files']:,}")
    print(f"  - Valid matches     : {val['matching_count']:,}")
    print(f"  - Missing files     : {val['missing_count']}")
    print(f"  - Status            : {'PASSED (100% Match)' if val['is_valid'] else 'FAILED'}")

    assert val["is_valid"], "Corpus integrity validation failed: mismatch between metadata and PDFs"
    return {"metadata_df": metadata_df, "pdf_files": pdf_files, "validation": val}


if __name__ == "__main__":
    run_section_02()
