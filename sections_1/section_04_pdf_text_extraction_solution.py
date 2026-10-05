"""
Section 04: PDF Text Extraction with PyMuPDF — SOLUTION
======================================================

This module extracts text from all 782 Supreme Court judgment PDFs using PyMuPDF
(fitz), saves the extracted text files to the processed text directory, and
creates an extraction audit log (Notebook Cells 30 to 43).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    """
    if not pdf_path.exists():
        return {
            "text": "",
            "num_pages": 0,
            "char_count": 0,
            "word_count": 0,
            "status": "FAILED",
            "error": f"File does not exist: {pdf_path}",
        }

    try:
        with pymupdf.open(pdf_path) as doc:
            pages = []
            for page in doc:
                text = page.get_text("text", sort=True)
                pages.append(text)
            full_text = "\n\n".join(pages)

            return {
                "text": full_text,
                "num_pages": len(doc),
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
                "status": "SUCCESS",
                "error": "",
            }
    except Exception as e:
        return {
            "text": "",
            "num_pages": 0,
            "char_count": 0,
            "word_count": 0,
            "status": "FAILED",
            "error": str(e),
        }


def batch_extract_corpus(
    metadata_df: pd.DataFrame,
    pdf_dir: Path = PDF_DIR,
    output_dir: Path = TEXT_DIR,
    skip_existing: bool = True,
    max_docs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract text across the corpus and record results in an extraction log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    target_df = metadata_df if max_docs is None else metadata_df.head(max_docs)
    total = len(target_df)

    for i, (_, row) in enumerate(target_df.iterrows(), start=1):
        case_id = row.get("case_id", f"case_{i}")
        doc_path = row["path"]
        pdf_file = pdf_dir / f"{doc_path}_EN.pdf"
        txt_file = output_dir / f"{doc_path}.txt"

        if skip_existing and txt_file.exists() and txt_file.stat().st_size > 0:
            try:
                content = txt_file.read_text(encoding="utf-8")
                records.append({
                    "judgment_id": doc_path,
                    "path": doc_path,
                    "case_id": case_id,
                    "num_pages": 0,  # cached from disk
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    "status": "SUCCESS",
                    "error": "",
                })
                continue
            except Exception:
                pass  # Fall through to re-extract

        # Perform extraction
        res = extract_text_from_pdf(pdf_file)
        if res["status"] == "SUCCESS":
            txt_file.write_text(res["text"], encoding="utf-8")

        records.append({
            "judgment_id": doc_path,
            "path": doc_path,
            "case_id": case_id,
            "num_pages": res["num_pages"],
            "char_count": res["char_count"],
            "word_count": res["word_count"],
            "status": res["status"],
            "error": res["error"],
        })

        if i % 100 == 0 or i == total:
            print(f"Processed [{i}/{total}] judgments...")

    log_df = pd.DataFrame(records)
    log_df.to_csv(EXTRACTION_LOG_FILE, index=False)
    return log_df


def validate_extraction_results(log_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that all extractions succeeded with non-empty text.
    """
    total = len(log_df)
    success = (log_df["status"] == "SUCCESS").sum()
    failed = (log_df["status"] == "FAILED").sum()
    empty = (log_df["char_count"] == 0).sum()

    return {
        "total_documents": int(total),
        "successful_extractions": int(success),
        "failed_extractions": int(failed),
        "empty_extractions": int(empty),
        "is_valid": bool(success == total and failed == 0 and empty == 0),
    }


def run_section_04(force: bool = False) -> Dict[str, Any]:
    """Execute complete Section 04 pipeline."""
    print_header("Section 4: PDF Text Extraction via PyMuPDF")

    metadata_df = load_raw_metadata()
    print(f"Total judgments to extract : {len(metadata_df):,}")
    print(f"Target text directory      : {TEXT_DIR}")
    print(f"Extraction log path        : {EXTRACTION_LOG_FILE}")

    log_df = batch_extract_corpus(
        metadata_df=metadata_df,
        pdf_dir=PDF_DIR,
        output_dir=TEXT_DIR,
        skip_existing=not force,
    )

    val = validate_extraction_results(log_df)
    print("\nExtraction Audit Report:")
    print(f"  - Total Judgments Processed : {val['total_documents']:,}")
    print(f"  - Successful Extractions    : {val['successful_extractions']:,}")
    print(f"  - Failed Extractions        : {val['failed_extractions']}")
    print(f"  - Empty Extractions         : {val['empty_extractions']}")
    print(f"  - Total Words Extracted     : {log_df['word_count'].sum():,}")
    print(f"  - Average Words / Judgment  : {log_df['word_count'].mean():,.1f}")
    print(f"  - Audit Status              : {'PASSED (782/782)' if val['is_valid'] else 'FAILED'}")

    assert val["is_valid"], "Extraction failed validation: missing or failed document texts"
    return {"log_df": log_df, "validation": val}


if __name__ == "__main__":
    run_section_04()
