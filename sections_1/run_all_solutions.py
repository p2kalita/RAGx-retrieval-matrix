"""
NYAYaRAG Master Solution Runner
===============================

Execute individual section solutions or run all 17 sections sequentially.

Usage:
------
    # Run all fast sections
    python sections/run_all_solutions.py

    # Run a specific section (e.g. section 12)
    python sections/run_all_solutions.py --section 12

    # Run all 17 sections including dense models
    python sections/run_all_solutions.py --all
"""

import sys
import argparse
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sections_1.section_01_environment_setup_solution import run_section_01
from sections_1.section_02_dataset_loading_solution import run_section_02
from sections_1.section_03_metadata_validation_solution import run_section_03
from sections_1.section_04_pdf_text_extraction_solution import run_section_04
from sections_1.section_05_text_cleaning_pipeline_solution import run_section_05
from sections_1.section_06_eda_length_and_outliers_solution import run_section_06
from sections_1.section_07_eda_structural_and_lexical_solution import run_section_07
from sections_1.section_08_paragraph_aware_chunking_solution import run_section_08
from sections_1.section_09_front_matter_removal_solution import run_section_09
from sections_1.section_10_refined_chunker_generation_solution import run_section_10
from sections_1.section_11_chunk_quality_validation_solution import run_section_11
from sections_1.section_12_tfidf_indexing_solution import run_section_12
from sections_1.section_13_bm25_indexing_solution import run_section_13
from sections_1.section_14_e5_dense_indexing_solution import run_section_14
from sections_1.section_15_bge_dense_indexing_solution import run_section_15
from sections_1.section_16_inlegalbert_indexing_solution import run_section_16
from sections_1.section_17_multi_model_evaluation_solution import run_section_17

SECTION_MAP = {
    1: ("Environment Setup & Project Config", run_section_01),
    2: ("Dataset Acquisition & Corpus Loading", run_section_02),
    3: ("Metadata Quality & Deduplication", run_section_03),
    4: ("PDF Text Extraction via PyMuPDF", run_section_04),
    5: ("Text Cleaning Pipeline", run_section_05),
    6: ("EDA — Length & Outliers", run_section_06),
    7: ("EDA — Structural & Statutory", run_section_07),
    8: ("Universal Paragraph Chunker", run_section_08),
    9: ("Front Matter Removal", run_section_09),
    10: ("Corpus-Wide Chunk Generation", run_section_10),
    11: ("Chunk Quality Validation", run_section_11),
    12: ("TF-IDF Lexical Indexing", run_section_12),
    13: ("BM25 Lexical Indexing", run_section_13),
    14: ("E5 Dense Semantic Retrieval", run_section_14),
    15: ("BGE Dense Semantic Retrieval", run_section_15),
    16: ("InLegalBERT Legal Retrieval", run_section_16),
    17: ("Multi-Model Final Validation", run_section_17),
}


def main():
    parser = argparse.ArgumentParser(description="NYAYaRAG Modular Pipeline Runner")
    parser.add_argument(
        "--section",
        type=int,
        choices=range(1, 18),
        help="Run a single specific section (1 to 17)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 17 sections from start to finish",
    )
    args = parser.parse_args()

    if args.section:
        title, func = SECTION_MAP[args.section]
        print(f"\n>>> Running Section {args.section:02d}: {title}...")
        func()
        print(f"\n Section {args.section:02d} completed successfully!")
    else:
        # Default: run selected sections or all if --all specified
        sections_to_run = range(1, 18) if args.all else [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 17]
        print(f"Executing {len(sections_to_run)} pipeline sections...")
        for sec in sections_to_run:
            title, func = SECTION_MAP[sec]
            print(f"\n{'='*70}\n[RUNNING SECTION {sec:02d}] {title}\n{'='*70}")
            func()
        print("\n ALL EXECUTED SECTIONS COMPLETED AND VALIDATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
