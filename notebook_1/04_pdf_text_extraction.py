"""
Section 4: PDF Text Extraction

The Supreme Court judgments are available as PDF documents, which must be converted into machine-readable text before they can be processed by the retrieval pipeline. PDF text extraction is therefore a critical step in constructing the NYAYaRAG corpus, as the extracted text forms the basis for subsequent cleaning, chunking, embedding generation, and retrieval.

This section extracts textual content from each judgment while preserving the original document structure as closely as possible. The extracted text is stored separately from the original PDF files to facilitate reproducibility and avoid repeated extraction during subsequent experiments.

For each judgment, the notebook records the extraction status and stores the extracted text as an individual text file. Any extraction failures or empty outputs are also recorded to enable quality assurance before proceeding to corpus preparation.
"""

from pathlib import Path
import pandas as pd
import pymupdf
from tqdm import tqdm


def init_text_directory(processed_dir):
    """
    Define and create Extracted Text Directory
    """
    text_dir = processed_dir / "text" / "year=2024" / "english"
    text_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracted text directory: {text_dir}")
    print(f"Directory exists        : {text_dir.exists()}")
    return text_dir


def test_extraction_on_sample(sample_pdf):
    """
    Observation

    The sample PDF was extracted successfully using PyMuPDF. The 18-page judgment produced 46,657 characters and 7,249 words of readable text.

    A preliminary inspection of the extracted content confirms that legally significant elements, including the case citation, party names, bench composition, issues for consideration, statutory references, headnotes, and judgment text, have been preserved. The text is therefore suitable for corpus construction and subsequent cleaning.

    Since the sample extraction produced a complete and readable output without errors, the same extraction procedure can now be applied to the full corpus of 782 judgments.
    """
    try:
        with pymupdf.open(sample_pdf) as document:
            sample_pages = []

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text", sort=True)
                sample_pages.append(page_text)

            sample_text = "\n\n".join(sample_pages)

        print("=" * 60)
        print("Sample PDF Extraction")
        print("=" * 60)

        print(f"PDF file          : {sample_pdf.name}")
        print(f"Number of pages   : {len(sample_pages)}")
        print(f"Characters found  : {len(sample_text):,}")
        print(f"Words found       : {len(sample_text.split()):,}")

        print("\nFirst 2,000 characters:\n")
        print(sample_text[:2000])
        return sample_text

    except Exception as error:
        print(f"Extraction failed: {error}")
        return None


def extract_all_judgments(pdf_files, text_dir):
    """
    Observation

    Text extraction was completed successfully for all 782 Supreme Court judgment PDFs. No extraction failures were recorded, resulting in a 100% extraction success rate.

    The extraction log confirms that each PDF was converted into a corresponding UTF-8 text file, with page count, character count, word count, extraction status, and error information recorded for every judgment. The sample outputs indicate substantial text content across the corpus, confirming that the judgments are available in machine-readable form.

    The complete extraction of all documents provides a reliable textual corpus for the next stages of quality validation, text cleaning, exploratory analysis, and chunk generation.
    """
    extraction_log = []

    for pdf_file in tqdm(pdf_files, desc="Extracting Judgments"):

        txt_filename = pdf_file.stem.replace("_EN", "") + ".txt"
        txt_path = text_dir / txt_filename

        try:
            # Skip extraction if text file already exists
            if txt_path.exists():

                extraction_log.append({
                    "pdf_file": pdf_file.name,
                    "text_file": txt_filename,
                    "pages": None,
                    "characters": None,
                    "words": None,
                    "status": "Already Exists",
                    "error": None
                })

                continue

            # Open PDF
            document = pymupdf.open(pdf_file)

            pages = []

            for page in document:
                pages.append(page.get_text("text", sort=True))

            full_text = "\n\n".join(pages)

            # Save extracted text
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            extraction_log.append({
                "pdf_file": pdf_file.name,
                "text_file": txt_filename,
                "pages": len(document),
                "characters": len(full_text),
                "words": len(full_text.split()),
                "status": "Success",
                "error": None
            })

            document.close()

        except Exception as e:

            extraction_log.append({
                "pdf_file": pdf_file.name,
                "text_file": txt_filename,
                "pages": None,
                "characters": None,
                "words": None,
                "status": "Failed",
                "error": str(e)
            })

    # Convert log into DataFrame
    extraction_df = pd.DataFrame(extraction_log)

    print("\nExtraction completed.")

    print("=" * 70)
    print("PDF Text Extraction Summary")
    print("=" * 70)

    print(f"Total PDFs              : {len(extraction_df)}")
    print(f"Successful extractions  : {(extraction_df['status']=='Success').sum()}")
    print(f"Already existing files  : {(extraction_df['status']=='Already Exists').sum()}")
    print(f"Failed extractions      : {(extraction_df['status']=='Failed').sum()}")

    print(extraction_df.head())
    return extraction_df


def format_and_save_extraction_log(extraction_df, pdf_dir, text_dir, processed_dir):
    """
    4.1 Extraction Report

    An extraction report was generated to document the outcome of the PDF text extraction process for each judgment in the corpus. The report records the source PDF file, the corresponding extracted text file, file locations, page count, character count, word count, extraction status, and any extraction errors.

    The report confirms that every judgment was processed successfully and provides a reproducible audit trail for the corpus construction process. Recording these details facilitates quality assurance, enables verification of individual documents, and supports subsequent stages of text cleaning, chunking, and retrieval index construction.
    """
    # Rename columns
    extraction_df = extraction_df.rename(
        columns={
            "status": "extraction_status",
            "error": "extraction_error"
        }
    )

    # Add full paths
    extraction_df["pdf_path"] = extraction_df["pdf_file"].apply(
        lambda x: str(pdf_dir / x)
    )

    extraction_df["text_path"] = extraction_df["text_file"].apply(
        lambda x: str(text_dir / x)
    )

    # Arrange columns
    extraction_df = extraction_df[
        [
            "pdf_file",
            "pdf_path",
            "text_file",
            "text_path",
            "pages",
            "characters",
            "words",
            "extraction_status",
            "extraction_error"
        ]
    ]

    extraction_log_file = processed_dir / "extraction_log.csv"
    extraction_df.to_csv(extraction_log_file, index=False)

    print(f"Extraction log saved to: {extraction_log_file}")
    print(f"File exists            : {extraction_log_file.exists()}")
    return extraction_df


def validate_extracted_text_corpus(text_dir, pdf_files):
    """
    4.2 Extraction Validation

    Before cleaning the extracted judgment text, the corpus is validated to confirm that every PDF produced a corresponding, non-empty text file. The validation also identifies unusually short documents that may indicate incomplete extraction.

    A provisional threshold of 100 words is used only as a screening criterion. Files below this threshold are not automatically removed; they are flagged for manual inspection because some valid judicial documents may naturally be brief.

    Observation:
    The extraction validation confirms that all 782 Supreme Court judgments were successfully converted into machine-readable text files. The number of extracted text files exactly matches the number of source PDF documents, indicating that the extraction process achieved complete corpus coverage.

    No empty text files, read errors, or unusually short documents (fewer than 100 words) were identified. The extracted corpus contains an average of 8,725 words per judgment, with document lengths ranging from 320 to 144,445 words. This wide variation is expected because Supreme Court judgments differ considerably in complexity and length.

    The validation results indicate that the extracted text corpus is complete, readable, and suitable for the subsequent stages of text cleaning, corpus analysis, and chunk generation.
    """
    # Retrieve all extracted text files
    text_files = sorted(text_dir.glob("*.txt"))

    validation_records = []

    for text_file in text_files:
        try:
            text = text_file.read_text(encoding="utf-8")

            word_count = len(text.split())
            character_count = len(text)

            validation_records.append({
                "text_file": text_file.name,
                "text_path": str(text_file),
                "characters": character_count,
                "words": word_count,
                "is_empty": character_count == 0,
                "is_unusually_short": word_count < 100,
                "validation_status": "Valid" if character_count > 0 else "Empty"
            })

        except Exception as error:
            validation_records.append({
                "text_file": text_file.name,
                "text_path": str(text_file),
                "characters": None,
                "words": None,
                "is_empty": None,
                "is_unusually_short": None,
                "validation_status": "Read Error",
                "validation_error": str(error)
            })

    # Convert validation results into a DataFrame
    validation_df = pd.DataFrame(validation_records)

    print("=" * 70)
    print("Extracted Text Validation Summary")
    print("=" * 70)

    print(f"Expected text files       : {len(pdf_files):,}")
    print(f"Text files found          : {len(text_files):,}")
    print(f"Empty text files          : {validation_df['is_empty'].fillna(False).sum():,}")
    print(f"Files below 100 words     : {validation_df['is_unusually_short'].fillna(False).sum():,}")
    print(f"Files with read errors    : {(validation_df['validation_status'] == 'Read Error').sum():,}")

    print("\nWord Count Statistics:")
    print(
        validation_df["words"]
        .describe()
        .round(2)
        .to_frame(name="Value")
    )

    print("\nFive shortest extracted judgments:")
    print(
        validation_df
        .sort_values("words")
        .head(5)[
            [
                "text_file",
                "characters",
                "words",
                "is_empty",
                "is_unusually_short",
                "validation_status"
            ]
        ]
    )
    return validation_df


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
    PDF_DIR = RAW_DATA_DIR / "pdf"
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

    pdf_dir = PDF_DIR / "year=2024" / "english"
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    text_dir = init_text_directory(PROCESSED_DIR)
    if pdf_files:
        test_extraction_on_sample(pdf_files[0])
        extraction_df = extract_all_judgments(pdf_files, text_dir)
        extraction_df = format_and_save_extraction_log(extraction_df, pdf_dir, text_dir, PROCESSED_DIR)
        validation_df = validate_extracted_text_corpus(text_dir, pdf_files)
