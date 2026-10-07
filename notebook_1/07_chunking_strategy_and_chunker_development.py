"""
Section 7: Chunk Generation and Structural Optimization
(Part 1: Chunking Strategy, Front Matter Removal, and Chunker Development)

Supreme Court judgments vary substantially in length and may contain several distinct legal issues, factual discussions, statutory references, and judicial conclusions. Indexing complete judgments as single retrieval units would therefore be inefficient and could reduce retrieval precision, particularly for transformer-based embedding models with limited input lengths.

This section evaluates multiple chunking configurations to identify a structurally suitable strategy for the NYAYaRAG corpus. The analysis focuses on chunk size and overlap, both of which influence the balance between semantic coherence, contextual continuity, corpus expansion, and retrieval efficiency.

Six candidate configurations are examined using target chunk sizes of 384, 512, and 640 words, each with approximately 10% and 20% overlap. The configurations are assessed using structural measures such as total chunk count, chunk-length distribution, chunks per judgment, paragraph coverage, tiny and empty chunks, duplication, and qualitative preservation of legal reasoning.

The configuration selected through this structural assessment will be fixed for all subsequent retrieval models, ensuring that TF-IDF, BM25, E5, BGE, InLegalBERT, and hybrid retrieval are compared using an identical chunk corpus.
"""

from pathlib import Path
import re
import pandas as pd

MARKER = "Judgment / Order of the Supreme Court"


def get_candidate_chunk_configs():
    """
    7.1 Candidate Chunking Configurations

    To identify a structurally appropriate chunking strategy for the Supreme Court judgment corpus, three candidate chunk sizes of 256, 384, and 512 tokens were selected. These chunk sizes represent progressively increasing levels of contextual coverage, allowing the study to evaluate the balance between fine-grained segmentation and preservation of multi-paragraph legal reasoning. The selected range was guided by the structural characteristics of the judgment corpus identified during exploratory data analysis.

    Based on the exploratory analysis of the corpus, the selected chunk sizes approximately preserve between two and four paragraphs of legal reasoning, thereby allowing the study to examine the trade-off between fine-grained retrieval and contextual completeness. A fixed overlap of approximately 10% of the target chunk size was maintained across all configurations to preserve continuity between adjacent chunks while limiting redundant content.
    """
    chunk_configs = {
        "chunk_256_26": {
            "chunk_size": 256,
            "chunk_overlap": 26,
            "overlap_percent": "10%"
        },
        "chunk_256_51": {
            "chunk_size": 256,
            "chunk_overlap": 51,
            "overlap_percent": "20%"
        },
        "chunk_384_40": {
            "chunk_size": 384,
            "chunk_overlap": 40,
            "overlap_percent": "10%"
        },
        "chunk_384_80": {
            "chunk_size": 384,
            "chunk_overlap": 80,
            "overlap_percent": "20%"
        },
        "chunk_512_50": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "overlap_percent": "10%"
        },
        "chunk_512_100": {
            "chunk_size": 512,
            "chunk_overlap": 100,
            "overlap_percent": "20%"
        }
    }

    config_df = (
        pd.DataFrame.from_dict(chunk_configs, orient="index")
        .reset_index()
    )

    config_df.columns = [
        "Configuration",
        "Target Chunk Size (Words)",
        "Overlap (Words)",
        "Overlap (%)"
    ]

    print("=" * 70)
    print("Candidate Chunking Configurations")
    print("=" * 70)
    print(config_df)
    return chunk_configs


def load_and_validate_processed_corpus(processed_corpus_file):
    """
    7.2 Load the Processed Corpus

    The structurally optimized chunking experiments require the cleaned and validated judgment corpus produced in the preceding sections. The processed corpus is therefore loaded from the saved Parquet file to avoid repeating PDF extraction, cleaning, and exploratory analysis.

    Before chunk generation, the corpus is validated to confirm that all 782 judgments are present, the `clean_text` field is available, and no judgment contains missing or empty cleaned text. This ensures that every candidate chunking configuration is applied to the same complete source corpus.
    """
    corpus_df = pd.read_parquet(processed_corpus_file)

    missing_clean_text = corpus_df["clean_text"].isna().sum()
    empty_clean_text = (
        corpus_df["clean_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print("=" * 70)
    print("Processed Corpus Validation")
    print("=" * 70)

    print(f"Corpus file exists       : {processed_corpus_file.exists()}")
    print(f"Total judgments loaded   : {len(corpus_df):,}")
    print(f"Total columns            : {corpus_df.shape[1]}")
    print(f"Missing clean_text       : {missing_clean_text:,}")
    print(f"Empty clean_text         : {empty_clean_text:,}")
    print(f"Ready for chunking       : {missing_clean_text == 0 and empty_clean_text == 0}")

    return corpus_df


def count_words(text):
    """
    7.4.1 Measure Paragraph Length

    The Universal Paragraph-Aware Chunker measures the size of each paragraph using word counts rather than model-specific tokenizers. This approach ensures that the chunking process remains independent of any particular embedding model while providing a consistent and reproducible measure of chunk size across the entire corpus.

    Word counts are used solely to control the approximate size of each chunk and do not influence the semantic representation of the text during the retrieval stage.
    """
    return len(re.findall(r"\b\w+\b", str(text)))


def split_into_paragraphs(text):
    """
    7.4.2 Split a Judgment into Paragraphs

    Each judgment is first divided into individual paragraphs using blank lines as the primary separator. Empty or whitespace-only paragraphs are removed to ensure that only meaningful textual content is processed during chunk generation.

    Paragraphs represent the natural structural units of judicial writing and therefore serve as the fundamental building blocks of the Universal Paragraph-Aware Chunker.
    """
    paragraphs = re.split(r"\n\s*\n", str(text))
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def remove_editorial_front_matter(text):
    """
    7.5 Removal of Editorial Front Matter

    The preliminary chunking experiment revealed that the opening sections of several judgments contained editorial material, including case summaries, issue statements, headnotes, and publisher annotations. These sections produced unusually long paragraphs, resulting in chunk sizes substantially larger than the configured targets.

    Since the objective of NYAYaRAG is to retrieve and generate answers grounded in the judicial reasoning of Supreme Court judgments, editorial front matter was excluded prior to chunk generation. The preprocessing retained only the judicial text beginning from the actual judgment section, thereby improving the structural consistency of the corpus while preserving the authoritative legal reasoning.

    7.5.1 Removal of Editorial Front Matter

    Based on the structural inspection of multiple Supreme Court judgments, the opening sections of each document were found to contain publisher-generated editorial material, including citations, issue summaries, headnotes, appearance details, and page metadata. These sections are not part of the judicial reasoning and may adversely affect chunk generation and retrieval quality.

    To ensure that NYAYaRAG indexes only the authoritative judicial text, the editorial front matter is removed by retaining the document content beginning from the marker "Judgment / Order of the Supreme Court". This preprocessing step is applied uniformly across the corpus to maintain consistency and reproducibility.
    """
    text = str(text)
    position = text.find(MARKER)

    if position == -1:
        return text

    return text[position:].strip()


def paragraph_aware_chunker(text, target_size, overlap_size):
    """
    7.6.2 Implementation of the Refined Universal Paragraph-Aware Chunker

    Based on the structural evaluation, the Universal Paragraph-Aware Chunker is refined to improve adherence to the configured chunk sizes while preserving the semantic integrity of judicial reasoning.

    The initial implementation appended every complete paragraph until the target chunk size was reached or exceeded. This occasionally resulted in substantially oversized chunks when the next paragraph itself was unusually long.

    To address this limitation, the refined algorithm introduces a tolerance threshold equal to 20% of the configured target chunk size. Before appending the next paragraph, the algorithm estimates the resulting chunk size. If adding the paragraph would cause the chunk size to exceed the target by more than the specified tolerance, the current chunk is finalised and the next paragraph begins a new chunk. Individual paragraphs are still divided only when their own length exceeds the configured target size.

    This refinement provides a better balance between preserving complete legal paragraphs and maintaining consistent chunk sizes, thereby improving the suitability of the generated chunks for embedding generation and retrieval experiments.
    """
    if overlap_size >= target_size:
        raise ValueError("Overlap must be smaller than target size.")

    # Maximum permissible chunk size (20% tolerance)
    tolerance = 0.20
    maximum_chunk_size = int(target_size * (1 + tolerance))

    # Split document into paragraphs
    original_paragraphs = split_into_paragraphs(text)

    # Precompute paragraph units
    paragraph_units = []

    for paragraph in original_paragraphs:

        words = paragraph.split()

        # Paragraph fits within target size
        if len(words) <= target_size:

            paragraph_units.append({
                "text": paragraph,
                "word_count": len(words),
                "was_split": False
            })

        # Split only oversized paragraphs
        else:

            start = 0

            while start < len(words):

                segment_words = words[start:start + target_size]

                paragraph_units.append({
                    "text": " ".join(segment_words),
                    "word_count": len(segment_words),
                    "was_split": True
                })

                start += target_size

    # Generate chunks
    chunks = []
    start_index = 0

    while start_index < len(paragraph_units):

        current_units = []
        current_words = 0
        end_index = start_index

        # Build current chunk
        while end_index < len(paragraph_units):

            unit = paragraph_units[end_index]
            projected_size = current_words + unit["word_count"]

            # Always include first paragraph
            if len(current_units) == 0:
                current_units.append(unit)
                current_words += unit["word_count"]
                end_index += 1
                continue

            # Stop if tolerance exceeded
            if projected_size > maximum_chunk_size:
                break

            current_units.append(unit)
            current_words += unit["word_count"]
            end_index += 1

            # Target reached
            if current_words >= target_size:
                break

        chunk_text = "\n\n".join(
            unit["text"] for unit in current_units
        )

        chunks.append({
            "chunk_text": chunk_text.strip(),
            "word_count": current_words,
            "paragraph_count": len(current_units),
            "split_paragraph_count": sum(
                unit["was_split"] for unit in current_units
            ),
            "start_unit": start_index,
            "end_unit": end_index - 1
        })

        # Stop after last chunk
        if end_index >= len(paragraph_units):
            break

        # Compute overlap
        overlap_words = 0
        overlap_start = end_index

        while overlap_start > start_index:
            previous_unit = paragraph_units[overlap_start - 1]
            overlap_words += previous_unit["word_count"]
            overlap_start -= 1

            if overlap_words >= overlap_size:
                break

        # Guarantee forward progress
        start_index = max(overlap_start, start_index + 1)

    return chunks


def validate_chunking_on_sample(sample_text, chunk_configs):
    """
    Validate Chunking on Sample Text across Candidate Configurations
    """
    validation_results = []
    sample_chunk_sets = {}

    for config_name, config in chunk_configs.items():

        chunks = paragraph_aware_chunker(
            text=sample_text,
            target_size=config["chunk_size"],
            overlap_size=config["chunk_overlap"]
        )

        sample_chunk_sets[config_name] = chunks

        word_counts = [chunk["word_count"] for chunk in chunks]
        paragraph_counts = [chunk["paragraph_count"] for chunk in chunks]
        split_counts = [chunk["split_paragraph_count"] for chunk in chunks]

        validation_results.append({
            "Configuration": config_name,
            "Target Size": config["chunk_size"],
            "Overlap": config["chunk_overlap"],
            "Total Chunks": len(chunks),
            "Average Words": round(sum(word_counts) / len(word_counts), 2),
            "Median Words": round(pd.Series(word_counts).median(), 2),
            "Minimum Words": min(word_counts),
            "Maximum Words": max(word_counts),
            "Average Paragraphs": round(sum(paragraph_counts) / len(paragraph_counts), 2),
            "Split Paragraphs": sum(split_counts)
        })

    validation_df = pd.DataFrame(validation_results)

    print("=" * 70)
    print("Chunk Validation on Sample Judgment")
    print("=" * 70)
    print(validation_df)
    return validation_df, sample_chunk_sets


def apply_editorial_front_matter_removal(corpus_df):
    """
    7.5.2 Validation of Editorial Front Matter Removal

    The cleaned corpus was validated to ensure that editorial front matter had been successfully removed without introducing missing or empty documents. The validation also confirms that the newly created judgment_text field is ready for chunk generation.
    """
    corpus_df["judgment_text"] = corpus_df["clean_text"].apply(
        remove_editorial_front_matter
    )

    print("=" * 70)
    print("Editorial Front Matter Removal Completed")
    print("=" * 70)

    print(f"Total Judgments : {len(corpus_df)}")
    print(f"New Column      : judgment_text")

    print("\nValidation:")
    print(f"Missing judgment_text : {corpus_df['judgment_text'].isna().sum()}")
    print(f"Empty judgment_text   : {(corpus_df['judgment_text'].str.strip() == '').sum()}")

    original = len(corpus_df.loc[0, "clean_text"])
    cleaned = len(corpus_df.loc[0, "judgment_text"])

    print(f"Sample 0 Original characters : {original:,}")
    print(f"Sample 0 Cleaned characters  : {cleaned:,}")
    print(f"Sample 0 Characters removed  : {original - cleaned:,}")

    return corpus_df


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    processed_corpus_file = PROCESSED_DIR / "metadata_2024_processed.parquet"

    chunk_configs = get_candidate_chunk_configs()
    corpus_df = load_and_validate_processed_corpus(processed_corpus_file)

    sample_text_raw = corpus_df.loc[0, "clean_text"]
    paragraphs = split_into_paragraphs(sample_text_raw)
    print(f"Paragraphs in sample judgment: {len(paragraphs)}")

    corpus_df = apply_editorial_front_matter_removal(corpus_df)
    sample_text_judicial = corpus_df.loc[0, "judgment_text"]

    validation_df, sample_chunk_sets = validate_chunking_on_sample(sample_text_judicial, chunk_configs)
