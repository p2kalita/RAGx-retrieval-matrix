"""
Section 7: Chunk Generation and Structural Optimization
(Part 2: Corpus-wide Chunk Generation and Structural Validation)

After validating and refining the Universal Paragraph-Aware Chunker using a representative Supreme Court judgment, the finalized algorithm is applied to the complete corpus of 782 Indian Supreme Court judgments. Chunk generation is performed separately for each candidate chunking configuration to evaluate their structural characteristics at the corpus level.

For every generated chunk, relevant metadata including the source judgment, chunk identifier, word count, paragraph count, and chunk text are retained. The resulting chunk collections form the input for subsequent embedding generation and retrieval experiments.

Corpus-wide statistics are then computed for each configuration, including the total number of generated chunks, average and median chunk size, paragraph distribution, and the frequency of paragraph splitting. These statistics provide an empirical basis for selecting the final chunking configuration for the NYAYaRAG framework.
"""

from pathlib import Path
import re
import pandas as pd


def split_into_paragraphs(text):
    """
    Split judgment into paragraphs and remove empty paragraphs.
    """
    paragraphs = re.split(r"\n\s*\n", str(text))
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def paragraph_aware_chunker(text, target_size, overlap_size):
    """
    Generate paragraph-aware chunks using word counts with 20% tolerance.
    """
    if overlap_size >= target_size:
        raise ValueError("Overlap must be smaller than target size.")

    tolerance = 0.20
    maximum_chunk_size = int(target_size * (1 + tolerance))

    original_paragraphs = split_into_paragraphs(text)
    paragraph_units = []

    for paragraph in original_paragraphs:
        words = paragraph.split()
        if len(words) <= target_size:
            paragraph_units.append({
                "text": paragraph,
                "word_count": len(words),
                "was_split": False
            })
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

    chunks = []
    start_index = 0

    while start_index < len(paragraph_units):
        current_units = []
        current_words = 0
        end_index = start_index

        while end_index < len(paragraph_units):
            unit = paragraph_units[end_index]
            projected_size = current_words + unit["word_count"]

            if len(current_units) == 0:
                current_units.append(unit)
                current_words += unit["word_count"]
                end_index += 1
                continue

            if projected_size > maximum_chunk_size:
                break

            current_units.append(unit)
            current_words += unit["word_count"]
            end_index += 1

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

        if end_index >= len(paragraph_units):
            break

        overlap_words = 0
        overlap_start = end_index

        while overlap_start > start_index:
            previous_unit = paragraph_units[overlap_start - 1]
            overlap_words += previous_unit["word_count"]
            overlap_start -= 1

            if overlap_words >= overlap_size:
                break

        start_index = max(overlap_start, start_index + 1)

    return chunks


def generate_corpus_chunks(corpus_df, chunk_configs):
    """
    7.7.1 Generate Chunks for the Complete Corpus

    The refined Universal Paragraph-Aware Chunker is applied to the complete corpus of 782 Indian Supreme Court judgments. Chunk generation is performed independently for each candidate chunking configuration.

    For every generated chunk, the corresponding judgment metadata, chunk identifier, chunk text, word count, paragraph count, and chunking configuration are recorded. The resulting datasets provide the basis for corpus-level structural analysis, embedding generation, and retrieval evaluation.

    Observation:
    The refined Universal Paragraph-Aware Chunker was successfully applied to all 782 Supreme Court judgments using each of the six candidate chunking configurations. As expected, the total number of generated chunks decreased with increasing target chunk size. The 256-word configurations produced the largest number of chunks (45,971–46,075), while the 512-word configurations generated the fewest chunks (27,233–27,868). Increasing the overlap from approximately 10% to 20% resulted in a modest increase in the total number of generated chunks for each target size, reflecting the additional contextual continuity introduced by overlapping content. These chunk collections will be analysed further to compare their structural characteristics and to identify the most suitable configuration for embedding generation and retrieval evaluation.
    """
    all_chunk_sets = {}

    for config_name, config in chunk_configs.items():

        print(f"\nGenerating chunks for {config_name}...")

        corpus_chunks = []
        chunk_id = 1

        for _, row in corpus_df.iterrows():

            chunks = paragraph_aware_chunker(
                text=row["judgment_text"],
                target_size=config["chunk_size"],
                overlap_size=config["chunk_overlap"]
            )

            for local_chunk_id, chunk in enumerate(chunks, start=1):

                corpus_chunks.append({
                    "global_chunk_id": chunk_id,
                    "judgment_id": row["citation"],
                    "case_title": row["title"],
                    "configuration": config_name,
                    "local_chunk_id": local_chunk_id,
                    "chunk_text": chunk["chunk_text"],
                    "word_count": chunk["word_count"],
                    "paragraph_count": chunk["paragraph_count"],
                    "split_paragraph_count": chunk["split_paragraph_count"]
                })

                chunk_id += 1

        all_chunk_sets[config_name] = pd.DataFrame(corpus_chunks)

    print("\nChunk generation completed.")

    print("=" * 70)
    print("Corpus-wide Chunk Generation Summary")
    print("=" * 70)

    for config_name, df in all_chunk_sets.items():
        print(f"{config_name:18} : {len(df):7,} chunks")

    return all_chunk_sets


def analyze_corpus_chunk_statistics(all_chunk_sets):
    """
    7.7.2 Corpus-wide Structural Analysis of Candidate Chunking Configurations

    The structural characteristics of the generated chunk sets are analysed across the complete corpus. For each candidate configuration, descriptive statistics including the total number of chunks, average and median chunk size, minimum and maximum chunk length, average number of paragraphs per chunk, and the frequency of paragraph splitting are computed. These corpus-level statistics provide an objective basis for comparing the candidate chunking configurations before selecting the final configuration for embedding generation.

    Observation:
    Corpus-wide structural analysis demonstrated clear differences among the six candidate chunking configurations. As expected, increasing the target chunk size reduced the total number of generated chunks while increasing the average number of paragraphs contained within each chunk. The 256-word configurations produced the largest number of chunks (45,971–46,075) with an average chunk size of approximately 220 words and frequent paragraph splitting. The 384-word configurations generated a moderate number of chunks (30,521–31,169) with an average chunk size of approximately 363 words, providing a balanced trade-off between retrieval granularity, semantic coherence, and computational efficiency. The 512-word configurations produced the fewest chunks (27,233–27,868), preserved almost all paragraph boundaries (only 94 split paragraphs across the entire corpus), but resulted in larger retrieval units. Increasing the overlap from approximately 10% to 20% produced only a marginal increase in the total number of generated chunks, while the remaining structural characteristics remained largely unchanged. Based on these structural characteristics, the 384-word configuration with approximately 10% overlap (chunk_384_40) was selected for the subsequent embedding generation and retrieval experiments, as it provides an effective balance between contextual continuity, retrieval granularity, and computational efficiency.

    The selected chunk configuration was subsequently used as the common corpus for all retrieval models to ensure that differences in retrieval performance were attributable to the retrieval methods rather than document segmentation.
    """
    corpus_statistics = []

    for config_name, df in all_chunk_sets.items():

        corpus_statistics.append({
            "Configuration": config_name,
            "Total Chunks": len(df),
            "Average Words": round(df["word_count"].mean(), 2),
            "Median Words": round(df["word_count"].median(), 2),
            "Minimum Words": int(df["word_count"].min()),
            "Maximum Words": int(df["word_count"].max()),
            "Average Paragraphs": round(df["paragraph_count"].mean(), 2),
            "Split Paragraphs": int(df["split_paragraph_count"].sum())
        })

    corpus_statistics_df = pd.DataFrame(corpus_statistics)

    print("=" * 80)
    print("Corpus-wide Structural Statistics")
    print("=" * 80)
    print(corpus_statistics_df)
    return corpus_statistics_df


def validate_selected_chunks(selected_chunks, corpus_df):
    """
    7.8 Validation of the Selected Chunking Configuration

    Following the structural comparison of the candidate chunking configurations, the 384-word chunk size with approximately 10% overlap (chunk_384_40) was selected for the remainder of the NYAYaRAG framework. Before proceeding to embedding generation, the selected chunk dataset is validated to ensure its completeness, structural consistency, and suitability for retrieval experiments.

    The validation includes checks for the total number of generated chunks, average chunks per judgment, chunk length statistics, empty and duplicate chunks, overlap preservation, and retention of legal context across consecutive chunks.

    7.8.3 Empty Chunk Validation
    7.8.4 Duplicate Chunk Validation

    Observations:
    The selected chunk_384_40 configuration generated 30,521 chunks across the corpus, with an average of 39.03 chunks per judgment. The average chunk length was 363.35 words, with a median of 381 words, indicating that most chunks closely approximate the configured target size of 384 words. Chunk lengths ranged from 15 to 460 words, reflecting natural variation in judgment length and paragraph structure. Overall, the results demonstrate that the refined Universal Paragraph-Aware Chunker produced structurally consistent chunks suitable for embedding generation and retrieval.

    No duplicate chunks were identified in the selected chunk dataset. This confirms that the refined Universal Paragraph-Aware Chunker generated a unique collection of text segments without unintended duplication, ensuring an efficient and reliable corpus for embedding generation and retrieval.
    """
    print("=" * 70)
    print("Selected Chunk Configuration")
    print("=" * 70)

    print(f"Configuration : chunk_384_40")
    print(f"Total Chunks  : {len(selected_chunks):,}")
    print(f"Judgments     : {corpus_df.shape[0]}")

    avg_chunks = len(selected_chunks) / len(corpus_df)

    print("=" * 70)
    print("Chunk Statistics")
    print("=" * 70)

    print(f"Total Chunks                : {len(selected_chunks):,}")
    print(f"Average Chunks / Judgment   : {avg_chunks:.2f}")
    print(f"Minimum Chunk Length        : {selected_chunks['word_count'].min()} words")
    print(f"Maximum Chunk Length        : {selected_chunks['word_count'].max()} words")
    print(f"Average Chunk Length        : {selected_chunks['word_count'].mean():.2f} words")
    print(f"Median Chunk Length         : {selected_chunks['word_count'].median():.2f} words")

    empty_chunks = (
        selected_chunks["chunk_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print("=" * 70)
    print("Empty Chunk Validation")
    print("=" * 70)

    print(f"Total Chunks : {len(selected_chunks):,}")
    print(f"Empty Chunks : {empty_chunks}")

    duplicate_chunks = selected_chunks["chunk_text"].duplicated().sum()

    print("=" * 70)
    print("Duplicate Chunk Validation")
    print("=" * 70)

    print(f"Total Chunks      : {len(selected_chunks):,}")
    print(f"Duplicate Chunks  : {duplicate_chunks:,}")


def validate_consecutive_chunks(selected_chunks):
    """
    7.8.5 Validation of Consecutive Chunks and Overlap

    The selected chunk dataset is inspected to verify that consecutive chunks preserve contextual continuity through the configured overlap. Sample consecutive chunks from the same judgment are examined to confirm that the chunking algorithm maintains the intended overlap while preserving paragraph boundaries and legal reasoning.

    Observation:
    Examination of consecutive chunks confirmed that the refined Universal Paragraph-Aware Chunker successfully preserved contextual continuity through the configured overlap. The factual narrative and judicial reasoning flowed naturally from one chunk to the next without abrupt discontinuities. The overlap ensured that important legal context was retained across chunk boundaries, thereby reducing the risk of information loss during retrieval. Minor PDF extraction artefacts, including page headers and footnote references, were observed within some chunks. These artefacts originate from the source PDF formatting rather than the chunking algorithm and do not affect the continuity of the judicial reasoning.
    """
    sample_case = selected_chunks["judgment_id"].iloc[0]

    sample_chunks = (
        selected_chunks[selected_chunks["judgment_id"] == sample_case]
        .sort_values("local_chunk_id")
        .head(3)
    )

    print("=" * 80)
    print("Validation of Consecutive Chunks")
    print("=" * 80)

    print(f"Judgment : {sample_case}")

    for _, row in sample_chunks.iterrows():

        print("\n" + "=" * 80)
        print(f"Chunk {row['local_chunk_id']}")
        print("=" * 80)

        print(f"Words      : {row['word_count']}")
        print(f"Paragraphs : {row['paragraph_count']}")

        print("\nPreview")
        print("-" * 80)
        print(row["chunk_text"][:1000])


def inspect_legal_context_preservation(selected_chunks):
    """
    7.8.6 Preservation of Legal Context

    The final validation examines whether the selected chunking configuration preserves the essential legal context required for retrieval-augmented generation. Representative chunks are inspected to verify that important legal information, including case citations, judicial reasoning, statutory references, party names, paragraph numbering, and contextual continuity, remain intact after chunking. This validation ensures that each chunk retains sufficient semantic information to support accurate retrieval and citation-grounded answer generation.

    Observation:
    Inspection of the selected chunk confirmed that the chunking process successfully preserved the essential legal context required for retrieval. The generated chunk retained the case citation, case title, judicial authorship, paragraph numbering, party names, statutory references, and the continuous factual and legal narrative of the judgment. These elements ensure that each chunk contains sufficient contextual information for embedding generation and citation-grounded retrieval. Although minor PDF extraction artefacts, such as page references and footnote numbering, remain within the text, they originate from the source documents and do not materially affect the semantic integrity or legal reasoning preserved in the chunks.
    """
    sample = selected_chunks.iloc[0]

    print("=" * 80)
    print("Preservation of Legal Context")
    print("=" * 80)

    print(f"Judgment ID      : {sample['judgment_id']}")
    print(f"Case Title       : {sample['case_title']}")
    print(f"Chunk ID         : {sample['local_chunk_id']}")
    print(f"Word Count       : {sample['word_count']}")
    print(f"Paragraph Count  : {sample['paragraph_count']}")

    print("\nChunk Text")
    print("-" * 80)
    print(sample["chunk_text"][:2500])


def save_selected_chunks(selected_chunks, output_path=r"D:\_ResearchNotebooks\NyayaRAG\data\processed\chunks_384_40.parquet"):
    """
    Save the Selected Chunk Corpus
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    selected_chunks.to_parquet(
        output_file,
        index=False
    )
    print(f"Selected chunks saved to: {output_file}")


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    processed_corpus_file = PROCESSED_DIR / "metadata_2024_processed.parquet"

    corpus_df = pd.read_parquet(processed_corpus_file)
    if "judgment_text" not in corpus_df.columns:
        MARKER = "Judgment / Order of the Supreme Court"
        corpus_df["judgment_text"] = corpus_df["clean_text"].apply(
            lambda t: str(t)[str(t).find(MARKER):].strip() if str(t).find(MARKER) != -1 else str(t)
        )

    chunk_configs = {
        "chunk_256_26": {"chunk_size": 256, "chunk_overlap": 26, "overlap_percent": "10%"},
        "chunk_256_51": {"chunk_size": 256, "chunk_overlap": 51, "overlap_percent": "20%"},
        "chunk_384_40": {"chunk_size": 384, "chunk_overlap": 40, "overlap_percent": "10%"},
        "chunk_384_80": {"chunk_size": 384, "chunk_overlap": 80, "overlap_percent": "20%"},
        "chunk_512_50": {"chunk_size": 512, "chunk_overlap": 50, "overlap_percent": "10%"},
        "chunk_512_100": {"chunk_size": 512, "chunk_overlap": 100, "overlap_percent": "20%"}
    }

    all_chunk_sets = generate_corpus_chunks(corpus_df, chunk_configs)
    corpus_statistics_df = analyze_corpus_chunk_statistics(all_chunk_sets)
    selected_chunks = all_chunk_sets["chunk_384_40"]
    validate_selected_chunks(selected_chunks, corpus_df)
    validate_consecutive_chunks(selected_chunks)
    inspect_legal_context_preservation(selected_chunks)
    save_selected_chunks(selected_chunks, str(PROCESSED_DIR / "chunks_384_40.parquet"))
