# NYAYaRAG Modular Pipeline: 17-Section Curriculum (Exercises & Solutions)

This package breaks down the monolithic Jupyter notebook [`01_NYAYaRAG_Data_Preparation_and_Indexing.ipynb`](file:///d:/_ResearchNotebooks/NyayaRAG/notebook_1/01_NYAYaRAG_Data_Preparation_and_Indexing.ipynb) into **17 standalone, modular sections**.

Each section provides:
1. **An Exercise Script (`*_exercise.py`)**: Designed for guided learning with rich docstrings, architectural background, typed function skeletons, explicit `TODO` implementation prompts, and self-contained assertion test suites.
2. **A Solution Script (`*_solution.py`)**: A production-grade, executable Python script featuring the complete reference implementation, logging, performance profiling, persistence, and validation gates.

---

## Architecture and Curriculum Map

```text
Section 01: Environment Setup & Project Configuration
    │
    ▼
Section 02: Dataset Acquisition & Raw Corpus Loading
    │
    ▼
Section 03: Metadata Quality Assessment & Deduplication
    │
    ▼
Section 04: PDF Text Extraction via PyMuPDF (782 PDFs)
    │
    ▼
Section 05: Text Cleaning & Preprocessing Pipeline
    │
    ▼
Section 06: EDA — Corpus Length & Outliers
    │
    ▼
Section 07: EDA — Structural, Lexical & Statutory Profiling
    │
    ▼
Section 08: Universal Paragraph-Aware Chunking Strategy
    │
    ▼
Section 09: Editorial Front Matter Detection & Removal
    │
    ▼
Section 10: Refined Chunker & Corpus-Wide Chunk Generation (30,521 chunks)
    │
    ▼
Section 11: Chunk Quality Validation & Legal Context Preservation
    │
    ├────────────────────────┬────────────────────────┐
    ▼                        ▼                        ▼
Section 12: TF-IDF       Section 13: BM25      Sections 14-16: Dense Embeddings
(Sparse Lexical)         (Probabilistic)              │
                                           ┌──────────┼──────────┐
                                           ▼          ▼          ▼
                                      Section 14  Section 15 Section 16
                                          E5         BGE     InLegalBERT
                                           │          │          │
                                           └──────────┼──────────┘
                                                      │
                                                      ▼
                                       Section 17: Multi-Model Evaluation
                                       & Final System Validation Report
```

---

## Detailed Section Breakdown

| Section # | Module Name | Notebook Cells | Exercise Script | Solution Script | Description |
|:---:|:---|:---:|:---|:---|:---|
| **01** | **Environment Setup** | 0 – 6 | [`section_01_environment_setup_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_01_environment_setup_exercise.py) | [`section_01_environment_setup_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_01_environment_setup_solution.py) | Seeds, directory creation, dependency audits |
| **02** | **Dataset Loading** | 7 – 15 | [`section_02_dataset_loading_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_02_dataset_loading_exercise.py) | [`section_02_dataset_loading_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_02_dataset_loading_solution.py) | Loading raw metadata parquet and 782 judgment PDFs |
| **03** | **Metadata Validation** | 16 – 29 | [`section_03_metadata_validation_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_03_metadata_validation_exercise.py) | [`section_03_metadata_validation_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_03_metadata_validation_solution.py) | Completeness, duplicate citations, date standardization |
| **04** | **PDF Text Extraction** | 30 – 43 | [`section_04_pdf_text_extraction_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_04_pdf_text_extraction_exercise.py) | [`section_04_pdf_text_extraction_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_04_pdf_text_extraction_solution.py) | PyMuPDF text extraction with positional sorting & audit log |
| **05** | **Text Cleaning Pipeline** | 44 – 58 | [`section_05_text_cleaning_pipeline_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_05_text_cleaning_pipeline_exercise.py) | [`section_05_text_cleaning_pipeline_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_05_text_cleaning_pipeline_solution.py) | 9-step regex cleaner preserving statutory citations |
| **06** | **EDA: Length & Outliers** | 59 – 74 | [`section_06_eda_length_and_outliers_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_06_eda_length_and_outliers_exercise.py) | [`section_06_eda_length_and_outliers_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_06_eda_length_and_outliers_solution.py) | Word distribution, 1.5*IQR outlier detection, extreme cases |
| **07** | **EDA: Structural Profiling** | 75 – 116 | [`section_07_eda_structural_and_lexical_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_07_eda_structural_and_lexical_exercise.py) | [`section_07_eda_structural_and_lexical_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_07_eda_structural_and_lexical_solution.py) | Sentences, paragraphs, top legal terms, statute regex |
| **08** | **Paragraph-Aware Chunking** | 117 – 138 | [`section_08_paragraph_aware_chunking_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_08_paragraph_aware_chunking_exercise.py) | [`section_08_paragraph_aware_chunking_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_08_paragraph_aware_chunking_solution.py) | Paragraph unit accumulation, overlap trailing buffer |
| **09** | **Front Matter Removal** | 139 – 158 | [`section_09_front_matter_removal_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_09_front_matter_removal_exercise.py) | [`section_09_front_matter_removal_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_09_front_matter_removal_solution.py) | Stripping publisher banners to isolate judicial reasoning |
| **10** | **Corpus Chunk Generation** | 159 – 171 | [`section_10_refined_chunker_generation_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_10_refined_chunker_generation_exercise.py) | [`section_10_refined_chunker_generation_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_10_refined_chunker_generation_solution.py) | Corpus generation producing exactly 30,521 chunks (`chunk_384_40`) |
| **11** | **Chunk Quality Validation** | 172 – 187 | [`section_11_chunk_quality_validation_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_11_chunk_quality_validation_exercise.py) | [`section_11_chunk_quality_validation_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_11_chunk_quality_validation_solution.py) | Verifying 0 empty chunks, 0 duplicates, overlap continuity |
| **12** | **TF-IDF Lexical Indexing** | 188 – 207 | [`section_12_tfidf_indexing_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_12_tfidf_indexing_exercise.py) | [`section_12_tfidf_indexing_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_12_tfidf_indexing_solution.py) | Unigram/bigram vectorizer, CSR matrix, cosine similarity |
| **13** | **BM25 Lexical Indexing** | 208 – 220 | [`section_13_bm25_indexing_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_13_bm25_indexing_exercise.py) | [`section_13_bm25_indexing_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_13_bm25_indexing_solution.py) | BM25Okapi inverted index, term saturation, score ranking |
| **14** | **E5 Dense Embeddings** | 221 – 245 | [`section_14_e5_dense_indexing_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_14_e5_dense_indexing_exercise.py) | [`section_14_e5_dense_indexing_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_14_e5_dense_indexing_solution.py) | `intfloat/e5-base-v2` with `passage:` and `query:` prefixes |
| **15** | **BGE Dense Embeddings** | 246 – 263 | [`section_15_bge_dense_indexing_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_15_bge_dense_indexing_exercise.py) | [`section_15_bge_dense_indexing_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_15_bge_dense_indexing_solution.py) | `BAAI/bge-base-en-v1.5` normalized semantic representations |
| **16** | **InLegalBERT Domain Index** | 264 – 281 | [`section_16_inlegalbert_indexing_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_16_inlegalbert_indexing_exercise.py) | [`section_16_inlegalbert_indexing_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_16_inlegalbert_indexing_solution.py) | `law-ai/InLegalBERT` Indian legal domain pre-trained retrieval |
| **17** | **Multi-Model Evaluation** | 282 – 287 | [`section_17_multi_model_evaluation_exercise.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_17_multi_model_evaluation_exercise.py) | [`section_17_multi_model_evaluation_solution.py`](file:///d:/_ResearchNotebooks/NyayaRAG/sections/section_17_multi_model_evaluation_solution.py) | Master metadata JSON, cross-model retrieval, audit report |

---

## Running the Code

### 1. Run Any Single Section Solution
```powershell
python sections/section_12_tfidf_indexing_solution.py
```

### 2. Run Any Single Section Exercise to Test Your Implementation
```powershell
python sections/section_12_tfidf_indexing_exercise.py
```

### 3. Run via Master Solution Runner
```powershell
# Run a specific section (e.g. section 7)
python sections/run_all_solutions.py --section 7

# Run all fast sections
python sections/run_all_solutions.py

# Run all 17 sections end-to-end
python sections/run_all_solutions.py --all
```
