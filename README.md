# NYAYaRAG — Data Preparation and Indexing

This notebook prepares the **NYAYaRAG legal corpus** for downstream retrieval and RAG experiments.

The pipeline takes Supreme Court of India judgments and metadata, extracts and cleans the judgment text, creates retrieval chunks using the `chunk_384_40` configuration, performs corpus validation/analysis, and builds retrieval indexes for lexical and dense retrieval methods.

## Pipeline Overview

```text
Raw PDFs + Metadata
        │
        ▼
Data Validation
        │
        ▼
PDF Text Extraction
        │
        ▼
Text Cleaning
        │
        ▼
Corpus Analysis / EDA
        │
        ▼
Chunking
(chunk_384_40)
        │
        ▼
30,521 chunks
        │
        ├───────────────┬────────────────┐
        ▼               ▼                ▼
      TF-IDF           BM25       Dense Embeddings
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                             E5         BGE    InLegalBERT
                              │          │          │
                              ▼          ▼          ▼
                           ChromaDB   ChromaDB   ChromaDB
```

## Dataset

The notebook processes:

- **782 Supreme Court of India judgments**
- **2024 judgments**
- English-language judgment PDFs
- Corresponding metadata records

The notebook validates that the metadata and PDF corpus correspond to one another before processing.

## Directory Structure

The notebook expects a project structure similar to:

```text
NyayaRAG/
├── data/
│   ├── raw/
│   │   ├── pdf/
│   │   │   └── year=2024/
│   │   │       └── english/
│   │   └── metadata/
│   │       └── year=2024/
│   │           └── metadata_2024.parquet
│   │
│   ├── processed/
│   └── indexes/
│
└── results/
```

The exact generated files/paths should be taken from the notebook configuration when reproducing the pipeline.

## 1. Environment Setup

The notebook:

- Sets up the working directories.
- Mounts/uses Google Drive where required.
- Imports the libraries used throughout the pipeline.
- Sets a random seed (`42`) for reproducibility.
- Reports relevant Python/library versions.

## 2. Metadata Validation

The metadata is loaded from the 2024 metadata Parquet file.

The notebook checks:

- Number of metadata records
- Metadata columns
- Data types
- Missing values
- Duplicate records
- Duplicate citations
- PDF-path consistency
- Date fields

`decision_date` is converted to a datetime representation so that it can be handled consistently during analysis.

The expected corpus contains **782 metadata records** corresponding to the 782 judgment PDFs.

## 3. PDF Text Extraction

Judgment PDFs are processed with **PyMuPDF** to extract their text.

The extraction pipeline is:

```text
Judgment PDF
     │
     ▼
PyMuPDF
     │
     ▼
Raw extracted text
     │
     ▼
Saved text representation
```

The notebook validates extraction on the corpus and reports:

- PDFs processed: **782**
- Successful extractions: **782**
- Failed extractions: **0**

This creates the text corpus used by subsequent preprocessing steps.

## 4. Text Cleaning

PDF-extracted text can contain formatting artifacts introduced during PDF parsing.

The notebook cleans the extracted judgment text while attempting to preserve the substantive legal content.

The cleaning stage is intended to retain:

- Legal reasoning
- Case information
- Citations
- Statutory references
- Dates
- Paragraph content

while reducing extraction-related formatting noise.

The notebook also compares raw and cleaned text statistics to validate the transformation.

## 5. Corpus Analysis

Before indexing, the notebook analyzes the cleaned corpus.

The reported corpus statistics include approximately:

| Metric | Value |
|---|---:|
| Judgments | 782 |
| Words | 6,820,528 |
| Characters | 40,992,347 |
| Sentences | 342,405 |
| Paragraphs | 47,258 |
| Average judgment length | ~8,722 words |
| Median judgment length | ~5,240 words |

The notebook also examines judgment-length distributions and identifies unusually short/long judgments.

This analysis helps determine whether the corpus is suitable for chunk-based retrieval.

## 6. Chunking

The cleaned judgments are divided into smaller retrieval units.

The selected configuration is:

```text
chunk_384_40
```

which corresponds to approximately:

```text
Chunk size : 384 words
Overlap    : 40 words
```

Conceptually:

```text
Chunk 1: [──────────────── 384 words ────────────────]

Chunk 2:                         [──────────────── 384 words ────────────────]
                                  ← 40-word overlap →

Chunk 3:                                                        [──────────────]
```

The overlap helps preserve context across chunk boundaries.

### Final Chunk Corpus

The notebook produces:

```text
Judgments : 782
Chunks    : 30,521
```

The chunking stage also performs validation checks such as:

- Empty chunks
- Duplicate chunks
- Chunk continuity/overlap
- Representative chunk inspection

The notebook reports **0 empty chunks**.

## 7. Retrieval Indexes

The resulting 30,521 chunks are used to build several retrieval approaches.

### Lexical Retrieval

#### TF-IDF

TF-IDF provides a sparse lexical retrieval baseline.

```text
Query
  │
  ▼
TF-IDF representation
  │
  ▼
Similarity against chunk representations
  │
  ▼
Top-k chunks
```

#### BM25

BM25 provides another lexical retrieval baseline based on term matching and document/chunk statistics.

```text
Query
  │
  ▼
BM25 scoring
  │
  ▼
Ranked chunks
```

These two methods provide non-neural baselines against which dense retrieval can be compared.

## 8. Dense Retrieval

The notebook also creates semantic embeddings for every chunk.

The dense retrieval models used are:

- **E5**
- **BGE**
- **InLegalBERT**

The general process is:

```text
Legal chunk
    │
    ▼
Embedding model
    │
    ▼
Dense vector
    │
    ▼
Vector database
```

### E5

The E5 model converts each legal chunk into an embedding vector for semantic similarity search.

### BGE

BGE is used as another dense embedding baseline.

### InLegalBERT

InLegalBERT provides a legal-domain-oriented embedding representation for the Indian legal corpus.

This allows the project to compare general-purpose semantic representations with a legal-domain model.

## 9. ChromaDB

The dense embeddings are stored in ChromaDB collections.

Conceptually:

```text
E5
 └── 30,521 chunk embeddings

BGE
 └── 30,521 chunk embeddings

InLegalBERT
 └── 30,521 chunk embeddings
```

Each vector is associated with chunk/judgment metadata so that retrieved results can be traced back to their source.

Typical retrieval metadata includes information such as:

- Judgment identifier
- Case title
- Chunk identifier
- Chunk text
- Source/judgment metadata

## 10. Retrieval Sanity Checks

After creating the indexes, the notebook executes sample retrieval queries.

For dense retrieval, the query is embedded using the corresponding embedding model and compared against the stored chunk vectors.

The notebook displays retrieval information such as:

```text
Rank
Chunk ID
Distance
Case Title
Judgment ID
Chunk preview
```

These checks are intended to verify that the indexes are functioning and returning legally relevant material.

## 11. Reproducibility and Configuration

The notebook records model/index configuration information, including relevant embedding-model parameters and corpus/index characteristics.

This is important because downstream experiments should use the same corpus and indexing configuration when comparing retrieval methods.

## Final Outputs

The major outputs of this notebook are:

```text
782 judgments
        │
        ▼
Extracted + cleaned text
        │
        ▼
30,521 chunks
        │
        ▼
chunk_384_40
        │
        ├── TF-IDF index
        ├── BM25 index
        ├── E5 embeddings / ChromaDB
        ├── BGE embeddings / ChromaDB
        └── InLegalBERT embeddings / ChromaDB
```

These outputs form the retrieval layer used by subsequent NYAYaRAG experiments.

## Retrieval Methods Compared

| Method | Type | Representation | Purpose |
|---|---|---|---|
| TF-IDF | Sparse | Term-weight vectors | Lexical baseline |
| BM25 | Sparse | Term-based ranking | Lexical baseline |
| E5 | Dense | Semantic embeddings | General semantic retrieval |
| BGE | Dense | Semantic embeddings | General semantic retrieval |
| InLegalBERT | Dense | Legal-domain embeddings | Indian legal semantic retrieval |

## Relationship to Downstream QA/RAG Evaluation

This notebook is an **upstream data-preparation and indexing stage**.

The relationship is:

```text
NYAYaRAG Data Preparation
        │
        ├── 782 judgments
        ├── cleaned text
        ├── chunk_384_40
        └── 30,521 chunks
                 │
                 ▼
       Retrieval / QA experiments
                 │
                 ▼
          RAG evaluation
```

Therefore, `chunk_384_40` refers to the **retrieval corpus configuration**, not to the QA dataset itself.

## Key Takeaway

The purpose of this notebook is to create a **clean, validated, consistently chunked, and indexed legal corpus** so that different retrieval strategies can be evaluated fairly on the same underlying Supreme Court judgment data.

The core transformation is:

```text
782 Supreme Court judgments
          ↓
PDF text extraction
          ↓
Text cleaning
          ↓
384-word chunks
40-word overlap
          ↓
30,521 retrieval chunks
          ↓
TF-IDF / BM25 / E5 / BGE / InLegalBERT
          ↓
Searchable retrieval indexes
```
