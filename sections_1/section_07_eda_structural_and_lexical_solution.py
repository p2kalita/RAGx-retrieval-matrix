"""
Section 07: EDA — Structural, Lexical, and Statutory Profiling — SOLUTION
=========================================================================

This module computes sentence and paragraph distributions, vocabulary statistics,
frequent legal terms, Indian statutory citations, and lexical diversity metrics,
and saves the consolidated metadata parquet file (Notebook Cells 75 to 116).
"""

import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sections_1.common import (
    PROCESSED_METADATA_FILE,
    print_header,
    load_processed_corpus,
)

STATUTE_PATTERNS = {
    "Constitution of India": r"\bConstitution\b|\bConstitution of India\b|\bArticle\s+\d+\b",
    "Indian Penal Code (IPC)": r"\bIndian Penal Code\b|\bIPC\b|\bSection\s+302\b|\bSection\s+307\b",
    "Code of Criminal Procedure (CrPC)": r"\bCode of Criminal Procedure\b|\bCrPC\b|\bSection\s+438\b|\bSection\s+482\b",
    "Code of Civil Procedure (CPC)": r"\bCode of Civil Procedure\b|\bCPC\b|\bOrder\s+[X|V|I]+\b",
    "Indian Evidence Act": r"\bEvidence Act\b|\bIndian Evidence Act\b|\bSection\s+27\b",
    "Negotiable Instruments Act": r"\bNegotiable Instruments Act\b|\bSection\s+138\b|\bNI Act\b",
    "Arbitration and Conciliation Act": r"\bArbitration and Conciliation Act\b|\bSection\s+34\b|\bSection\s+11\b|\bSection\s+9\b",
    "Insolvency and Bankruptcy Code (IBC)": r"\bInsolvency and Bankruptcy Code\b|\bIBC\b|\bNCLT\b|\bNCLAT\b|\bSection\s+7\b|\bSection\s+9\b",
    "Income Tax Act": r"\bIncome Tax Act\b|\bAssessing Officer\b|\bITAT\b",
    "Motor Vehicles Act": r"\bMotor Vehicles Act\b|\bMACT\b",
}


def compute_structural_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentence count, paragraph count, and words-per-paragraph for each judgment.
    """
    df = df.copy()

    # Sentence count
    df["sentence_count"] = df["clean_text"].apply(
        lambda text: len([s for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()])
    )

    # Paragraph count
    df["paragraph_count"] = df["clean_text"].apply(
        lambda text: len([p for p in re.split(r"\n\s*\n", str(text)) if p.strip()])
    )

    if "clean_words" not in df.columns:
        df["clean_words"] = df["clean_text"].astype(str).str.split().str.len()

    # Average words per paragraph
    df["avg_words_per_paragraph"] = (
        df["clean_words"] / df["paragraph_count"].replace(0, 1)
    ).round(2)

    return df


def extract_top_legal_terms(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """
    Count frequency of standard legal terms across the entire corpus.
    """
    legal_target_terms = [
        "court", "section", "act", "state", "case", "order",
        "appellant", "respondent", "appeal", "judgment", "high court",
        "law", "constitution", "counsel", "jurisdiction", "trial", "evidence"
    ]
    all_text = " ".join(df["clean_text"].astype(str)).lower()

    results = []
    for term in legal_target_terms:
        # Match word or phrase boundary
        pat = rf"\b{re.escape(term)}\b"
        count = len(re.findall(pat, all_text))
        results.append({"Legal Term": term, "Frequency": count})

    term_df = pd.DataFrame(results).sort_values("Frequency", ascending=False).reset_index(drop=True)
    return term_df.head(top_k)


def compute_statute_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count mentions of major Indian statutes using regular expressions.
    """
    results = []

    for name, pattern in STATUTE_PATTERNS.items():
        total_mentions = 0
        citing_judgments = 0

        for text in df["clean_text"].astype(str):
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                total_mentions += matches
                citing_judgments += 1

        results.append({
            "Statute": name,
            "Total Mentions": total_mentions,
            "Judgments Citing": citing_judgments,
            "Corpus Citing %": round((citing_judgments / len(df)) * 100, 2),
        })

    statute_df = pd.DataFrame(results).sort_values("Total Mentions", ascending=False).reset_index(drop=True)
    return statute_df


def compute_lexical_diversity(text: str) -> float:
    """
    Calculate Type-Token Ratio (unique alphabetic words / total alphabetic words).
    """
    words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def run_section_07(save_metadata: bool = True) -> Dict[str, Any]:
    """Execute complete Section 07 pipeline."""
    print_header("Section 7: EDA — Structural, Lexical & Statutory Profiling")

    corpus_df = load_processed_corpus()
    print(f"Analyzing {len(corpus_df):,} judgments...")

    corpus_df = compute_structural_statistics(corpus_df)
    corpus_df["lexical_diversity"] = corpus_df["clean_text"].apply(compute_lexical_diversity)

    print("\n1. Structural Statistics:")
    print(f"  - Total Sentences in Corpus : {corpus_df['sentence_count'].sum():,}")
    print(f"  - Avg Sentences / Judgment  : {corpus_df['sentence_count'].mean():,.1f}")
    print(f"  - Total Paragraphs in Corpus: {corpus_df['paragraph_count'].sum():,}")
    print(f"  - Avg Paragraphs / Judgment : {corpus_df['paragraph_count'].mean():,.1f}")
    print(f"  - Avg Words / Paragraph     : {corpus_df['avg_words_per_paragraph'].mean():,.1f}")

    print("\n2. Lexical Diversity (Type-Token Ratio):")
    print(f"  - Mean Lexical Diversity    : {corpus_df['lexical_diversity'].mean():.4f}")
    print(f"  - Median Lexical Diversity  : {corpus_df['lexical_diversity'].median():.4f}")

    print("\n3. Top Legal Terms:")
    terms_df = extract_top_legal_terms(corpus_df, top_k=8)
    print(terms_df.to_string(index=False))

    print("\n4. Statute Citation Distribution (Top 6):")
    statute_df = compute_statute_frequencies(corpus_df)
    print(statute_df.head(6).to_string(index=False))

    if save_metadata:
        corpus_df.to_parquet(PROCESSED_METADATA_FILE, index=False)
        print(f"\nSaved enriched processed metadata to: {PROCESSED_METADATA_FILE}")

    print("\n Section 7 structural and statutory profiling completed successfully.")
    return {
        "corpus_df": corpus_df,
        "terms_df": terms_df,
        "statute_df": statute_df,
    }


if __name__ == "__main__":
    run_section_07()
