"""
=============================================================================
NyayaRAG Generation & Evaluation — Script 04: Evidence Packaging & Prompt Engineering
=============================================================================

This script implements Sections 5, 6, and 7 of the generation notebook (Cells 34-44):
  - Evidence Package Builder: Assembles Top-5 Hybrid retrieved judgment chunks
    (Case Title, Official Citation, Retrieval Rank, Chunk Excerpt) as SOURCE 1..5.
  - Frozen System Prompt: FINAL_SYSTEM_PROMPT with 8 mandatory legal grounding rules
    and strict output specifications (Format A: Sufficient Evidence vs. Format B: Insufficient Evidence).
  - Question-Specific User Prompt: Combines legal query and retrieved excerpts without
    leaking the ground-truth answerability label.
  - Structured Response Parser: Robust regex extraction of LEGAL ANSWER and
    AUTHORITIES RELIED UPON, validating formatting and identifying abstention decisions.
=============================================================================
"""

import re
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Ensure UTF-8 output on Windows if available
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(legacy_windows=False)

PROJECT_ROOT = Path(r"D:\_ResearchNotebooks\NyayaRAG")

# =============================================================================
# 1. Frozen System Prompt & Format Definitions
# =============================================================================

FINAL_SYSTEM_PROMPT = """
You are NYAYaRAG, an evidence-grounded legal research assistant.

Answer the legal question ONLY from the supplied excerpts of Supreme Court
judgments.

MANDATORY RULES

1. Do not use outside legal knowledge, memory, assumptions, or information
   not contained in the supplied evidence.

2. State only legal propositions that are supported by the supplied judgment
   excerpts.

3. Write the answer as one or more concise legal paragraphs. Avoid numbered
   lists unless they are necessary for clarity.

4. Whenever relying on a judgment, cite its case title and official citation
   exactly as supplied.

5. Do not cite internal retrieval labels such as SOURCE 1, S1, or S2 in the
   legal answer.

6. Do not treat counsel's submissions, allegations, or lower-court reasoning
   as a holding of the Supreme Court unless the excerpt clearly adopts it.

7. Do not invent or alter case names, citations, statutory provisions,
   quotations, facts, procedural history, or holdings.

8. If the supplied evidence does not reliably answer the central legal issue,
   do not guess. Return the prescribed abstention response.

Use exactly one of the following two formats.

FORMAT A — SUFFICIENT EVIDENCE

LEGAL ANSWER

<Substantive legal answer in one or more concise paragraphs, containing
case-title and official-citation references where relied upon.>

AUTHORITIES RELIED UPON

<List each relied-upon judgment once, using its case title and official
citation.>


FORMAT B — INSUFFICIENT EVIDENCE

LEGAL ANSWER

The retrieved judgments do not provide sufficient evidence to answer this
question reliably.

AUTHORITIES RELIED UPON

None

Do not add any other heading, decision label, reason, preface, or conclusion.
""".strip()

ABSTENTION_TEXT = (
    "The retrieved judgments do not provide sufficient evidence "
    "to answer this question reliably."
)


# =============================================================================
# 2. Evidence Package Construction
# =============================================================================

def build_evidence_package(question_id: str, hybrid_retrieval: pd.DataFrame) -> str:
    """
    Build the evidence package formatted as SOURCE 1..5.
    (Notebook Cell 37)
    """
    rows = (
        hybrid_retrieval
        .query("question_id == @question_id")
        .sort_values("retrieval_rank")
    )

    evidence_parts = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        part = (
            f"==================================================\n"
            f"SOURCE {i}\n"
            f"==================================================\n\n"
            f"Case Title:\n{row['case_title']}\n\n"
            f"Official Citation:\n{row['judgment_id']}\n\n"
            f"Retrieval Rank:\n{int(row['retrieval_rank'])}\n\n"
            f"Evidence Excerpt:\n{row['chunk_text']}\n"
        )
        evidence_parts.append(part)

    return "\n".join(evidence_parts)


def build_final_user_prompt(
    question_id: str,
    pilot_questions: pd.DataFrame,
    hybrid_retrieval: pd.DataFrame
) -> str:
    """
    Build the question-specific prompt without exposing the answerability label.
    (Notebook Cell 43)
    """
    q_rows = pilot_questions.loc[pilot_questions["question_id"].eq(question_id), "question"]
    if len(q_rows) != 1:
        raise ValueError(f"Expected 1 question for {question_id}, found {len(q_rows)}")

    question = q_rows.iloc[0]
    evidence = build_evidence_package(question_id, hybrid_retrieval)

    return (
        f"LEGAL QUESTION\n\n"
        f"{question}\n\n\n"
        f"RETRIEVED SUPREME COURT JUDGMENT EXCERPTS\n\n"
        f"{evidence}\n\n\n"
        f"TASK\n\n"
        f"Answer the legal question in accordance with the mandatory rules and one of "
        f"the two permitted response formats."
    ).strip()


# =============================================================================
# 3. Response Parser and Abstention Identifier
# =============================================================================

def parse_generation_response(response_text: str) -> Dict[str, Any]:
    """
    Extract legal answer and authorities from the structured response.
    (Notebook Cell 44)
    """
    text = str(response_text or "").strip()

    answer_match = re.search(
        r"LEGAL ANSWER\s*(.*?)\s*AUTHORITIES RELIED UPON",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    authorities_match = re.search(
        r"AUTHORITIES RELIED UPON\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    legal_answer = answer_match.group(1).strip() if answer_match else ""
    authorities_text = authorities_match.group(1).strip() if authorities_match else ""

    norm_answer = " ".join(legal_answer.lower().split())
    norm_abstention = " ".join(ABSTENTION_TEXT.lower().split())

    model_decision = (
        "INSUFFICIENT_EVIDENCE"
        if norm_answer == norm_abstention
        else "ANSWER"
    )

    format_valid = bool(
        answer_match
        and authorities_match
        and legal_answer
        and authorities_text
    )

    return {
        "model_decision": model_decision,
        "generated_answer": legal_answer,
        "authorities_text": authorities_text,
        "format_valid": format_valid
    }


# =============================================================================
# 4. Unit Testing and Demonstration
# =============================================================================

def run_prompt_and_parser_demonstration() -> None:
    """Validate prompt assembly and parser with test fixtures."""
    # Load data from script 01
    s1_path = Path(__file__).resolve().parent / "01_environment_and_benchmark_setup.py"
    spec = importlib.util.spec_from_file_location("s1_mod", s1_path)
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    pilot_questions, hybrid_retrieval = s1.load_pilot_benchmark_and_retrieval()

    # Test sample prompt construction for RAW001
    sample_prompt = build_final_user_prompt("RAW001", pilot_questions, hybrid_retrieval)
    assert "LEGAL QUESTION" in sample_prompt
    assert "SOURCE 1" in sample_prompt
    assert "SOURCE 5" in sample_prompt
    assert "TASK" in sample_prompt

    console.print(Panel(
        f"[bold cyan]Sample Generated User Prompt (RAW001 snippet, first 400 chars):[/bold cyan]\n\n"
        f"{sample_prompt[:400]}...\n\n"
        f"[dim]Total prompt length: {len(sample_prompt):,} characters[/dim]",
        title="[bold green]Prompt Engineering Verification[/bold green]",
        box=box.ROUNDED
    ))

    # Test parser on dummy valid answer
    dummy_answer = (
        "LEGAL ANSWER\n\n"
        "Delay in lodging an FIR is not fatal per se if satisfactorily explained.\n\n"
        "AUTHORITIES RELIED UPON\n\n"
        "RAMA DEVI versus THE STATE OF BIHAR AND OTHERS ([2024] 10 S.C.R. 1313)"
    )
    p1 = parse_generation_response(dummy_answer)
    assert p1["format_valid"] is True
    assert p1["model_decision"] == "ANSWER"
    assert "Delay in lodging" in p1["generated_answer"]
    assert "RAMA DEVI" in p1["authorities_text"]

    # Test parser on dummy abstention
    dummy_abstention = (
        "LEGAL ANSWER\n\n"
        "The retrieved judgments do not provide sufficient evidence to answer this question reliably.\n\n"
        "AUTHORITIES RELIED UPON\n\n"
        "None"
    )
    p2 = parse_generation_response(dummy_abstention)
    assert p2["format_valid"] is True
    assert p2["model_decision"] == "INSUFFICIENT_EVIDENCE"
    assert p2["authorities_text"] == "None"

    console.print(Panel(
        "[bold green][OK] Response Parser Verified:[/bold green]\n"
        "• Format A (Sufficient Evidence) parsed successfully -> model_decision = ANSWER\n"
        "• Format B (Insufficient Evidence) parsed successfully -> model_decision = INSUFFICIENT_EVIDENCE",
        box=box.ROUNDED
    ))


def main():
    console.print(Panel(
        "[bold bright_white]NYAYARAG GENERATION & EVALUATION — SCRIPT 04[/bold bright_white]\n"
        "[dim cyan]Evidence Package Construction & Prompt Engineering Protocol[/dim cyan]",
        box=box.DOUBLE,
        border_style="bright_blue"
    ))

    # Display system prompt rules summary
    table = Table(title="Mandatory System Rules for Legal Evidence Grounding", box=box.ROUNDED)
    table.add_column("Rule #", style="bold yellow", justify="center")
    table.add_column("Rule Summary", style="white")

    rules = [
        ("Rule 1", "Zero outside knowledge, memory, or assumptions."),
        ("Rule 2", "Strict grounding: state only propositions found in judgment excerpts."),
        ("Rule 3", "Concise legal paragraph style (no lists unless necessary)."),
        ("Rule 4", "Exact case title and official citation format."),
        ("Rule 5", "No internal retrieval tokens (SOURCE 1..5) in the generated answer."),
        ("Rule 6", "No counsel submissions treated as holdings unless explicitly adopted."),
        ("Rule 7", "Zero fabrication of statutes, cases, or judicial holdings."),
        ("Rule 8", "Safe abstention: if evidence is insufficient, do not guess; abstain.")
    ]
    for r_num, r_desc in rules:
        table.add_row(r_num, r_desc)
    console.print(table)

    # Test prompt and parser
    console.print("\n[cyan]Executing prompt construction and parser validation test...[/cyan]")
    run_prompt_and_parser_demonstration()

    console.print(Panel(
        "[bold green][OK] Script 04 finished: Evidence packaging and prompt protocol validated.[/bold green]",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
