"""
NYAYaRAG: A Legal Research Assistant
Script 03: Evaluation Rubric, Weighted Criteria & Prompt Builder (Notebook 3 Simulated)

This script encapsulates Step 4 of the LLM-as-a-Judge workflow:
- Defines the complete 7-dimension judicial evaluation rubric.
- Enforces strict dimension weightings (totaling 100%).
- Formats structured row contexts into the frozen evaluation prompt template.
- Specifies the exact JSON output schema for deterministic evaluation parsing.
- Displays Rich tables of evaluation dimensions, criteria, and relative weights.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Official dimension weights from Notebook 3 Step 4 & 5
WEIGHTS: Dict[str, float] = {
    "humanization_score": 0.25,
    "indian_english_authenticity_score": 0.20,
    "conversational_naturalness_score": 0.15,
    "experience_level_alignment_score": 0.10,
    "clarity_score": 0.10,
    "legal_terminology_context_score": 0.10,
    "reconstruction_plausibility_score": 0.10,
}

RUBRIC_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "dimension": "Humanization",
        "key": "humanization_score",
        "weight": "25%",
        "description": "Does it sound naturally written by a real person? Phrasing, flow, tone, lack of robotic wording.",
    },
    {
        "dimension": "Indian English Authenticity",
        "key": "indian_english_authenticity_score",
        "weight": "20%",
        "description": "Natural Indian legal phrasing, terminology, and institutional register without forced colloquialisms.",
    },
    {
        "dimension": "Conversational Naturalness",
        "key": "conversational_naturalness_score",
        "weight": "15%",
        "description": "Would a real legal practitioner or judicial officer plausibly ask this question in research?",
    },
    {
        "dimension": "Experience-Level Alignment",
        "key": "experience_level_alignment_score",
        "weight": "10%",
        "description": "Does vocabulary and complexity align with the contributor's experience band without stereotyping?",
    },
    {
        "dimension": "Clarity",
        "key": "clarity_score",
        "weight": "10%",
        "description": "Is the question understandable, focused, and unambiguous in its legal proposition?",
    },
    {
        "dimension": "Legal Terminology & Context",
        "key": "legal_terminology_context_score",
        "weight": "10%",
        "description": "Appropriate statutory terminology and doctrines; penalizes false legal assumptions.",
    },
    {
        "dimension": "Reconstruction Plausibility",
        "key": "reconstruction_plausibility_score",
        "weight": "10%",
        "description": "Fidelity to the visible text fragment and supplied candidate questions; penalizes unsupported additions.",
    },
]

# Canonical Evaluation Prompt from Notebook 3 Cell 10
EVALUATION_PROMPT = """
You are an expert evaluator of questions generated for NyayaRAG,
a legal question-answering system focused on Indian law.

Evaluate the PREFERRED QUESTION using the supplied row context.

Score each dimension from 0 to 100.

1. Humanization:
   Does it sound naturally written by a real person?
   Assess phrasing, flow, tone, conciseness, and lack of robotic wording.

2. Indian English authenticity:
   Is the wording natural and appropriate for Indian users?
   Consider Indian legal terminology, institutional context, and register.
   Do not reward grammatical errors, forced Hindi expressions,
   stereotypes, or unnecessary Indianisms.

3. Conversational naturalness:
   Would a real user plausibly ask this question in this context?

4. Experience-level alignment:
   Does the vocabulary and complexity fit the supplied experience level?
   Treat experience as context, not a stereotype.

5. Clarity:
   Is the question understandable, focused, and unambiguous?

6. Legal terminology and context:
   Is the terminology appropriate for an Indian legal question?
   Flag questionable legal assumptions. Do not confuse good wording
   with legal correctness.

7. Reconstruction plausibility:
   Does the preferred reconstruction fit the visible fragment and
   available candidates? Penalize unsupported additions.
   The original question is unavailable, so do not claim certainty
   about the original wording.

Important:
- Evaluate the preferred question exactly as written.
- Use the fragment, candidate questions, reasoning, and experience
  level as context.
- Distinguish linguistic quality from reconstruction fidelity.
- A written question cannot establish a spoken accent.
- Do not invent facts or legal conclusions.
- If a dimension cannot be assessed, use null and explain why.
- Return only valid JSON, with no Markdown fences.

Return this schema:
{
  "humanization_score": 0,
  "indian_english_authenticity_score": 0,
  "conversational_naturalness_score": 0,
  "experience_level_alignment_score": 0,
  "clarity_score": 0,
  "legal_terminology_context_score": 0,
  "reconstruction_plausibility_score": 0,
  "overall_quality_score": 0,
  "humanization_reason": "",
  "indian_english_reason": "",
  "experience_level_reason": "",
  "legal_risk_flags": [],
  "unsupported_additions": [],
  "suggested_revision": "",
  "original_question_verifiable": false,
  "evaluation_confidence": "Low"
}

Calculate overall_quality_score using these weights:
Humanization 25%, Indian English authenticity 20%,
conversational naturalness 15%, experience alignment 10%,
clarity 10%, legal terminology/context 10%,
reconstruction plausibility 10%.

If any score is null, set overall_quality_score to null.
All non-null scores must be integers from 0 to 100.

ROW CONTEXT:
{row_context}
"""


def safe_text(value: Any) -> str:
    """
    Safely convert possibly NaN pandas values to clean strings.
    """
    if pd.isna(value):
        return ""
    return str(value)


def extract_row_context(row: pd.Series | Dict[str, Any]) -> Dict[str, str]:
    """
    Extract relevant context columns from a row for LLM judge evaluation.
    """
    context_columns = [
        "Question_ID",
        "Experience",
        "Visible_Fragment",
        "Candidate_1",
        "Candidate_2",
        "Candidate_3",
        "Preferred_Reconstruction",
        "Confidence",
        "Reasoning",
        "Verification_Status",
        "Data_Provenance",
    ]

    return {
        col: safe_text(row[col])
        for col in context_columns
        if col in row
    }


def build_evaluation_prompt(row_context: Dict[str, str]) -> str:
    """
    Inject serialized row context JSON into the evaluation prompt template.
    Uses .replace() rather than .format() to avoid KeyError with JSON curly brackets.
    """
    context_json = json.dumps(row_context, ensure_ascii=False, indent=2)
    prompt = EVALUATION_PROMPT.replace("{row_context}", context_json)
    return prompt


def display_rubric() -> None:
    """
    Render the evaluation rubric and weights via Rich.
    """
    console.print(
        Panel.fit(
            "[bold bright_white]NYAYARAG LLM JUDGE — SCRIPT 03[/bold bright_white]\n"
            "[bright_cyan]Evaluation Rubric, Weighted Criteria & Schema Specification[/bright_cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    table = Table(
        title="Multi-Dimensional Evaluation Rubric (Total Weight: 100%)",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue",
    )
    table.add_column("No.", style="bold cyan", justify="center")
    table.add_column("Dimension", style="bold white")
    table.add_column("Weight", justify="center", style="bold yellow")
    table.add_column("Evaluation Core Criteria", style="dim", max_width=50)

    for idx, r in enumerate(RUBRIC_DEFINITIONS, start=1):
        table.add_row(str(idx), r["dimension"], r["weight"], r["description"])

    console.print(table)


def main() -> Dict[str, Any]:
    """
    Main execution workflow for Script 03.
    """
    display_rubric()
    sample_context = {
        "Question_ID": "RAW001",
        "Experience": "11–20 years",
        "Visible_Fragment": "Under what circumstances can delay in lodging ...",
        "Preferred_Reconstruction": "Under what circumstances can delay in lodging an FIR be considered adequately explained?",
    }
    sample_prompt = build_evaluation_prompt(sample_context)
    console.print(f"[dim]Sample prompt generated successfully ({len(sample_prompt)} characters).[/dim]")
    return {
        "weights": WEIGHTS,
        "rubric": RUBRIC_DEFINITIONS,
        "prompt_template": EVALUATION_PROMPT,
    }


if __name__ == "__main__":
    main()
