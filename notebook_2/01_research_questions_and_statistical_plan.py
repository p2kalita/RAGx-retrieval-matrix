"""
NYAYaRAG – Model Evaluation and Statistical Analysis
Notebook 2

Research Title:
A Retrieval-Augmented Generation Framework for Legal Question Answering Using Indian Supreme Court Judgments

---

Purpose:
This notebook evaluates the retrieval and question answering performance of the proposed NYAYaRAG framework. It implements a rigorous experimental methodology for comparing multiple retrieval approaches over Indian Supreme Court judgments, performs statistical hypothesis testing to determine significant differences between retrieval models, and evaluates citation-grounded answer generation using a Large Language Model.

The notebook follows reproducible research practices by documenting each experimental step, validating intermediate results, and reporting all evaluation metrics together with appropriate statistical analyses.

---

Objectives:
The objectives of this notebook are to:
- Justify the evaluation sample size using statistical power analysis.
- Prepare the evaluation dataset comprising manually curated legal questions.
- Load and initialise all retrieval models developed in Notebook 1.
- Evaluate retrieval performance using multiple information retrieval metrics.
- Compare retrieval models using appropriate non-parametric statistical tests.
- Generate citation-grounded legal answers using the best-performing retrieval model together with a Large Language Model.
- Evaluate the framework's behaviour on both answerable and unanswerable legal queries.
- Present graphical summaries, statistical findings, hypothesis decisions, and research question-wise conclusions.

---

Research Questions:
RQ1. Do semantic retrieval methodologies significantly outperform traditional lexical retrieval methodologies for legal information retrieval over Indian Supreme Court judgments for the year 2024?
RQ2. Which retrieval model provides the best retrieval performance for legal question answering over Indian Supreme Court judgments for the year 2024?
RQ3. How accurately does the proposed NYAYaRAG framework generate evidence-grounded legal answers based on retrieved Supreme Court judgments for the year 2024?
RQ4. Can the proposed NYAYaRAG framework reliably detect insufficient retrieval evidence and abstain from answering, thereby reducing hallucinated legal responses?

---

Experimental Workflow:
The notebook is organised into the following sections:

Part A — Sample Size Justification
- Statistical power analysis
- Research hypotheses
- Selection of statistical tests
- Required evaluation sample size

Part B — Evaluation Dataset Preparation
- Load evaluation questions
- Validate dataset integrity
- Assign relevance labels
- Categorise answerable and unanswerable questions

Part C — Retrieval Model Loading and Initialisation
- Load ChromaDB
- Initialise TF-IDF
- Initialise BM25
- Load dense embedding models
- Initialise Hybrid retrieval

Part D — Retrieval Evaluation
Manual implementation followed by validation of:
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Hit Rate@k

Part E — Statistical Analysis
- Friedman Test
- Post-hoc Wilcoxon Signed-Rank Tests
- Effect Size Calculation
- Multiple Comparison Correction
- Hypothesis Testing

Part F — NYAYaRAG Answer Generation
- Retrieve supporting judgments
- Generate citation-grounded answers using the Docker-hosted Large Language Model
- Evaluate generated responses

Part G — Results and Conclusions
- Tables
- Graphs
- Statistical summaries
- Research question-wise conclusions
- Final observations

---

Reproducibility:
All experiments reported in this notebook are fully reproducible. Wherever applicable, evaluation metrics are first implemented manually to illustrate their underlying mathematical formulation and are subsequently validated using standard Python libraries.

Random seeds are fixed wherever stochastic procedures are involved to ensure consistent experimental results.

---

Section 1 — Sample Size Justification

1.1 Introduction:
An adequate sample size is essential to ensure that the statistical analyses conducted in this study have sufficient power to detect meaningful differences in retrieval and answer-generation performance. A sample that is too small may fail to identify genuine performance differences, whereas an unnecessarily large sample may increase annotation and computational effort without providing proportionate analytical benefit.

In this study, the unit of analysis is the evaluation question. Each question is processed using the same retrieval models and, where applicable, the same answer-generation framework. This repeated-measures design allows model performance to be compared on identical legal queries, thereby reducing variability arising from differences in question difficulty.

The required sample size is assessed separately for each research question because the statistical test, study design, outcome measure, and effect-size assumption differ across the evaluation tasks. The final evaluation sample size is determined by the largest minimum sample-size requirement obtained across RQ1–RQ4, ensuring that all planned statistical analyses are adequately powered.

The sample-size analysis is based on the following statistical criteria:
- Significance level: alpha = 0.05
- Statistical power: 1 - beta = 0.80
- Two-sided hypothesis testing
- Medium effect size assumptions where pilot estimates are unavailable.

The planned evaluation dataset contains 50 manually curated legal questions, including both answerable and unanswerable queries. The adequacy of this sample will be assessed against the minimum sample-size requirement calculated for each research question.

---

1.2 Research Questions and Statistical Analysis Plan:
The proposed NYAYaRAG framework is evaluated through four research questions addressing three complementary components of retrieval-augmented legal question answering: retrieval performance, evidence-grounded answer generation, and safe abstention. Each research question is associated with an appropriate statistical procedure and corresponding evaluation metrics selected according to the study design, outcome variable, and measurement scale.

RQ1 evaluates retrieval performance at the retrieval-paradigm level by comparing semantic retrieval models with traditional lexical retrieval models. Since each benchmark legal research question is evaluated under both retrieval paradigms, the observations are paired. Accordingly, a one-sided Wilcoxon signed-rank test is employed to determine whether semantic retrieval significantly outperforms lexical retrieval. Retrieval performance is evaluated using Recall@k, Mean Reciprocal Rank (MRR), nDCG@k, Hit Rate@k, paired lexical-semantic score differences, p-values, and rank-biserial correlation as the effect-size measure.

RQ2 compares the retrieval performance of the six individual retrieval methodologies (TF-IDF, BM25, E5, BGE, InLegalBERT, and Hybrid Retrieval). As each benchmark question is evaluated using all six retrieval models, the study follows a repeated-measures design. Consequently, the Friedman test is used to detect overall differences among the retrieval models. When the Friedman test is statistically significant, Wilcoxon signed-rank post-hoc comparisons with Holm adjustment are performed to identify the specific retrieval models that differ significantly. Retrieval performance is assessed using Recall@k, MRR, nDCG@k, Hit Rate@k, Friedman mean ranks, Holm-adjusted p-values, and appropriate effect sizes.

RQ3 evaluates the quality of evidence-grounded answers generated by NYAYaRAG. Each benchmark legal research question produces a single composite answer-quality score derived from predefined evaluation criteria. The observed answer-quality scores are first summarised descriptively and are then compared against a predefined minimum acceptable benchmark using a one-sample Wilcoxon signed-rank test. The principal evaluation metrics include evidence-groundedness, legal correctness, citation accuracy, answer relevance, completeness, and the composite answer-quality score.

RQ4 evaluates the ability of NYAYaRAG to correctly abstain from answering legal research questions when the retrieved judgments do not provide sufficient supporting evidence. Abstention performance is summarised using correct-abstention rate, abstention precision, abstention recall, F1-score, false-answer rate, false-abstention rate, and hallucination rate. The observed correct-abstention rate is subsequently evaluated using an exact one-sided binomial test to determine whether it significantly exceeds a predefined minimum acceptable abstention benchmark.
"""

from pathlib import Path
import sys
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize Rich Console
console = Console(force_terminal=True, legacy_windows=False)

# Define project root
PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")


def build_statistical_plan() -> pd.DataFrame:
    """
    Builds and returns the statistical analysis plan DataFrame defining
    research questions, hypotheses, statistical analyses, and evaluation metrics.
    """
    statistical_plan = pd.DataFrame({
        "RQ": [
            "RQ1",
            "RQ2",
            "RQ3",
            "RQ4"
        ],

        "Research Question": [
            (
                "Do semantic retrieval models significantly outperform traditional "
                "lexical retrieval models for legal information retrieval over "
                "Indian Supreme Court judgments for the year 2024?"
            ),
            (
                "Which retrieval model provides the best retrieval performance for "
                "legal question answering over Indian Supreme Court judgments for "
                "the year 2024?"
            ),
            (
                "How accurately does the proposed NYAYaRAG framework generate "
                "evidence-grounded legal answers based on retrieved Supreme Court "
                "judgments for the year 2024?"
            ),
            (
                "Can the proposed NYAYaRAG framework reliably detect insufficient "
                "retrieval evidence and correctly abstain from answering unanswerable "
                "legal research questions, thereby reducing hallucinated legal responses?"
            )
        ],

        "Null Hypothesis (H₀)": [
            (
                "Semantic retrieval models do not provide significantly higher "
                "retrieval-performance scores than traditional lexical retrieval models."
            ),
            (
                "There is no statistically significant difference in retrieval "
                "performance among TF-IDF, BM25, E5, BGE, InLegalBERT, and "
                "Hybrid Retrieval."
            ),
            (
                "The evidence-grounded answer-quality scores generated by NYAYaRAG "
                "do not significantly exceed the predefined minimum acceptable benchmark."
            ),
            (
                "The true correct-abstention rate of NYAYaRAG does not exceed the "
                "predefined minimum acceptable abstention benchmark."
            )
        ],

        "Alternative Hypothesis (H₁)": [
            (
                "Semantic retrieval models provide significantly higher "
                "retrieval-performance scores than traditional lexical retrieval models."
            ),
            (
                "At least one of the six retrieval models differs significantly "
                "in retrieval performance from the others."
            ),
            (
                "The evidence-grounded answer-quality scores generated by NYAYaRAG "
                "significantly exceed the predefined minimum acceptable benchmark."
            ),
            (
                "The true correct-abstention rate of NYAYaRAG exceeds the "
                "predefined minimum acceptable abstention benchmark."
            )
        ],

        "Statistical Analysis": [
            (
                "For each benchmark question, an aggregate lexical score is calculated "
                "from TF-IDF and BM25, and an aggregate semantic score is calculated "
                "from E5, BGE, and InLegalBERT. The paired scores are compared using "
                "a one-sided Wilcoxon signed-rank test. Rank-biserial correlation is "
                "reported as the effect-size measure."
            ),
            (
                "The six individual retrieval models are compared using the Friedman "
                "test. If the overall result is statistically significant, pairwise "
                "Wilcoxon signed-rank post-hoc comparisons are conducted with Holm "
                "adjustment. Friedman mean ranks and appropriate effect sizes are reported."
            ),
            (
                "Descriptive answer-quality analysis and a one-sample Wilcoxon "
                "signed-rank test are used to determine whether the composite "
                "answer-quality scores significantly exceed the predefined minimum "
                "acceptable answer-quality benchmark."
            ),
            (
                "Confusion-matrix analysis is used to evaluate abstention performance. "
                "An exact one-sided binomial test is conducted to determine whether "
                "the observed correct-abstention rate significantly exceeds the "
                "predefined minimum acceptable abstention benchmark."
            )
        ],

        "Principal Evaluation Metrics": [
            (
                "Recall@k, MRR, nDCG@k, Hit Rate@k, paired lexical-semantic "
                "score differences, p-values, and rank-biserial effect size."
            ),
            (
                "Recall@k, MRR, nDCG@k, Hit Rate@k, Friedman mean ranks, "
                "Holm-adjusted p-values, and effect sizes."
            ),
            (
                "Evidence-groundedness, legal correctness, citation accuracy, "
                "answer relevance, completeness, and composite answer-quality score."
            ),
            (
                "Correct-abstention rate, abstention precision, abstention recall, "
                "F1-score, false-answer rate, false-abstention rate, and "
                "hallucination rate."
            )
        ]
    })
    return statistical_plan


def get_styled_statistical_plan(statistical_plan: pd.DataFrame):
    """
    Applies custom styling to the statistical analysis plan DataFrame
    matching the presentation specifications in Notebook 2.
    """
    styled_plan = (
        statistical_plan.style
        .hide(axis="index")
        .set_caption(
            "Table 1. Research Questions, Hypotheses, Statistical Analyses, "
            "and Evaluation Metrics"
        )
        .set_properties(**{
            "text-align": "left",
            "vertical-align": "top",
            "white-space": "normal",
            "font-size": "12px",
            "padding": "8px"
        })
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("font-size", "15px"),
                    ("font-weight", "bold"),
                    ("text-align", "left"),
                    ("caption-side", "top"),
                    ("margin-bottom", "10px")
                ]
            },
            {
                "selector": "table",
                "props": [
                    ("width", "100%"),
                    ("table-layout", "fixed"),
                    ("border-collapse", "collapse")
                ]
            },
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                    ("vertical-align", "top"),
                    ("font-weight", "bold"),
                    ("white-space", "normal"),
                    ("padding", "8px"),
                    ("border", "1px solid #cccccc")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #cccccc"),
                    ("word-wrap", "break-word"),
                    ("overflow-wrap", "break-word"),
                    ("white-space", "normal"),
                    ("padding", "8px")
                ]
            },
            {
                "selector": "th:nth-child(1)",
                "props": [("width", "5%")]
            },
            {
                "selector": "th:nth-child(2)",
                "props": [("width", "20%")]
            },
            {
                "selector": "th:nth-child(3)",
                "props": [("width", "18%")]
            },
            {
                "selector": "th:nth-child(4)",
                "props": [("width", "18%")]
            },
            {
                "selector": "th:nth-child(5)",
                "props": [("width", "21%")]
            },
            {
                "selector": "th:nth-child(6)",
                "props": [("width", "18%")]
            }
        ])
    )
    return styled_plan


def create_rich_statistical_plan_table(statistical_plan: pd.DataFrame) -> Table:
    """
    Constructs a styled Rich Table for Table 1: Research Questions, Hypotheses,
    Statistical Analyses, and Evaluation Metrics.
    """
    table = Table(
        title="Table 1. Research Questions, Hypotheses, Statistical Analyses, and Evaluation Metrics",
        box=box.ROUNDED,
        title_style="bold bright_white",
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("RQ", justify="center", style="bold cyan", no_wrap=True)
    table.add_column("Research Question", style="white", ratio=3)
    table.add_column("Null Hypothesis (H₀)", style="red", ratio=2)
    table.add_column("Alternative Hypothesis (H₁)", style="green", ratio=2)
    table.add_column("Statistical Analysis", style="yellow", ratio=3)
    table.add_column("Principal Evaluation Metrics", style="bright_blue", ratio=2)

    for _, row in statistical_plan.iterrows():
        table.add_row(
            str(row["RQ"]),
            str(row["Research Question"]),
            str(row["Null Hypothesis (H₀)"]),
            str(row["Alternative Hypothesis (H₁)"]),
            str(row["Statistical Analysis"]),
            str(row["Principal Evaluation Metrics"])
        )

    return table


def display_rich_statistical_plan(
    statistical_plan: pd.DataFrame,
    console_instance: Console = None
) -> None:
    """
    Renders and prints the statistical analysis plan using the Rich library.
    """
    con = console_instance or console
    rich_table = create_rich_statistical_plan_table(statistical_plan)
    con.print(rich_table)


def display_statistical_plan(
    statistical_plan: pd.DataFrame,
    use_rich: bool = True
) -> None:
    """
    Displays the styled statistical analysis plan table using Rich for terminal output,
    with support for pandas Styler in Jupyter/IPython environments.
    """
    pd.set_option("display.max_colwidth", None)

    if use_rich:
        display_rich_statistical_plan(statistical_plan)
        return

    try:
        from IPython.display import display
        styled_plan = get_styled_statistical_plan(statistical_plan)
        display(styled_plan)
    except (ImportError, Exception):
        print("=" * 80)
        print("Table 1. Research Questions, Hypotheses, Statistical Analyses, and Evaluation Metrics")
        print("=" * 80)
        for _, row in statistical_plan.iterrows():
            print(f"[{row['RQ']}]")
            print(f"  Research Question: {row['Research Question']}")
            print(f"  Null Hypothesis (H0): {row['Null Hypothesis (H₀)']}")
            print(f"  Alternative Hypothesis (H1): {row['Alternative Hypothesis (H₁)']}")
            print(f"  Statistical Analysis: {row['Statistical Analysis']}")
            print(f"  Principal Metrics: {row['Principal Evaluation Metrics']}")
            print("-" * 80)


def main():
    """
    Main execution function for building and presenting the research questions
    and statistical analysis plan with Rich terminal formatting.
    """
    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Model Evaluation and Statistical Analysis[/bold cyan]\n"
        "[italic bright_white]Notebook 2 — Section 1.2: Research Questions & Statistical Plan[/italic bright_white]",
        border_style="bright_blue",
        title="[bold green]Research Framework[/bold green]"
    ))

    console.print("[bold yellow]Building NYAYaRAG Statistical Analysis Plan (RQ1 - RQ4)...[/bold yellow]\n")
    plan_df = build_statistical_plan()
    display_rich_statistical_plan(plan_df)
    console.print("\n[bold green]✔ Statistical Analysis Plan constructed and rendered successfully.[/bold green]\n")


if __name__ == "__main__":
    main()

