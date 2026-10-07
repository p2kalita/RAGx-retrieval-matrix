"""
Table 5. Final Statistical Tests and Sample-Size Decisions Across Research Questions

| RQ | Statistical Test | Minimum Required Sample Size | Final Sample Size Decided |
|:---:|:---|:---:|:---:|
| RQ1 | One-sided Wilcoxon signed-rank test | 27 answerable questions | 50 answerable questions |
| RQ2 | Friedman test with Wilcoxon signed-rank post-hoc comparisons and Holm correction | 13 answerable questions | 50 answerable questions |
| RQ3 | One-sample Wilcoxon signed-rank test | 27 answerable questions | 50 answerable questions |
| RQ4 | Exact one-sided binomial test using a 60% benchmark | 21 unanswerable questions | 25 unanswerable questions |

Note:
RQ1–RQ3 will be evaluated using the same set of 50 answerable benchmark legal research questions.
RQ4 will be evaluated separately using 25 unanswerable benchmark legal research questions.
The resulting final evaluation benchmark will therefore comprise 75 questions in total:
50 answerable and 25 unanswerable questions.
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


def build_final_sample_decisions_table() -> pd.DataFrame:
    """
    Constructs the final decision synthesis table across all research questions (RQ1-RQ4).
    """
    final_decisions = pd.DataFrame({
        "RQ": [
            "RQ1",
            "RQ2",
            "RQ3",
            "RQ4"
        ],
        "Statistical Test": [
            "One-sided Wilcoxon signed-rank test",
            "Friedman test with Wilcoxon signed-rank post-hoc comparisons and Holm correction",
            "One-sample Wilcoxon signed-rank test",
            "Exact one-sided binomial test using a 60% benchmark"
        ],
        "Minimum Required Sample Size": [
            "27 answerable questions",
            "13 answerable questions",
            "27 answerable questions",
            "21 unanswerable questions"
        ],
        "Final Sample Size Decided": [
            "50 answerable questions",
            "50 answerable questions",
            "50 answerable questions",
            "25 unanswerable questions"
        ]
    })
    return final_decisions


def validate_final_decisions(final_decisions: pd.DataFrame) -> None:
    """
    Validates structure, row counts, and values of the final decisions table.
    """
    assert final_decisions.shape == (4, 4), (
        f"Expected 4 rows and 4 columns, found {final_decisions.shape}"
    )

    expected_cols = [
        "RQ",
        "Statistical Test",
        "Minimum Required Sample Size",
        "Final Sample Size Decided"
    ]
    assert final_decisions.columns.tolist() == expected_cols, (
        "Column names do not match expected structure."
    )

    assert final_decisions["RQ"].tolist() == ["RQ1", "RQ2", "RQ3", "RQ4"], (
        "Research questions order mismatch."
    )

    assert final_decisions.isna().sum().sum() == 0, (
        "Final decisions table contains missing values."
    )


def create_rich_final_decisions_table(final_decisions: pd.DataFrame) -> Table:
    """
    Constructs a styled Rich Table for Table 5: Final Statistical Tests and Sample-Size Decisions.
    """
    table = Table(
        title="Table 5. Final Statistical Tests and Sample-Size Decisions Across Research Questions",
        box=box.ROUNDED,
        title_style="bold bright_white",
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("RQ", justify="center", style="bold cyan", width=6)
    table.add_column("Statistical Test", style="white")
    table.add_column("Minimum Required Sample Size", justify="center", style="yellow")
    table.add_column("Final Sample Size Decided", justify="center", style="bold green")

    for _, row in final_decisions.iterrows():
        table.add_row(
            str(row["RQ"]),
            str(row["Statistical Test"]),
            str(row["Minimum Required Sample Size"]),
            str(row["Final Sample Size Decided"])
        )

    return table


def display_rich_final_decisions(
    final_decisions: pd.DataFrame,
    console_instance: Console = None
) -> None:
    """
    Displays Table 5 using the Rich library.
    """
    con = console_instance or console
    rich_table = create_rich_final_decisions_table(final_decisions)
    con.print(rich_table)


def display_final_decisions(
    final_decisions: pd.DataFrame,
    use_rich: bool = True
) -> None:
    """
    Renders styled table using Rich for terminal output or pandas Styler in Jupyter/IPython.
    """
    if use_rich:
        display_rich_final_decisions(final_decisions)
        return

    try:
        from IPython.display import display
        styled = (
            final_decisions.style
            .hide(axis="index")
            .set_caption(
                "Table 5. Final Statistical Tests and Sample-Size Decisions Across Research Questions"
            )
            .set_properties(**{
                "text-align": "left",
                "vertical-align": "top",
                "padding": "6px"
            })
            .set_table_styles([
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "14px"),
                        ("font-weight", "bold"),
                        ("text-align", "left"),
                        ("margin-bottom", "10px")
                    ]
                },
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),
                        ("border", "1px solid #ccc"),
                        ("padding", "6px")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("border", "1px solid #ccc")
                    ]
                }
            ])
        )
        display(styled)
    except (ImportError, Exception):
        print("\n" + "=" * 80)
        print("Table 5. Final Statistical Tests and Sample-Size Decisions Across Research Questions")
        print("=" * 80)
        for _, row in final_decisions.iterrows():
            print(f"[{row['RQ']}] {row['Statistical Test']}")
            print(f"  Minimum Required : {row['Minimum Required Sample Size']}")
            print(f"  Final Decided    : {row['Final Sample Size Decided']}")
            print("-" * 80)


def print_final_summary_notes():
    """
    Prints comprehensive experimental design summary notes and allocation breakdown using Rich.
    """
    summary_text = (
        "[bold cyan]• Answerable Benchmark Questions   :[/bold cyan] [bold bright_white]50 legal research questions[/bold bright_white]\n"
        "  [yellow]- RQ1:[/yellow] Paired lexical vs. semantic retrieval evaluation (Wilcoxon Signed-Rank)\n"
        "  [yellow]- RQ2:[/yellow] Multi-system comparison across 6 retrieval methodologies (Friedman Test)\n"
        "  [yellow]- RQ3:[/yellow] Citation-grounded answer quality evaluation (One-Sample Wilcoxon)\n\n"
        "[bold cyan]• Unanswerable Benchmark Questions :[/bold cyan] [bold bright_white]25 legal research questions[/bold bright_white]\n"
        "  [yellow]- RQ4:[/yellow] Insufficient evidence detection & safe abstention evaluation (Binomial Test)\n\n"
        "[bold green]• Total Evaluation Dataset Benchmark:[/bold green] [bold bright_white]75 evaluation questions[/bold bright_white]\n\n"
        "[bold green]✔ All planned sample sizes exceed the respective minimum power requirements (power ≥ 0.80).[/bold green]\n"
        "[italic white]Dataset allocation ensures rigorous statistical power while maintaining manual review feasibility.[/italic white]"
    )

    console.print(Panel(
        summary_text,
        border_style="green",
        title="[bold bright_white]NYAYaRAG Evaluation Dataset Allocation Breakdown[/bold bright_white]"
    ))


def main():
    """
    Constructs, validates, and reports the final sample size decisions across all research questions with Rich formatting.
    """
    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Model Evaluation and Statistical Analysis[/bold cyan]\n"
        "[italic bright_white]Notebook 2 — Table 5: Final Statistical Tests & Sample-Size Decisions[/italic bright_white]",
        border_style="bright_blue",
        title="[bold green]Synthesis & Decisions[/bold green]"
    ))

    console.print("[bold yellow]Constructing Table 5: Final Statistical Tests and Sample-Size Decisions...[/bold yellow]\n")
    final_df = build_final_sample_decisions_table()
    validate_final_decisions(final_df)
    display_final_decisions(final_df)
    print_final_summary_notes()


if __name__ == "__main__":
    main()

