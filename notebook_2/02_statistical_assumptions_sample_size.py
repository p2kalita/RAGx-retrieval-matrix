"""
Section 1 — Sample Size Justification

1.3 Sample Size Determination

An appropriate sample size is essential to ensure that the statistical analyses performed in this study have sufficient power to detect meaningful differences in retrieval performance and answer-generation quality. An inadequate sample size increases the probability of Type II errors, whereas an excessively large sample may result in unnecessary computational effort and manual annotation without providing proportionate analytical benefit.

The required sample size depends on the statistical test employed, the expected effect size, the desired significance level, and the statistical power. Since the four research questions investigate different aspects of retrieval performance, answer generation, and abstention, separate prospective sample-size calculations are performed for each research question.

The sample-size calculations presented in this section are based on the following statistical assumptions:

- Significance level (alpha): 0.05
- Statistical power (1 - beta): 0.80
- Confidence level: 95%
- Two-sided hypothesis tests, where applicable.
- Conservative prospective effect-size assumptions appropriate to the statistical analysis used for each research question, in the absence of pilot-derived estimates.

Because different statistical procedures are used across the four research questions, the minimum required sample size is estimated separately for each analysis. The final benchmark size is then selected as the largest minimum sample-size requirement obtained across RQ1–RQ4, ensuring adequate statistical power for all planned analyses while remaining practically feasible for detailed manual evaluation.
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


def build_statistical_assumptions_table() -> pd.DataFrame:
    """
    Creates and returns Table 2 containing the statistical assumptions
    and methodological justifications used for sample size determination.
    """
    table_2 = pd.DataFrame({
        "Parameter": [
            "Significance Level (α)",
            "Statistical Power (1 − β)",
            "Direction of Statistical Tests",
            "Effect-Size Assumption",
            "Study Design",
            "Number of Retrieval Models",
            "Final Evaluation Sample"
        ],

        "Value": [
            "0.05",
            "0.80",
            "Two-sided, unless otherwise justified",
            (
                "Conservative prospective effect-size assumption "
                "appropriate to each statistical analysis"
            ),
            "Repeated-measures and single-sample evaluation designs",
            "6",
            "To be determined through prospective sample-size analysis"
        ],

        "Justification": [
            (
                "A significance level of 0.05 permits a 5% probability of rejecting "
                "the null hypothesis when it is true."
            ),
            (
                "A statistical power of 0.80 provides an 80% probability of detecting "
                "an effect of the assumed magnitude when that effect exists."
            ),
            (
                "Two-sided tests are adopted to avoid assuming the direction of an "
                "effect before observing the evaluation results. Directional tests "
                "will be used only where a clear methodological justification exists."
            ),
            (
                "Where pilot-derived estimates are unavailable, a conservative "
                "prospective effect-size assumption is selected using the effect-size "
                "measure appropriate to each statistical analysis. Kendall's "
                "coefficient of concordance (W) is used for the Friedman test. "
                "For the answer-generation and abstention analyses, the effect-size "
                "measure will be selected according to the statistical methodology "
                "appropriate to those evaluation designs."
            ),
            (
                "RQ1 and RQ2 use a repeated-measures design because the same legal "
                "research questions are evaluated using every retrieval methodology. "
                "RQ3 and RQ4 use question-level, single-system answer-generation and "
                "abstention outcomes."
            ),
            (
                "The retrieval methodologies are TF-IDF, BM25, E5, BGE, "
                "InLegalBERT, and Hybrid Retrieval."
            ),
            (
                "The final number of benchmark legal research questions will be "
                "selected after calculating the minimum sample-size requirements for "
                "the analyses corresponding to RQ1-RQ4. The largest minimum "
                "requirement, together with the need for adequate legal-domain "
                "coverage and question diversity, will guide the final evaluation "
                "sample."
            )
        ]
    })
    return table_2


def validate_statistical_assumptions(table_2: pd.DataFrame) -> None:
    """
    Executes structural, completeness, and value validation checks on Table 2.
    """
    expected_columns = [
        "Parameter",
        "Value",
        "Justification"
    ]

    assert table_2.shape == (7, 3), (
        f"Expected Table 2 to contain 7 rows and 3 columns, "
        f"but found {table_2.shape}."
    )

    assert table_2.columns.tolist() == expected_columns, (
        "The Table 2 column names do not match the expected structure."
    )

    assert table_2.isna().sum().sum() == 0, (
        "Table 2 contains one or more missing values."
    )

    assert table_2.astype(str).apply(
        lambda column: column.str.strip().ne("").all()
    ).all(), (
        "Table 2 contains one or more blank cells."
    )

    assert table_2.loc[
        table_2["Parameter"] == "Significance Level (α)", "Value"
    ].iloc[0] == "0.05", (
        "The significance level must be 0.05."
    )

    assert table_2.loc[
        table_2["Parameter"] == "Statistical Power (1 − β)", "Value"
    ].iloc[0] == "0.80", (
        "The target statistical power must be 0.80."
    )

    assert table_2.loc[
        table_2["Parameter"] == "Number of Retrieval Models", "Value"
    ].iloc[0] == "6", (
        "The number of retrieval methodologies must be 6."
    )

    assert "50" not in table_2.loc[
        table_2["Parameter"] == "Final Evaluation Sample", "Value"
    ].iloc[0], (
        "The final evaluation sample should not be fixed before "
        "completing the prospective sample-size analyses."
    )


def get_styled_assumptions_table(table_2: pd.DataFrame):
    """
    Applies custom styling to the assumptions table.
    """
    styled_table = (
        table_2.style
        .set_caption(
            "Table 2. Statistical Assumptions Used for Sample Size Determination"
        )
        .set_properties(**{
            "text-align": "left",
            "vertical-align": "top",
            "white-space": "normal"
        })
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("font-size", "14px"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("margin-bottom", "10px")
                ]
            },
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                    ("font-weight", "bold"),
                    ("white-space", "normal")
                ]
            }
        ])
    )
    return styled_table


def create_rich_assumptions_table(table_2: pd.DataFrame) -> Table:
    """
    Constructs a styled Rich Table for Table 2: Statistical Assumptions Used for
    Sample Size Determination.
    """
    table = Table(
        title="Table 2. Statistical Assumptions Used for Sample Size Determination",
        box=box.ROUNDED,
        title_style="bold bright_white",
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("Parameter", style="bold cyan", width=28)
    table.add_column("Value", style="bold green", width=25)
    table.add_column("Justification", style="white")

    for _, row in table_2.iterrows():
        table.add_row(
            str(row["Parameter"]),
            str(row["Value"]),
            str(row["Justification"])
        )

    return table


def display_rich_assumptions(
    table_2: pd.DataFrame,
    console_instance: Console = None
) -> None:
    """
    Displays Table 2 assumptions using the Rich library.
    """
    con = console_instance or console
    rich_table = create_rich_assumptions_table(table_2)
    con.print(rich_table)


def display_statistical_assumptions(
    table_2: pd.DataFrame,
    use_rich: bool = True
) -> None:
    """
    Displays the styled assumptions table, with Rich terminal rendering or
    pandas Styler in Jupyter/IPython.
    """
    pd.set_option("display.max_colwidth", None)

    if use_rich:
        display_rich_assumptions(table_2)
        return

    try:
        from IPython.display import display
        display(get_styled_assumptions_table(table_2))
    except (ImportError, Exception):
        print("=" * 80)
        print("Table 2. Statistical Assumptions Used for Sample Size Determination")
        print("=" * 80)
        for _, row in table_2.iterrows():
            print(f"[{row['Parameter']}] = {row['Value']}")
            print(f"  Justification: {row['Justification']}")
            print("-" * 80)


def print_validation_summary(table_2: pd.DataFrame) -> None:
    """
    Prints validation summary information to standard output using Rich.
    """
    console.print(Panel(
        f"[bold green]✔ Table 2 validation completed successfully.[/bold green]\n"
        f"• Number of assumptions documented : [bold cyan]{table_2.shape[0]}[/bold cyan]\n"
        f"• Number of table columns          : [bold cyan]{table_2.shape[1]}[/bold cyan]\n"
        f"[italic white]Final evaluation sample will be determined after completing "
        f"the prospective sample-size analyses for RQ1-RQ4.[/italic white]",
        border_style="green",
        title="[bold bright_white]Validation Summary[/bold bright_white]"
    ))


def main():
    """
    Main function to construct, validate, and display Table 2 assumptions with Rich formatting.
    """
    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Model Evaluation and Statistical Analysis[/bold cyan]\n"
        "[italic bright_white]Notebook 2 — Section 1.3: Statistical Assumptions for Sample Size Determination[/italic bright_white]",
        border_style="bright_blue",
        title="[bold green]Statistical Design[/bold green]"
    ))

    console.print("[bold yellow]Constructing Table 2: Statistical Assumptions Used for Sample Size Determination...[/bold yellow]\n")
    assumptions_df = build_statistical_assumptions_table()
    validate_statistical_assumptions(assumptions_df)
    display_statistical_assumptions(assumptions_df)
    print_validation_summary(assumptions_df)


if __name__ == "__main__":
    main()

