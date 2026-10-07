"""
1.4 Sample Size Determination for Retrieval Evaluation (RQ1 and RQ2)

1.4 Sample Size Determination for RQ1

1.4.1 Statistical Basis:
Research Question 1 evaluates whether semantic retrieval models significantly outperform traditional lexical retrieval models for legal information retrieval over Indian Supreme Court judgments.

The retrieval paradigms considered in this comparison comprise:
- Lexical retrieval models: TF-IDF and BM25
- Semantic retrieval models: E5, BGE, and InLegalBERT

Each benchmark legal research question is evaluated using both retrieval paradigms. Consequently, every question produces a paired lexical retrieval score and semantic retrieval score. Because the observations originate from the same benchmark questions, they are related rather than independent, resulting in a paired repeated-measures experimental design.

The primary statistical test selected for RQ1 is the Wilcoxon signed-rank test. This non-parametric test is appropriate for comparing two related samples when the distribution of paired differences cannot be assumed to be normal. Retrieval-performance measures such as Recall@k, Mean Reciprocal Rank (MRR), and nDCG@k are frequently bounded, skewed, and may contain tied observations. The Wilcoxon signed-rank test therefore provides a robust statistical procedure for evaluating whether semantic retrieval consistently produces higher retrieval-performance scores than lexical retrieval.

Because the research hypothesis is directional and specifically proposes that semantic retrieval outperforms lexical retrieval, a one-sided Wilcoxon signed-rank test is adopted.

The hypotheses for RQ1 are defined as follows:

Research Question 1:
- Null Hypothesis (H₀₁): Semantic retrieval models do not provide significantly higher retrieval performance than traditional lexical retrieval models for legal information retrieval over Indian Supreme Court judgments.
- Alternative Hypothesis (H₁₁): Semantic retrieval models provide significantly higher retrieval performance than traditional lexical retrieval models for legal information retrieval over Indian Supreme Court judgments.

Unlike the Friedman test, the Wilcoxon signed-rank test is based on paired observations. Consequently, the required sample size is determined using a prospective power analysis for a paired non-parametric comparison.

The prospective sample-size analysis assumes:
- Significance level (alpha): 0.05
- Statistical power (1 - beta): 0.80
- One-sided hypothesis test
- Conservative prospective effect-size assumption in the absence of pilot-derived estimates.

The required sample size is identified as the smallest number of benchmark legal research questions that achieves the target statistical power under these assumptions. This minimum requirement will subsequently be compared with the corresponding requirements for RQ2, RQ3, and RQ4, and the largest minimum sample size will be adopted as the final benchmark evaluation sample.

---

Table 1. Statistical Design and Power Analysis Parameters for Retrieval Evaluation (RQ1):

| Parameter | Specification |
|:---|:---|
| Applicable Research Question | RQ1 |
| Retrieval Paradigms Compared | Lexical (TF-IDF, BM25) vs Semantic (E5, BGE, InLegalBERT) |
| Experimental Unit | Individual benchmark legal research question |
| Study Design | Paired repeated-measures design |
| Primary Statistical Test | One-sided Wilcoxon signed-rank test |
| Significance Level (alpha) | 0.05 |
| Target Statistical Power (1 - beta) | 0.80 |
| Effect Size Measure | Rank-biserial correlation (or equivalent Wilcoxon effect-size measure) |
| Effect Size Assumption | Conservative prospective effect-size assumption |
| Planned Evaluation Sample | To be determined through prospective sample-size analysis |

Because this analysis is conducted before the retrieval evaluation, the assumed effect size is used solely for prospective sample-size planning. Following completion of the retrieval evaluation, the Wilcoxon signed-rank test statistic, the corresponding p-value, and the rank-biserial correlation effect size will be reported based on the observed retrieval performance.

The minimum sample-size estimate obtained from this analysis applies only to RQ1, which evaluates the retrieval performance of semantic and lexical retrieval paradigms. The final benchmark evaluation sample will be selected after comparing the minimum sample-size requirements across RQ1–RQ4, with the largest requirement adopted for the study.

---

Observations:
The prospective power analysis for RQ1 was performed using a paired-samples normal approximation based on a one-sided comparison between semantic and lexical retrieval performance. The analysis assumed a significance level of alpha = 0.05, a target statistical power of 0.80, and a conservative standardized paired effect size (d_z = 0.50).

The estimated statistical power increased monotonically with the number of benchmark legal research questions. The minimum sample size required to achieve the target statistical power was 27 evaluation questions, at which the estimated power reached 0.812.

For the planned benchmark dataset of 50 legal research questions, the estimated statistical power was 0.967, substantially exceeding the target power of 0.80. This indicates that the planned evaluation dataset is adequately powered to detect the assumed effect size for the paired comparison between semantic and lexical retrieval performance.

Accordingly, the planned benchmark of 50 evaluation questions exceeds the minimum sample-size requirement for RQ1 and is therefore considered sufficient for the subsequent Wilcoxon signed-rank analysis.
"""

from pathlib import Path
import sys
import pandas as pd
from statsmodels.stats.power import TTestPower

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

# ------------------------------------------------------------
# Power-analysis parameters for RQ1
# ------------------------------------------------------------
ALPHA = 0.05
TARGET_POWER = 0.80

# RQ1 contains a directional hypothesis:
# semantic retrieval performance > lexical retrieval performance
TEST_ALTERNATIVE = "larger"

# Conservative medium standardized paired effect-size assumption
# d_z represents the mean paired difference divided by the
# standard deviation of the paired differences.
ASSUMED_EFFECT_SIZE_DZ = 0.50

# Candidate numbers of benchmark legal research questions
SAMPLE_SIZES = range(5, 101)

# Planned evaluation sample
PLANNED_SAMPLE_SIZE = 50


def calculate_rq1_power_curve(
    sample_sizes=SAMPLE_SIZES,
    effect_size=ASSUMED_EFFECT_SIZE_DZ,
    alpha=ALPHA,
    alternative=TEST_ALTERNATIVE,
    target_power=TARGET_POWER
) -> pd.DataFrame:
    """
    Calculates estimated statistical power across candidate sample sizes
    for the paired lexical-semantic retrieval comparison.
    """
    power_model = TTestPower()
    power_results = []

    for n_questions in sample_sizes:
        estimated_power = power_model.power(
            effect_size=effect_size,
            nobs=n_questions,
            alpha=alpha,
            alternative=alternative
        )

        power_results.append({
            "Evaluation Questions": n_questions,
            "Assumed Effect Size (d_z)": effect_size,
            "Estimated Power": estimated_power,
            "Target Power Achieved": estimated_power >= target_power
        })

    return pd.DataFrame(power_results)


def solve_rq1_sample_size(
    power_df: pd.DataFrame,
    effect_size=ASSUMED_EFFECT_SIZE_DZ,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    alternative=TEST_ALTERNATIVE
):
    """
    Determines the minimum required sample size both iteratively from the power curve
    and analytically via TTestPower.solve_power.
    """
    adequate_sample_sizes = power_df[
        power_df["Estimated Power"] >= target_power
    ]

    if adequate_sample_sizes.empty:
        minimum_required_n = None
    else:
        minimum_required_n = int(
            adequate_sample_sizes.iloc[0]["Evaluation Questions"]
        )

    power_model = TTestPower()
    analytical_required_n = power_model.solve_power(
        effect_size=effect_size,
        nobs=None,
        alpha=alpha,
        power=target_power,
        alternative=alternative
    )
    rounded_required_n = int(-(-analytical_required_n // 1))

    return minimum_required_n, analytical_required_n, rounded_required_n


def validate_rq1_results(
    power_df: pd.DataFrame,
    minimum_required_n: int,
    rounded_required_n: int,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    effect_size=ASSUMED_EFFECT_SIZE_DZ,
    alternative=TEST_ALTERNATIVE,
    planned_size=PLANNED_SAMPLE_SIZE
) -> None:
    """
    Runs all 9 validation checks verifying data integrity and power calculations.
    """
    assert 0 < alpha < 1, (
        "Alpha must be between 0 and 1."
    )

    assert 0 < target_power < 1, (
        "Target power must be between 0 and 1."
    )

    assert effect_size > 0, (
        "The assumed effect size must be greater than zero."
    )

    assert alternative in ["two-sided", "larger", "smaller"], (
        "The specified test direction is invalid."
    )

    assert planned_size in power_df[
        "Evaluation Questions"
    ].values, (
        "The planned sample size is not included in the candidate range."
    )

    assert power_df["Estimated Power"].between(0, 1).all(), (
        "All estimated power values must be between 0 and 1."
    )

    assert power_df["Estimated Power"].is_monotonic_increasing, (
        "Estimated power should increase as sample size increases."
    )

    assert power_df.isna().sum().sum() == 0, (
        "The results table contains missing values."
    )

    assert minimum_required_n == rounded_required_n, (
        "The iterative and analytical sample-size estimates do not agree."
    )


def build_rq1_display_table(
    power_df: pd.DataFrame,
    minimum_required_n: int,
    sample_sizes=SAMPLE_SIZES,
    planned_sample_size=PLANNED_SAMPLE_SIZE
) -> pd.DataFrame:
    """
    Extracts the most relevant candidate sample size rows around the minimum required
    and planned sample size.
    """
    if minimum_required_n is not None:
        lower_limit = max(
            min(sample_sizes),
            minimum_required_n - 3
        )
        upper_limit = minimum_required_n + 3

        nearby_rows = power_df[
            power_df["Evaluation Questions"].between(
                lower_limit,
                upper_limit
            )
        ]

        planned_row = power_df[
            power_df["Evaluation Questions"] == planned_sample_size
        ]

        display_table = pd.concat(
            [nearby_rows, planned_row],
            ignore_index=True
        ).drop_duplicates(
            subset=["Evaluation Questions"]
        )
    else:
        display_table = power_df.tail(10)

    return display_table


def create_rich_rq1_table(display_table: pd.DataFrame) -> Table:
    """
    Constructs a styled Rich Table for RQ1 prospective power analysis.
    """
    table = Table(
        title="RQ1 Prospective Power Analysis for Paired Lexical–Semantic Retrieval Comparison",
        box=box.ROUNDED,
        title_style="bold bright_white",
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("Evaluation Questions", justify="center", style="bold cyan")
    table.add_column("Assumed Effect Size (d_z)", justify="center", style="yellow")
    table.add_column("Estimated Power", justify="center", style="bold green")
    table.add_column("Target Power Achieved", justify="center", style="bold magenta")

    for _, row in display_table.iterrows():
        achieved = row["Target Power Achieved"]
        status_str = "[bold green]True[/bold green]" if achieved else "[bold red]False[/bold red]"
        table.add_row(
            str(int(row["Evaluation Questions"])),
            f"{row['Assumed Effect Size (d_z)']:.2f}",
            f"{row['Estimated Power']:.3f}",
            status_str
        )

    return table


def display_rich_rq1_table(
    display_table: pd.DataFrame,
    console_instance: Console = None
) -> None:
    """
    Displays the RQ1 power table using the Rich library.
    """
    con = console_instance or console
    rich_table = create_rich_rq1_table(display_table)
    con.print(rich_table)


def display_rq1_table(
    display_table: pd.DataFrame,
    use_rich: bool = True
) -> None:
    """
    Displays styled table using Rich for terminal output or pandas Styler in Jupyter/IPython.
    """
    if use_rich:
        display_rich_rq1_table(display_table)
        return

    try:
        from IPython.display import display
        styled = (
            display_table.style
            .hide(axis="index")
            .format({
                "Assumed Effect Size (d_z)": "{:.2f}",
                "Estimated Power": "{:.3f}"
            })
            .set_caption(
                "RQ1 Prospective Power Analysis for the "
                "Paired Lexical–Semantic Retrieval Comparison"
            )
            .set_properties(**{
                "text-align": "center",
                "vertical-align": "middle"
            })
        )
        display(styled)
    except (ImportError, Exception):
        print("\n" + "=" * 65)
        print("RQ1 Prospective Power Analysis Table (Selected Sample Sizes)")
        print("=" * 65)
        print(display_table.to_string(index=False))


def print_rq1_conclusion(
    minimum_required_n: int,
    analytical_required_n: float,
    planned_sample_power: float,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    effect_size=ASSUMED_EFFECT_SIZE_DZ,
    planned_size=PLANNED_SAMPLE_SIZE
) -> None:
    """
    Prints the final sample-size conclusion and methodological notes for RQ1 using Rich.
    """
    summary_text = (
        f"[bold cyan]Study design                       :[/bold cyan] Paired lexical-semantic comparison\n"
        f"[bold cyan]Planned final statistical test     :[/bold cyan] One-sided Wilcoxon signed-rank test\n"
        f"[bold cyan]Prospective planning approximation :[/bold cyan] Paired-samples normal approximation\n"
        f"[bold cyan]Significance level                 :[/bold cyan] {alpha:.2f}\n"
        f"[bold cyan]Target statistical power           :[/bold cyan] {target_power:.2f}\n"
        f"[bold cyan]Assumed standardized effect size   :[/bold cyan] d_z = {effect_size:.2f}\n"
        f"[bold cyan]Direction of hypothesis            :[/bold cyan] Semantic retrieval > lexical retrieval\n\n"
    )

    if minimum_required_n is not None:
        summary_text += (
            f"[bold green]• Minimum sample size achieving power ≥ {target_power:.2f}:[/bold green] [bold bright_white]{minimum_required_n} questions[/bold bright_white]\n"
            f"[bold green]• Unrounded analytical estimate      :[/bold green] {analytical_required_n:.3f}\n"
            f"[bold green]• Estimated power with {planned_size} questions      :[/bold green] [bold yellow]{planned_sample_power:.3f}[/bold yellow]\n\n"
        )
        if planned_size >= minimum_required_n:
            summary_text += (
                f"[bold green]Conclusion:[/bold green] The planned sample of [bold bright_white]{planned_size}[/bold bright_white] "
                f"benchmark questions [bold green]exceeds[/bold green] the estimated minimum sample-size requirement for RQ1.\n"
            )
        else:
            summary_text += (
                f"[bold red]Conclusion:[/bold red] The planned sample of [bold bright_white]{planned_size}[/bold bright_white] "
                f"benchmark questions does not meet the estimated minimum sample-size requirement for RQ1.\n"
            )
    else:
        summary_text += "[bold red]The target statistical power was not achieved within candidate range.[/bold red]\n"

    summary_text += (
        "\n[dim italic]Note: The prospective calculation uses a paired-samples normal approximation "
        "because a universally accepted closed-form power formula for the Wilcoxon signed-rank test is unavailable. "
        "The actual hypothesis test for RQ1 will remain the one-sided Wilcoxon signed-rank test.[/dim italic]"
    )

    console.print(Panel(
        summary_text,
        border_style="cyan",
        title="[bold bright_white]RQ1 Sample-Size Determination Summary[/bold bright_white]"
    ))


def main():
    """
    Executes prospective power analysis for RQ1 with Rich formatting.
    """
    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Model Evaluation and Statistical Analysis[/bold cyan]\n"
        "[italic bright_white]Notebook 2 — Section 1.4: Sample Size Determination for RQ1 (Wilcoxon Test)[/italic bright_white]",
        border_style="bright_blue",
        title="[bold green]Power Analysis — RQ1[/bold green]"
    ))

    console.print("[bold yellow]Performing RQ1 Prospective Sample-Size Analysis...[/bold yellow]\n")
    power_df = calculate_rq1_power_curve()
    min_n, analytical_n, rounded_n = solve_rq1_sample_size(power_df)
    
    planned_power = float(
        power_df.loc[
            power_df["Evaluation Questions"] == PLANNED_SAMPLE_SIZE,
            "Estimated Power"
        ].iloc[0]
    )

    validate_rq1_results(power_df, min_n, rounded_n)
    
    display_tbl = build_rq1_display_table(power_df, min_n)
    display_rq1_table(display_tbl)
    
    print_rq1_conclusion(min_n, analytical_n, planned_power)


if __name__ == "__main__":
    main()

