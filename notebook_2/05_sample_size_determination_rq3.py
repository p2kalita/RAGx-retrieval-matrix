"""
1.6 Sample Size Determination for NYAYaRAG Answer-Generation Evaluation (RQ3)

1.6.1 Statistical Basis:
Research Question 3 evaluates whether NYAYaRAG generates high-quality answers for legal research questions over Indian Supreme Court judgments.

Each benchmark legal research question is processed by NYAYaRAG to produce a single generated answer. The quality of each answer is evaluated using the predefined answer-evaluation rubric. Consequently, each evaluation question contributes one overall answer-quality score, resulting in a one-sample repeated evaluation design.

The primary statistical test selected for RQ3 is the one-sample Wilcoxon signed-rank test. This non-parametric test is appropriate for determining whether the median answer-quality score differs significantly from a predefined benchmark when the distribution of evaluation scores cannot be assumed to be normal.

The hypotheses for RQ3 are defined as follows:
- Null Hypothesis (H₀₃): The median NYAYaRAG answer-quality score is less than or equal to the predefined benchmark.
- Alternative Hypothesis (H₁₃): The median NYAYaRAG answer-quality score is greater than the predefined benchmark.

No universally accepted closed-form sample-size formula exists for the one-sample Wilcoxon signed-rank test. Therefore, the required sample size for RQ3 is estimated using a one-sample normal approximation, which provides a practical prospective power-analysis approach for planning the evaluation.

In the absence of pilot-derived estimates, the prospective power analysis assumes:
- Significance level (alpha): 0.05
- Target statistical power (1 - beta): 0.80
- Alternative hypothesis: One-sided
- Assumed standardized effect size (d): 0.50 (conservative prospective assumption)
- Prospective sample-size range: 5–100 benchmark legal research questions

The required sample size is identified as the smallest number of benchmark legal research questions for which the estimated statistical power reaches or exceeds 0.80. The resulting minimum sample size will subsequently be compared with the corresponding requirements obtained for RQ1, RQ2, and RQ4, with the largest minimum sample size adopted as the final benchmark evaluation dataset.

---

Table 3. Statistical Design and Power Analysis Parameters for NYAYaRAG Answer-Generation Evaluation (RQ3):

| Parameter | Value / Description |
|:---|:---|
| Applicable Research Question | RQ3 – Does NYAYaRAG generate answers whose quality exceeds a predefined acceptable benchmark? |
| System Evaluated | NYAYaRAG |
| Experimental Unit | Individual benchmark legal research question |
| Outcome Variable | Overall answer-quality score assigned using the predefined evaluation rubric |
| Study Design | One-sample evaluation design |
| Predefined Benchmark | Median answer-quality score of 3 on a five-point evaluation scale |
| Primary Statistical Test | One-sample Wilcoxon signed-rank test |
| Direction of Hypothesis | One-sided |
| Significance Level (alpha) | 0.05 |
| Target Statistical Power (1 - beta) | 0.80 |
| Effect-Size Measure | Standardized one-sample effect size (d) |
| Assumed Effect Size | d = 0.50 (conservative prospective assumption) |
| Sample-Size Estimation Method | One-sample normal approximation |
| Candidate Sample-Size Range | 5–100 benchmark legal research questions |
| Criterion for Minimum Sample Size | Smallest sample size achieving estimated statistical power >= 0.80 |
| Planned Evaluation Sample | 50 benchmark legal research questions |

The parameters presented in Table 3 define the statistical framework adopted for RQ3. The one-sample Wilcoxon signed-rank test will be used to determine whether the median NYAYaRAG answer-quality score is significantly greater than the predefined benchmark score of 3. Because no universally accepted closed-form sample-size formula exists for the one-sample Wilcoxon signed-rank test, a one-sample normal approximation is used for prospective power analysis. The resulting minimum sample-size estimate will be compared with the requirements obtained for RQ1, RQ2, and RQ4.

---

Observations:
The prospective power analysis for RQ3 was conducted using a one-sample normal approximation for the planned one-sample Wilcoxon signed-rank test. The analysis assumed a significance level of alpha = 0.05, a target statistical power of 1 - beta = 0.80, a one-sided alternative hypothesis, and a conservative standardized effect size of d = 0.50.

The estimated statistical power increased progressively as the number of benchmark legal research questions increased. The analytical calculation produced a required sample size of 26.138, which was rounded upward to the next whole number. Accordingly, the minimum sample size required to achieve the target power was 27 benchmark questions, at which the estimated statistical power was 0.812.

For the planned evaluation dataset of 50 benchmark legal research questions, the estimated statistical power was 0.967, which substantially exceeds the required power of 0.80. This indicates that the planned sample is sufficiently large to evaluate whether the median NYAYaRAG answer-quality score is significantly greater than the predefined benchmark.

Accordingly, the planned sample of 50 benchmark questions exceeds the minimum sample-size requirement for RQ3 and is considered adequate for the subsequent one-sample Wilcoxon signed-rank analysis.
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
# Power-analysis parameters for RQ3
# ------------------------------------------------------------
ALPHA = 0.05
TARGET_POWER = 0.80

# Conservative prospective standardized one-sample effect size
ASSUMED_EFFECT_SIZE = 0.50

# One-sided hypothesis:
# NYAYaRAG answer quality is greater than the benchmark
ALTERNATIVE = "larger"

# Candidate numbers of benchmark legal research questions
SAMPLE_SIZES = range(5, 101)

# Planned benchmark sample
PLANNED_SAMPLE_SIZE = 50


def calculate_rq3_power_curve(
    sample_sizes=SAMPLE_SIZES,
    effect_size=ASSUMED_EFFECT_SIZE,
    alpha=ALPHA,
    alternative=ALTERNATIVE,
    target_power=TARGET_POWER
) -> pd.DataFrame:
    """
    Calculates estimated statistical power across candidate sample sizes
    for the one-sample answer-quality evaluation.
    """
    power_analysis = TTestPower()
    power_results = []

    for n_questions in sample_sizes:
        estimated_power = power_analysis.power(
            effect_size=effect_size,
            nobs=n_questions,
            alpha=alpha,
            alternative=alternative
        )

        power_results.append({
            "Evaluation Questions": n_questions,
            "Assumed Effect Size": effect_size,
            "Estimated Power": estimated_power,
            "Target Power Achieved": estimated_power >= target_power
        })

    return pd.DataFrame(power_results)


def solve_rq3_sample_size(
    power_df: pd.DataFrame,
    effect_size=ASSUMED_EFFECT_SIZE,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    alternative=ALTERNATIVE
):
    """
    Determines minimum required sample size iteratively and confirms analytically.
    """
    adequate_sample_sizes = power_df[
        power_df["Estimated Power"] >= target_power
    ]

    if adequate_sample_sizes.empty:
        minimum_required_n_rq3 = None
    else:
        minimum_required_n_rq3 = int(
            adequate_sample_sizes.iloc[0]["Evaluation Questions"]
        )

    power_analysis = TTestPower()
    analytical_required_n_rq3 = power_analysis.solve_power(
        effect_size=effect_size,
        power=target_power,
        alpha=alpha,
        alternative=alternative
    )

    return minimum_required_n_rq3, analytical_required_n_rq3


def validate_rq3_results(
    power_df: pd.DataFrame,
    minimum_required_n_rq3: int,
    analytical_required_n_rq3: float,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    effect_size=ASSUMED_EFFECT_SIZE,
    planned_size=PLANNED_SAMPLE_SIZE
) -> None:
    """
    Runs the validation assertions for RQ3 power calculation.
    """
    assert 0 < alpha < 1, (
        "Alpha must be between 0 and 1."
    )

    assert 0 < target_power < 1, (
        "Target power must be between 0 and 1."
    )

    assert effect_size > 0, (
        "The assumed standardized effect size must be greater than 0."
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

    if minimum_required_n_rq3 is not None:
        assert minimum_required_n_rq3 == int(
            analytical_required_n_rq3
            if analytical_required_n_rq3.is_integer()
            else analytical_required_n_rq3 + 1
        ), (
            "The iterative and analytical sample-size estimates do not match."
        )


def build_rq3_display_table(
    power_df: pd.DataFrame,
    minimum_required_n_rq3: int,
    sample_sizes=SAMPLE_SIZES,
    planned_sample_size=PLANNED_SAMPLE_SIZE
) -> pd.DataFrame:
    """
    Builds the display table featuring candidate sample sizes around minimum required and planned N.
    """
    if minimum_required_n_rq3 is not None:
        lower_limit = max(
            min(sample_sizes),
            minimum_required_n_rq3 - 3
        )
        upper_limit = minimum_required_n_rq3 + 3

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


def create_rich_rq3_table(display_table: pd.DataFrame) -> Table:
    """
    Constructs a styled Rich Table for RQ3 one-sample power analysis.
    """
    table = Table(
        title="RQ3 One-Sample Power Analysis for NYAYaRAG Answer-Quality Evaluation",
        box=box.ROUNDED,
        title_style="bold bright_white",
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("Evaluation Questions", justify="center", style="bold cyan")
    table.add_column("Assumed Effect Size", justify="center", style="yellow")
    table.add_column("Estimated Power", justify="center", style="bold green")
    table.add_column("Target Power Achieved", justify="center", style="bold magenta")

    for _, row in display_table.iterrows():
        achieved = row["Target Power Achieved"]
        status_str = "[bold green]True[/bold green]" if achieved else "[bold red]False[/bold red]"
        table.add_row(
            str(int(row["Evaluation Questions"])),
            f"{row['Assumed Effect Size']:.2f}",
            f"{row['Estimated Power']:.3f}",
            status_str
        )

    return table


def display_rich_rq3_table(
    display_table: pd.DataFrame,
    console_instance: Console = None
) -> None:
    """
    Displays the RQ3 power table using the Rich library.
    """
    con = console_instance or console
    rich_table = create_rich_rq3_table(display_table)
    con.print(rich_table)


def display_rq3_table(
    display_table: pd.DataFrame,
    use_rich: bool = True
) -> None:
    """
    Displays styled table using Rich for terminal output or pandas Styler in Jupyter/IPython.
    """
    if use_rich:
        display_rich_rq3_table(display_table)
        return

    try:
        from IPython.display import display
        styled = (
            display_table.style
            .hide(axis="index")
            .format({
                "Assumed Effect Size": "{:.2f}",
                "Estimated Power": "{:.3f}"
            })
            .set_caption(
                "RQ3 One-Sample Power Analysis for NYAYaRAG Answer-Quality Evaluation"
            )
            .set_properties(**{
                "text-align": "center",
                "vertical-align": "middle"
            })
        )
        display(styled)
    except (ImportError, Exception):
        print("\n" + "=" * 65)
        print("RQ3 One-Sample Power Analysis Table (Selected Sample Sizes)")
        print("=" * 65)
        print(display_table.to_string(index=False))


def print_rq3_conclusion(
    minimum_required_n_rq3: int,
    analytical_required_n_rq3: float,
    planned_sample_power_rq3: float,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    effect_size=ASSUMED_EFFECT_SIZE,
    planned_size=PLANNED_SAMPLE_SIZE
) -> None:
    """
    Prints the final sample-size conclusion and methodological notes for RQ3 using Rich.
    """
    summary_text = (
        f"[bold cyan]System evaluated                    :[/bold cyan] NYAYaRAG\n"
        f"[bold cyan]Primary statistical test           :[/bold cyan] One-sample Wilcoxon signed-rank test\n"
        f"[bold cyan]Sample-size approximation           :[/bold cyan] One-sample normal approximation\n"
        f"[bold cyan]Direction of hypothesis             :[/bold cyan] One-sided\n"
        f"[bold cyan]Significance level                  :[/bold cyan] {alpha:.2f}\n"
        f"[bold cyan]Target statistical power            :[/bold cyan] {target_power:.2f}\n"
        f"[bold cyan]Assumed standardized effect size    :[/bold cyan] {effect_size:.2f}\n\n"
    )

    if minimum_required_n_rq3 is not None:
        summary_text += (
            f"[bold green]• Analytical required sample size      :[/bold green] {analytical_required_n_rq3:.3f}\n"
            f"[bold green]• Minimum whole-number sample size     :[/bold green] [bold bright_white]{minimum_required_n_rq3} questions[/bold bright_white]\n"
            f"[bold green]• Estimated power with {planned_size} questions        :[/bold green] [bold yellow]{planned_sample_power_rq3:.3f}[/bold yellow]\n\n"
        )
        if planned_size >= minimum_required_n_rq3:
            summary_text += (
                f"[bold green]Conclusion:[/bold green] The planned sample of [bold bright_white]{planned_size}[/bold bright_white] "
                f"benchmark questions [bold green]exceeds[/bold green] the estimated minimum sample-size requirement for RQ3.\n"
            )
        else:
            summary_text += (
                f"[bold red]Conclusion:[/bold red] The planned sample of [bold bright_white]{planned_size}[/bold bright_white] "
                f"benchmark questions does not meet the estimated minimum sample-size requirement for RQ3.\n"
            )
    else:
        summary_text += "[bold red]The target statistical power was not achieved within candidate range.[/bold red]\n"

    summary_text += (
        "\n[dim italic]Note: This calculation uses a one-sample normal approximation for prospective planning. "
        "The final hypothesis test for RQ3 will remain the one-sample Wilcoxon signed-rank test.[/dim italic]"
    )

    console.print(Panel(
        summary_text,
        border_style="cyan",
        title="[bold bright_white]RQ3 Sample-Size Determination Summary[/bold bright_white]"
    ))


def main():
    """
    Executes prospective power analysis for RQ3 with Rich formatting.
    """
    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Model Evaluation and Statistical Analysis[/bold cyan]\n"
        "[italic bright_white]Notebook 2 — Section 1.6: Sample Size Determination for RQ3 (Answer Quality)[/italic bright_white]",
        border_style="bright_blue",
        title="[bold green]Power Analysis — RQ3[/bold green]"
    ))

    console.print("[bold yellow]Performing RQ3 One-Sample Power Analysis for Answer-Quality Evaluation...[/bold yellow]\n")
    power_df = calculate_rq3_power_curve()
    min_n, analytical_n = solve_rq3_sample_size(power_df)
    
    planned_power = float(
        power_df.loc[
            power_df["Evaluation Questions"] == PLANNED_SAMPLE_SIZE,
            "Estimated Power"
        ].iloc[0]
    )

    validate_rq3_results(power_df, min_n, analytical_n)
    
    display_tbl = build_rq3_display_table(power_df, min_n)
    display_rq3_table(display_tbl)
    
    print_rq3_conclusion(min_n, analytical_n, planned_power)


if __name__ == "__main__":
    main()

