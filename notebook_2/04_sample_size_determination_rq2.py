"""
1.5 Sample Size Determination for RQ2

1.5.1 Statistical Basis:
Research Question 2 investigates which retrieval model provides the best retrieval performance for legal question answering over Indian Supreme Court judgments.

The final experimental design compares the following six retrieval methodologies:
1. TF-IDF
2. BM25
3. E5
4. BGE
5. InLegalBERT
6. Hybrid Retrieval

Each retrieval methodology will be evaluated using the same benchmark set of legal research questions. Therefore, every benchmark question produces retrieval-performance scores for all six methodologies. The observations are consequently related rather than independent, resulting in a repeated-measures experimental design.

The primary statistical test selected for RQ2 is the Friedman test. The Friedman test is a non-parametric alternative to the one-way repeated-measures analysis of variance and is appropriate for comparing three or more related groups when the assumptions required for parametric analysis cannot be guaranteed.

For each benchmark legal research question, the six retrieval methodologies will be ranked according to the selected retrieval-performance metric. The Friedman test will determine whether statistically significant differences exist among the methodologies.

If the Friedman test produces a statistically significant result, pairwise Wilcoxon signed-rank post-hoc comparisons will be conducted to determine which retrieval methodologies differ significantly from one another. The resulting p-values will be adjusted using the Holm correction to control the family-wise error rate arising from multiple comparisons.

The hypotheses for RQ2 are defined as follows:
- Null Hypothesis (H₀₂): There is no statistically significant difference in retrieval performance among TF-IDF, BM25, E5, BGE, InLegalBERT, and Hybrid Retrieval.
- Alternative Hypothesis (H₁₂): At least one retrieval methodology exhibits significantly different retrieval performance from the remaining methodologies.

No universally accepted closed-form sample-size formula exists for the Friedman test under a repeated-measures design involving multiple related groups. Therefore, the required sample size for RQ2 is estimated using an asymptotic non-central chi-square approximation based on Kendall’s coefficient of concordance (W).

Kendall’s W represents the magnitude of agreement or systematic ranking differences among the retrieval methodologies and serves as the effect-size measure associated with the Friedman test.

In the absence of pilot-derived estimates, the prospective power analysis assumes:
- Number of retrieval methodologies: 6
- Significance level (alpha): 0.05
- Target statistical power (1 - beta): 0.80
- Assumed Kendall’s coefficient of concordance: W = 0.20
- Prospective sample-size range: 5–100 benchmark questions

The required sample size is identified as the smallest number of benchmark legal research questions for which the estimated statistical power reaches or exceeds 0.80. The result will subsequently be compared with the sample-size requirements obtained for RQ1, RQ3, and RQ4.

---

Table 2. Statistical Design and Power Analysis Parameters for Retrieval Evaluation (RQ2):

| Parameter | Value / Description |
|:---|:---|
| Applicable Research Question | RQ2 – Which retrieval model provides the best retrieval performance for legal question answering over Indian Supreme Court judgments? |
| Retrieval Methodologies Compared | TF-IDF, BM25, E5, BGE, InLegalBERT, and Hybrid Retrieval |
| Experimental Unit | Individual benchmark legal research question |
| Number of Related Groups | Six retrieval methodologies |
| Study Design | Repeated-measures (within-subject) design |
| Primary Statistical Test | Friedman test |
| Post-hoc Analysis | Pairwise Wilcoxon signed-rank tests with Holm correction |
| Significance Level (alpha) | 0.05 |
| Target Statistical Power (1 - beta) | 0.80 |
| Effect-Size Measure | Kendall's coefficient of concordance (W) |
| Assumed Effect Size | W = 0.20 (conservative prospective assumption) |
| Sample-Size Estimation Method | Asymptotic non-central chi-square approximation for the Friedman test |
| Candidate Sample-Size Range | 5–100 benchmark legal research questions |
| Criterion for Minimum Sample Size | Smallest sample size achieving estimated statistical power >= 0.80 |
| Planned Evaluation Sample | 50 benchmark legal research questions |

The parameters presented in Table 2 define the repeated-measures statistical framework adopted for RQ2. The Friedman test was selected because the retrieval performance of six related retrieval methodologies is evaluated using the same benchmark legal research questions. Sample-size estimation is based on an analytical approximation employing Kendall's coefficient of concordance (W) as the effect-size measure. Following identification of the minimum sample size required to achieve the target statistical power, the planned benchmark of 50 legal research questions is assessed for adequacy before conducting the retrieval evaluation.

---

Observations:
The prospective power analysis for RQ2 was performed using an analytical approximation for the Friedman test based on Kendall's coefficient of concordance (W). The analysis assumed a significance level of alpha = 0.05, a target statistical power of 0.80, and a conservative effect-size assumption of W = 0.20.

The estimated statistical power increased monotonically with the number of benchmark legal research questions. The minimum sample size required to achieve the target statistical power was 13 evaluation questions, at which the estimated power reached 0.806.

For the planned benchmark dataset of 50 legal research questions, the estimated statistical power was 1.000, substantially exceeding the target statistical power of 0.80. This indicates that the planned evaluation dataset provides more than adequate statistical power to detect meaningful differences among the six retrieval methodologies using the Friedman test.

Accordingly, the planned benchmark of 50 evaluation questions exceeds the minimum sample-size requirement for RQ2 and is therefore considered sufficient for the subsequent Friedman test and Wilcoxon signed-rank post-hoc analyses.
"""

from pathlib import Path
import sys
import pandas as pd
from scipy.stats import chi2, ncx2

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
# Power-analysis parameters for RQ2
# ------------------------------------------------------------
ALPHA = 0.05
TARGET_POWER = 0.80

# Six retrieval methodologies:
# TF-IDF, BM25, E5, BGE, InLegalBERT, and Hybrid Retrieval
N_METHODOLOGIES = 6

# Conservative prospective effect-size assumption
ASSUMED_KENDALL_W = 0.20

# Candidate numbers of benchmark legal research questions
SAMPLE_SIZES = range(5, 101)

# Planned benchmark sample
PLANNED_SAMPLE_SIZE = 50

# Degrees of freedom
DEGREES_OF_FREEDOM = N_METHODOLOGIES - 1


def calculate_friedman_power(
    n_questions,
    n_methodologies,
    kendall_w,
    alpha
):
    """
    Estimate statistical power for the Friedman test using an
    asymptotic non-central chi-square approximation.

    Parameters
    ----------
    n_questions : int
        Number of benchmark legal research questions.

    n_methodologies : int
        Number of related retrieval methodologies.

    kendall_w : float
        Assumed Kendall's coefficient of concordance.

    alpha : float
        Significance level.

    Returns
    -------
    tuple
        Non-centrality parameter and estimated statistical power.
    """
    degrees_of_freedom = n_methodologies - 1

    critical_value = chi2.ppf(
        1 - alpha,
        df=degrees_of_freedom
    )

    # Approximate non-centrality parameter:
    #
    # lambda = N × (k - 1) × W
    #
    # where:
    # N = number of benchmark questions
    # k = number of retrieval methodologies
    # W = assumed Kendall's coefficient of concordance

    noncentrality_parameter = (
        n_questions
        * (n_methodologies - 1)
        * kendall_w
    )

    estimated_power = 1 - ncx2.cdf(
        critical_value,
        df=degrees_of_freedom,
        nc=noncentrality_parameter
    )

    return noncentrality_parameter, estimated_power


def calculate_rq2_power_curve(
    sample_sizes=SAMPLE_SIZES,
    n_methodologies=N_METHODOLOGIES,
    kendall_w=ASSUMED_KENDALL_W,
    alpha=ALPHA,
    target_power=TARGET_POWER
) -> pd.DataFrame:
    """
    Calculates estimated statistical power across candidate sample sizes
    for the Friedman test comparison of 6 retrieval methodologies.
    """
    power_results = []

    for n_questions in sample_sizes:
        noncentrality, estimated_power = calculate_friedman_power(
            n_questions=n_questions,
            n_methodologies=n_methodologies,
            kendall_w=kendall_w,
            alpha=alpha
        )

        power_results.append({
            "Evaluation Questions": n_questions,
            "Non-Centrality Parameter": noncentrality,
            "Estimated Power": estimated_power,
            "Target Power Achieved": estimated_power >= target_power
        })

    return pd.DataFrame(power_results)


def extract_rq2_sample_size(
    power_df: pd.DataFrame,
    target_power=TARGET_POWER,
    planned_sample_size=PLANNED_SAMPLE_SIZE
):
    """
    Extracts the minimum required sample size and the estimated power
    at the planned benchmark size.
    """
    adequate_sample_sizes = power_df[
        power_df["Estimated Power"] >= target_power
    ]

    if adequate_sample_sizes.empty:
        minimum_required_n_rq2 = None
    else:
        minimum_required_n_rq2 = int(
            adequate_sample_sizes.iloc[0]["Evaluation Questions"]
        )

    planned_sample_power_rq2 = float(
        power_df.loc[
            power_df["Evaluation Questions"] == planned_sample_size,
            "Estimated Power"
        ].iloc[0]
    )

    return minimum_required_n_rq2, planned_sample_power_rq2


def validate_rq2_results(
    power_df: pd.DataFrame,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    kendall_w=ASSUMED_KENDALL_W,
    n_methodologies=N_METHODOLOGIES,
    planned_sample_size=PLANNED_SAMPLE_SIZE
) -> None:
    """
    Runs the validation assertions for RQ2 power calculation.
    """
    assert 0 < alpha < 1, (
        "Alpha must be between 0 and 1."
    )

    assert 0 < target_power < 1, (
        "Target power must be between 0 and 1."
    )

    assert 0 < kendall_w <= 1, (
        "Kendall's W must be greater than 0 and no more than 1."
    )

    assert n_methodologies >= 3, (
        "The Friedman test requires at least three related groups."
    )

    assert planned_sample_size in power_df[
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


def build_rq2_display_table(
    power_df: pd.DataFrame,
    minimum_required_n_rq2: int,
    sample_sizes=SAMPLE_SIZES,
    planned_sample_size=PLANNED_SAMPLE_SIZE
) -> pd.DataFrame:
    """
    Extracts the key display rows around minimum required N and planned N.
    """
    if minimum_required_n_rq2 is not None:
        lower_limit = max(
            min(sample_sizes),
            minimum_required_n_rq2 - 3
        )
        upper_limit = minimum_required_n_rq2 + 3

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


def create_rich_rq2_table(display_table: pd.DataFrame) -> Table:
    """
    Constructs a styled Rich Table for RQ2 analytical power approximation.
    """
    table = Table(
        title="RQ2 Analytical Power Approximation for the Friedman Test",
        box=box.ROUNDED,
        title_style="bold bright_white",
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("Evaluation Questions", justify="center", style="bold cyan")
    table.add_column("Non-Centrality Parameter", justify="center", style="yellow")
    table.add_column("Estimated Power", justify="center", style="bold green")
    table.add_column("Target Power Achieved", justify="center", style="bold magenta")

    for _, row in display_table.iterrows():
        achieved = row["Target Power Achieved"]
        status_str = "[bold green]True[/bold green]" if achieved else "[bold red]False[/bold red]"
        table.add_row(
            str(int(row["Evaluation Questions"])),
            f"{row['Non-Centrality Parameter']:.3f}",
            f"{row['Estimated Power']:.3f}",
            status_str
        )

    return table


def display_rich_rq2_table(
    display_table: pd.DataFrame,
    console_instance: Console = None
) -> None:
    """
    Displays the RQ2 power table using the Rich library.
    """
    con = console_instance or console
    rich_table = create_rich_rq2_table(display_table)
    con.print(rich_table)


def display_rq2_table(
    display_table: pd.DataFrame,
    use_rich: bool = True
) -> None:
    """
    Displays styled table using Rich for terminal output or pandas Styler in Jupyter/IPython.
    """
    if use_rich:
        display_rich_rq2_table(display_table)
        return

    try:
        from IPython.display import display
        styled = (
            display_table.style
            .hide(axis="index")
            .format({
                "Non-Centrality Parameter": "{:.3f}",
                "Estimated Power": "{:.3f}"
            })
            .set_caption(
                "RQ2 Analytical Power Approximation for the Friedman Test"
            )
            .set_properties(**{
                "text-align": "center",
                "vertical-align": "middle"
            })
        )
        display(styled)
    except (ImportError, Exception):
        print("\n" + "=" * 65)
        print("RQ2 Friedman Test Analytical Power Table (Selected Sample Sizes)")
        print("=" * 65)
        print(display_table.to_string(index=False))


def print_rq2_conclusion(
    minimum_required_n_rq2: int,
    planned_sample_power_rq2: float,
    n_methodologies=N_METHODOLOGIES,
    degrees_of_freedom=DEGREES_OF_FREEDOM,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    kendall_w=ASSUMED_KENDALL_W,
    planned_sample_size=PLANNED_SAMPLE_SIZE
) -> None:
    """
    Prints the final sample-size conclusion and methodological notes for RQ2 using Rich.
    """
    critical_chi_sq = chi2.ppf(1 - alpha, df=degrees_of_freedom)

    summary_text = (
        f"[bold cyan]Number of retrieval methodologies :[/bold cyan] {n_methodologies}\n"
        f"[bold cyan]Degrees of freedom                 :[/bold cyan] {degrees_of_freedom}\n"
        f"[bold cyan]Primary statistical test           :[/bold cyan] Friedman test\n"
        f"[bold cyan]Post-hoc analysis                  :[/bold cyan] Wilcoxon signed-rank tests with Holm correction\n"
        f"[bold cyan]Significance level                 :[/bold cyan] {alpha:.2f}\n"
        f"[bold cyan]Target statistical power           :[/bold cyan] {target_power:.2f}\n"
        f"[bold cyan]Assumed Kendall's W                :[/bold cyan] {kendall_w:.2f}\n"
        f"[bold cyan]Critical chi-square value          :[/bold cyan] {critical_chi_sq:.3f}\n\n"
    )

    if minimum_required_n_rq2 is not None:
        summary_text += (
            f"[bold green]• Minimum sample size achieving power ≥ {target_power:.2f}:[/bold green] [bold bright_white]{minimum_required_n_rq2} questions[/bold bright_white]\n"
            f"[bold green]• Estimated power with {planned_sample_size} questions       :[/bold green] [bold yellow]{planned_sample_power_rq2:.3f}[/bold yellow]\n\n"
        )
        if planned_sample_size >= minimum_required_n_rq2:
            summary_text += (
                f"[bold green]Conclusion:[/bold green] The planned sample of [bold bright_white]{planned_sample_size}[/bold bright_white] "
                f"benchmark questions [bold green]exceeds[/bold green] the estimated minimum sample-size requirement for RQ2.\n"
            )
        else:
            summary_text += (
                f"[bold red]Conclusion:[/bold red] The planned sample of [bold bright_white]{planned_sample_size}[/bold bright_white] "
                f"benchmark questions does not meet the estimated minimum sample-size requirement for RQ2.\n"
            )
    else:
        summary_text += "[bold red]The target statistical power was not achieved within candidate range.[/bold red]\n"

    summary_text += (
        "\n[dim italic]Note: This result is based on an asymptotic non-central chi-square approximation "
        "and an assumed Kendall's W of 0.20. It represents a prospective planning estimate rather than "
        "an exact sample-size calculation.[/dim italic]"
    )

    console.print(Panel(
        summary_text,
        border_style="cyan",
        title="[bold bright_white]RQ2 Sample-Size Determination Summary[/bold bright_white]"
    ))


def main():
    """
    Executes prospective power analysis for RQ2 (Friedman test) with Rich formatting.
    """
    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Model Evaluation and Statistical Analysis[/bold cyan]\n"
        "[italic bright_white]Notebook 2 — Section 1.5: Sample Size Determination for RQ2 (Friedman Test)[/italic bright_white]",
        border_style="bright_blue",
        title="[bold green]Power Analysis — RQ2[/bold green]"
    ))

    console.print("[bold yellow]Performing RQ2 Analytical Power Approximation for the Friedman Test...[/bold yellow]\n")
    power_df = calculate_rq2_power_curve()
    min_n, planned_power = extract_rq2_sample_size(power_df)
    
    validate_rq2_results(power_df)
    
    display_tbl = build_rq2_display_table(power_df, min_n)
    display_rq2_table(display_tbl)
    
    print_rq2_conclusion(min_n, planned_power)


if __name__ == "__main__":
    main()

