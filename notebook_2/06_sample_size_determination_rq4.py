"""
1.7 Sample Size Determination for NYAYaRAG Abstention Evaluation (RQ4)

1.7.1 Statistical Basis:
Research Question 4 evaluates whether NYAYaRAG appropriately abstains from generating an answer when a legal research question cannot be answered from the retrieved Indian Supreme Court judgments. In such cases, the system should acknowledge that sufficient supporting information is unavailable rather than generate an unsupported or potentially hallucinated answer.

Each unanswerable benchmark question produces a binary outcome:
- Correct abstention: NYAYaRAG appropriately declines to answer because the retrieved judgments do not contain sufficient supporting information.
- Incorrect abstention decision: NYAYaRAG generates an unsupported answer despite the absence of adequate evidence in the retrieved judgments.

The outcome variable is therefore the proportion of correct abstention decisions. As each question produces either a correct or incorrect abstention outcome, RQ4 follows a one-sample binomial evaluation design.

The primary inferential procedure selected for RQ4 is the exact one-sided binomial test. This test will determine whether the observed correct-abstention proportion is significantly greater than a predefined benchmark proportion.

Because no universally established correct-abstention benchmark currently exists for legal retrieval-augmented generation systems, three prospective benchmark proportions were examined:
- 50% benchmark (p0 = 0.50): Determines whether NYAYaRAG performs better than a basic majority-level threshold.
- 60% benchmark (p0 = 0.60): Represents a moderate minimum acceptable level of correct abstention performance.
- 70% benchmark (p0 = 0.70): Represents a comparatively demanding minimum acceptable level of correct abstention performance.

For each benchmark, the hypotheses are expressed generally as:
- Null Hypothesis (H₀₄): The true proportion of correct abstention decisions is less than or equal to the specified benchmark proportion.
- Alternative Hypothesis (H₁₄): The true proportion of correct abstention decisions is greater than the specified benchmark proportion.

Mathematically:
H₀₄: p <= p0
H₁₄: p > p0

where:
- p represents the true correct-abstention proportion of NYAYaRAG; and
- p0 represents the selected benchmark proportion of 0.50, 0.60, or 0.70.

The prospective power analysis assumes an expected correct-abstention proportion of p1 = 0.85, representing the anticipated performance of NYAYaRAG under the alternative hypothesis. The remaining statistical assumptions are:
- Significance level (alpha): 0.05
- Target statistical power (1 - beta): 0.80
- Expected correct-abstention proportion (p1): 0.85
- Alternative hypothesis: One-sided
- Candidate sample-size range: 5–100 unanswerable benchmark questions

For each benchmark proportion, the minimum required sample size was defined as the smallest number of unanswerable benchmark questions for which the exact statistical power reached or exceeded 0.80.

---

Table 4. Comparison of Statistical Design and Sample-Size Requirements for Alternative Abstention Benchmarks (RQ4):

| Parameter | 50% Benchmark | 60% Benchmark | 70% Benchmark |
|:---|:---:|:---:|:---:|
| Benchmark proportion (p0) | 0.50 | 0.60 | 0.70 |
| Expected proportion (p1) | 0.85 | 0.85 | 0.85 |
| Significance level (alpha) | 0.05 | 0.05 | 0.05 |
| Target power (1 - beta) | 0.80 | 0.80 | 0.80 |
| Primary statistical test | Exact one-sided binomial test | Exact one-sided binomial test | Exact one-sided binomial test |
| Minimum required sample size | 13 | 21 | 49 |
| Critical correct abstentions | 10 of 13 | 17 of 21 | 40 of 49 |
| Estimated power at minimum sample size | 0.882 | 0.803 | 0.809 |
| Interpretation | Tests whether performance exceeds a basic 50% threshold | Tests whether performance exceeds a moderate 60% threshold | Tests whether performance exceeds a demanding 70% threshold |

The prospective analysis demonstrates that the minimum required sample size increases as the benchmark becomes more demanding. A benchmark of 50% requires 13 unanswerable questions, whereas benchmarks of 60% and 70% require 21 and 49 unanswerable questions, respectively.

The 50% benchmark provides the smallest sample-size requirement but represents a comparatively weak standard for a legal question-answering system. The 70% benchmark provides a stronger safety-oriented standard but substantially increases the required evaluation sample. The 60% benchmark offers an intermediate and practically feasible threshold while still requiring NYAYaRAG to demonstrate correct-abstention performance meaningfully above a basic majority-level benchmark.

Subject to the final methodological justification, the 60% benchmark may therefore be adopted for RQ4. Under this design, the final hypotheses would be:
H₀₄: p <= 0.60
H₁₄: p > 0.60

The corresponding minimum required sample size would be 21 unanswerable benchmark legal research questions. A planned sample of 25–30 unanswerable questions would exceed this minimum requirement and provide a modest allowance for exclusions, ambiguous classifications, or unusable evaluation instances.

---

Observations:
The exact one-sided binomial power analysis compared three possible benchmark proportions for evaluating the abstention performance of NYAYaRAG. In all three scenarios, the expected correct-abstention proportion was fixed at p1 = 0.85, the significance level was set at alpha = 0.05, and the target statistical power was 1 - beta = 0.80.

For the 50% benchmark (p0 = 0.50), the minimum required sample size was 13 unanswerable questions. At least 10 correct abstentions out of 13 questions would be required to reject the null hypothesis. The estimated statistical power at this sample size was 0.882. Although this design requires the smallest evaluation sample, a 50% benchmark represents a relatively weak standard because it only tests whether NYAYaRAG performs better than a basic majority-level threshold.

For the 60% benchmark (p0 = 0.60), the minimum required sample size was 21 unanswerable questions. At least 17 correct abstentions out of 21 questions would be required to reject the null hypothesis. The estimated statistical power was 0.803, which marginally exceeds the predefined target of 0.80. This benchmark represents a moderate minimum performance threshold and provides a balance between statistical adequacy and practical feasibility.

For the 70% benchmark (p0 = 0.70), the minimum required sample size was 49 unanswerable questions. At least 40 correct abstentions out of 49 questions would be required to reject the null hypothesis. The estimated statistical power was 0.809. This benchmark imposes a more demanding abstention-performance standard but substantially increases the number of unanswerable questions required for evaluation.

The results demonstrate that the required sample size increases as the benchmark proportion approaches the expected performance of 85%. The difference between the null benchmark and the expected performance is largest for the 50% benchmark and smallest for the 70% benchmark. Consequently, fewer questions are required to detect improvement over 50%, whereas a much larger sample is required to establish that the true correct-abstention rate exceeds 70%.

Considering both methodological rigour and practical feasibility, the 60% benchmark provides an appropriate intermediate threshold for the present study. Under this design, a minimum of 21 unanswerable benchmark legal research questions is required. A planned sample of approximately 25–30 unanswerable questions would exceed this minimum requirement and provide an allowance for exclusions, ambiguous classifications, or unusable evaluation instances.
"""

from pathlib import Path
import sys
import pandas as pd
from scipy.stats import binom

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
# Common parameters for RQ4
# ------------------------------------------------------------
EXPECTED_PROPORTION = 0.85
ALPHA = 0.05
TARGET_POWER = 0.80

# Compare different benchmark proportions
NULL_PROPORTIONS = [0.50, 0.60, 0.70]

# Candidate numbers of unanswerable benchmark questions
SAMPLE_SIZES = range(5, 101)


def calculate_exact_power(n, p0, p1, alpha):
    """
    Calculates the critical number of successes and exact binomial statistical power.

    Parameters
    ----------
    n : int
        Sample size (number of unanswerable questions).
    p0 : float
        Benchmark null hypothesis proportion.
    p1 : float
        Expected true proportion under alternative hypothesis.
    alpha : float
        Significance level.

    Returns
    -------
    tuple
        (critical_successes, power)
    """
    critical_successes = None

    for k in range(n + 1):
        if binom.sf(k - 1, n, p0) <= alpha:
            critical_successes = k
            break

    if critical_successes is None:
        return None, 0.0

    power = binom.sf(
        critical_successes - 1,
        n,
        p1
    )

    return critical_successes, power


def evaluate_abstention_benchmarks(
    null_proportions=NULL_PROPORTIONS,
    sample_sizes=SAMPLE_SIZES,
    expected_proportion=EXPECTED_PROPORTION,
    alpha=ALPHA,
    target_power=TARGET_POWER
) -> pd.DataFrame:
    """
    Evaluates exact power across different benchmark proportions (50%, 60%, 70%)
    and identifies the minimum required sample size and critical correct abstentions.
    """
    summary = []

    for p0 in null_proportions:
        minimum_n = None
        critical = None
        achieved_power = None

        for n in sample_sizes:
            critical_successes, power = calculate_exact_power(
                n,
                p0,
                expected_proportion,
                alpha
            )

            if power >= target_power:
                minimum_n = n
                critical = critical_successes
                achieved_power = power
                break

        summary.append({
            "Benchmark Proportion (p0)": p0,
            "Expected Proportion (p1)": expected_proportion,
            "Minimum Sample Size": minimum_n,
            "Critical Correct Abstentions": critical,
            "Estimated Power": achieved_power
        })

    return pd.DataFrame(summary)


def create_rich_rq4_table(summary_df: pd.DataFrame) -> Table:
    """
    Constructs a styled Rich Table for RQ4 benchmark comparisons.
    """
    table = Table(
        title="Comparison of Sample Size Requirements for Different Abstention Benchmarks",
        box=box.ROUNDED,
        title_style="bold bright_white",
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("Benchmark Proportion (p0)", justify="center", style="bold cyan")
    table.add_column("Expected Proportion (p1)", justify="center", style="yellow")
    table.add_column("Minimum Sample Size", justify="center", style="bold bright_white")
    table.add_column("Critical Correct Abstentions", justify="center", style="bold green")
    table.add_column("Estimated Power", justify="center", style="bold magenta")

    for _, row in summary_df.iterrows():
        min_n_str = str(int(row["Minimum Sample Size"])) if pd.notna(row["Minimum Sample Size"]) else "N/A"
        crit_str = str(int(row["Critical Correct Abstentions"])) if pd.notna(row["Critical Correct Abstentions"]) else "N/A"
        pwr_str = f"{row['Estimated Power']:.3f}" if pd.notna(row["Estimated Power"]) else "N/A"

        table.add_row(
            f"{row['Benchmark Proportion (p0)']:.2f}",
            f"{row['Expected Proportion (p1)']:.2f}",
            min_n_str,
            f"{crit_str} of {min_n_str}",
            pwr_str
        )

    return table


def display_rich_rq4_table(
    summary_df: pd.DataFrame,
    console_instance: Console = None
) -> None:
    """
    Displays the RQ4 benchmark comparison table using the Rich library.
    """
    con = console_instance or console
    rich_table = create_rich_rq4_table(summary_df)
    con.print(rich_table)


def display_rq4_benchmark_comparison(
    summary_df: pd.DataFrame,
    use_rich: bool = True
) -> None:
    """
    Displays the benchmark comparison table using Rich or pandas Styler in Jupyter.
    """
    if use_rich:
        display_rich_rq4_table(summary_df)
        return

    try:
        from IPython.display import display
        styled = (
            summary_df.style
            .hide(axis="index")
            .format({
                "Benchmark Proportion (p0)": "{:.2f}",
                "Expected Proportion (p1)": "{:.2f}",
                "Estimated Power": "{:.3f}"
            })
            .set_caption("Comparison of Sample Size Requirements for Different Abstention Benchmarks")
        )
        display(styled)
    except (ImportError, Exception):
        print("\n" + "=" * 75)
        print("Comparison of Sample Size Requirements for Different Abstention Benchmarks")
        print("=" * 75)
        print(summary_df.to_string(index=False))


def print_rq4_conclusion(summary_df: pd.DataFrame) -> None:
    """
    Prints methodological observations and conclusions for RQ4 using Rich.
    """
    summary_text = (
        f"[bold cyan]Significance Level (alpha)        :[/bold cyan] {ALPHA:.2f}\n"
        f"[bold cyan]Target Statistical Power (1-beta) :[/bold cyan] {TARGET_POWER:.2f}\n"
        f"[bold cyan]Expected Correct-Abstention (p1)  :[/bold cyan] {EXPECTED_PROPORTION:.2f}\n"
        f"[bold cyan]Primary Statistical Test          :[/bold cyan] Exact one-sided binomial test\n\n"
    )

    for _, row in summary_df.iterrows():
        p0 = row["Benchmark Proportion (p0)"]
        min_n = int(row["Minimum Sample Size"]) if pd.notna(row["Minimum Sample Size"]) else "N/A"
        crit = int(row["Critical Correct Abstentions"]) if pd.notna(row["Critical Correct Abstentions"]) else "N/A"
        pwr = row["Estimated Power"]
        summary_text += (
            f"• [bold yellow]Benchmark p0 = {p0:.2f}:[/bold yellow] Min N = [bold bright_white]{min_n}[/bold bright_white] questions, "
            f"Critical Abstentions = [bold green]{crit} of {min_n}[/bold green], Achieved Power = [bold green]{pwr:.3f}[/bold green]\n"
        )

    summary_text += (
        "\n[bold green]Recommended Decision for RQ4:[/bold green]\n"
        "The [bold bright_white]60% benchmark[/bold bright_white] provides an optimal balance between methodological\n"
        "rigour and practical feasibility, requiring at least [bold bright_white]21 unanswerable questions[/bold bright_white].\n"
        "A planned dataset of [bold green]25 unanswerable questions[/bold green] comfortably meets this requirement."
    )

    console.print(Panel(
        summary_text,
        border_style="cyan",
        title="[bold bright_white]RQ4 Abstention Sample-Size Determination Summary[/bold bright_white]"
    ))


def main():
    """
    Executes prospective power analysis for RQ4 across alternative abstention benchmarks with Rich formatting.
    """
    console.print(Panel.fit(
        "[bold cyan]NYAYaRAG: Model Evaluation and Statistical Analysis[/bold cyan]\n"
        "[italic bright_white]Notebook 2 — Section 1.7: Sample Size Determination for RQ4 (Abstention)[/italic bright_white]",
        border_style="bright_blue",
        title="[bold green]Power Analysis — RQ4[/bold green]"
    ))

    console.print("[bold yellow]Performing RQ4 Exact Binomial Power Analysis across Abstention Benchmarks...[/bold yellow]\n")
    summary_df = evaluate_abstention_benchmarks()
    display_rq4_benchmark_comparison(summary_df)
    print_rq4_conclusion(summary_df)


if __name__ == "__main__":
    main()

