"""Statistical analysis for LavaJEPA (Section 4, Table 1).

Implements the full statistical plan S1--S6:
  S1. Paired t-test (H0: mu_ANN = mu_SNN) per metric, per masking policy.
  S2. Wilcoxon signed-rank test as non-parametric counterpart.
  S3. Effect size via Cohen's d with 95 % bootstrap CI (10 000 resamples).
  S4. Holm--Bonferroni correction across P + A = 6 families (3 primary + 3 ablation).
  S5. Decision rule: reject H0 if p_adj < 0.05.
  S6. Report: CSV and LaTeX fragments.

All methods operate on arrays of paired observations (one per seed).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 -- Effect size
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectSizeResult:
    """Cohen's d with bootstrap confidence interval."""

    d: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for paired samples.

    d = mean(x - y) / sd(x - y)

    Parameters
    ----------
    x, y : np.ndarray
        Paired observations, shape ``(n_seeds,)``.

    Returns
    -------
    float
        Cohen's d.
    """
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(np.mean(diff) / sd)


def bootstrap_cohens_d(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> EffectSizeResult:
    """Cohen's d with bootstrap percentile confidence interval.

    Parameters
    ----------
    x, y : np.ndarray
        Paired observations, shape ``(n_seeds,)``.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (0.95 for 95 %).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    EffectSizeResult
        Point estimate and CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(x)
    d_obs = cohens_d(x, y)

    d_boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d_boot[i] = cohens_d(x[idx], y[idx])

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(d_boot, 100 * alpha / 2))
    ci_upper = float(np.percentile(d_boot, 100 * (1 - alpha / 2)))

    return EffectSizeResult(
        d=d_obs,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# S1 / S2 -- Hypothesis tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestResult:
    """Result of a single statistical test."""

    metric: str
    masking_policy: str
    condition: str  # "primary" or ablation label
    test_name: str  # "paired_t" or "wilcoxon"
    statistic: float
    p_value: float
    p_adjusted: Optional[float] = None  # Filled after Holm--Bonferroni
    reject: Optional[bool] = None       # Filled after Holm--Bonferroni
    effect_size: Optional[EffectSizeResult] = None
    n_seeds: int = 0
    mean_ann: float = 0.0
    mean_snn: float = 0.0
    std_ann: float = 0.0
    std_snn: float = 0.0


def paired_t_test(
    ann_values: np.ndarray,
    snn_values: np.ndarray,
    metric: str,
    masking_policy: str,
    condition: str = "primary",
) -> TestResult:
    """S1: Two-sided paired t-test.

    Parameters
    ----------
    ann_values, snn_values : np.ndarray
        Per-seed measurements, shape ``(n_seeds,)``.
    metric : str
        Metric name (e.g., "loss", "latency_ms", "energy_pJ").
    masking_policy : str
        Masking policy label.
    condition : str
        Experiment condition label.

    Returns
    -------
    TestResult
    """
    t_stat, p_val = sp_stats.ttest_rel(ann_values, snn_values)
    return TestResult(
        metric=metric,
        masking_policy=masking_policy,
        condition=condition,
        test_name="paired_t",
        statistic=float(t_stat),
        p_value=float(p_val),
        n_seeds=len(ann_values),
        mean_ann=float(np.mean(ann_values)),
        mean_snn=float(np.mean(snn_values)),
        std_ann=float(np.std(ann_values, ddof=1)),
        std_snn=float(np.std(snn_values, ddof=1)),
    )


def wilcoxon_test(
    ann_values: np.ndarray,
    snn_values: np.ndarray,
    metric: str,
    masking_policy: str,
    condition: str = "primary",
) -> TestResult:
    """S2: Wilcoxon signed-rank test (two-sided).

    Parameters
    ----------
    ann_values, snn_values : np.ndarray
        Per-seed measurements, shape ``(n_seeds,)``.
    metric : str
        Metric name.
    masking_policy : str
        Masking policy label.
    condition : str
        Experiment condition label.

    Returns
    -------
    TestResult
    """
    diff = ann_values - snn_values
    # Wilcoxon requires at least 1 non-zero difference
    if np.all(np.abs(diff) < 1e-15):
        return TestResult(
            metric=metric,
            masking_policy=masking_policy,
            condition=condition,
            test_name="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            n_seeds=len(ann_values),
            mean_ann=float(np.mean(ann_values)),
            mean_snn=float(np.mean(snn_values)),
            std_ann=float(np.std(ann_values, ddof=1)),
            std_snn=float(np.std(snn_values, ddof=1)),
        )

    w_stat, p_val = sp_stats.wilcoxon(ann_values, snn_values)
    return TestResult(
        metric=metric,
        masking_policy=masking_policy,
        condition=condition,
        test_name="wilcoxon",
        statistic=float(w_stat),
        p_value=float(p_val),
        n_seeds=len(ann_values),
        mean_ann=float(np.mean(ann_values)),
        mean_snn=float(np.mean(snn_values)),
        std_ann=float(np.std(ann_values, ddof=1)),
        std_snn=float(np.std(snn_values, ddof=1)),
    )


# ---------------------------------------------------------------------------
# S4 -- Holm--Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(results: List[TestResult], alpha: float = 0.05) -> List[TestResult]:
    """Apply Holm--Bonferroni correction to a family of tests.

    Parameters
    ----------
    results : list of TestResult
        Uncorrected test results within the same family.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    list of TestResult
        New TestResult objects with p_adjusted and reject fields set.
    """
    n = len(results)
    if n == 0:
        return []

    # Sort by p-value
    indexed = sorted(enumerate(results), key=lambda x: x[1].p_value)

    adjusted = [None] * n
    max_adj = 0.0

    for rank, (orig_idx, res) in enumerate(indexed):
        k = n - rank  # denominator: n, n-1, ..., 1
        p_adj = min(res.p_value * k, 1.0)
        # Enforce monotonicity: p_adj must be >= previous adjusted p
        p_adj = max(p_adj, max_adj)
        max_adj = p_adj

        adjusted[orig_idx] = TestResult(
            metric=res.metric,
            masking_policy=res.masking_policy,
            condition=res.condition,
            test_name=res.test_name,
            statistic=res.statistic,
            p_value=res.p_value,
            p_adjusted=p_adj,
            reject=p_adj < alpha,
            effect_size=res.effect_size,
            n_seeds=res.n_seeds,
            mean_ann=res.mean_ann,
            mean_snn=res.mean_snn,
            std_ann=res.std_ann,
            std_snn=res.std_snn,
        )

    return adjusted


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

@dataclass
class ComparisonSpec:
    """Specification of a single ANN vs SNN comparison."""

    metric: str
    masking_policy: str
    condition: str
    ann_values: np.ndarray
    snn_values: np.ndarray


@dataclass
class AnalysisReport:
    """Complete statistical analysis report."""

    t_tests: List[TestResult] = field(default_factory=list)
    wilcoxon_tests: List[TestResult] = field(default_factory=list)
    effect_sizes: Dict[str, EffectSizeResult] = field(default_factory=dict)


def run_full_analysis(
    comparisons: List[ComparisonSpec],
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    rng_seed: int = 42,
) -> AnalysisReport:
    """Run the full statistical analysis pipeline S1--S5.

    Parameters
    ----------
    comparisons : list of ComparisonSpec
        All pairwise comparisons to evaluate.
    alpha : float
        Family-wise error rate for Holm--Bonferroni.
    n_bootstrap : int
        Bootstrap resamples for Cohen's d CI.
    rng_seed : int
        Seed for bootstrap RNG.

    Returns
    -------
    AnalysisReport
        Complete analysis with corrected p-values.
    """
    rng = np.random.default_rng(rng_seed)

    # S1: Paired t-tests
    raw_t = [
        paired_t_test(
            c.ann_values, c.snn_values, c.metric, c.masking_policy, c.condition,
        )
        for c in comparisons
    ]

    # S2: Wilcoxon tests
    raw_w = [
        wilcoxon_test(
            c.ann_values, c.snn_values, c.metric, c.masking_policy, c.condition,
        )
        for c in comparisons
    ]

    # S3: Effect sizes
    effect_sizes = {}
    for c in comparisons:
        key = f"{c.condition}/{c.masking_policy}/{c.metric}"
        es = bootstrap_cohens_d(c.ann_values, c.snn_values, n_bootstrap, 0.95, rng)
        effect_sizes[key] = es

    # Attach effect sizes to t-test results
    enriched_t = []
    for res, c in zip(raw_t, comparisons):
        key = f"{c.condition}/{c.masking_policy}/{c.metric}"
        enriched_t.append(TestResult(
            metric=res.metric,
            masking_policy=res.masking_policy,
            condition=res.condition,
            test_name=res.test_name,
            statistic=res.statistic,
            p_value=res.p_value,
            effect_size=effect_sizes[key],
            n_seeds=res.n_seeds,
            mean_ann=res.mean_ann,
            mean_snn=res.mean_snn,
            std_ann=res.std_ann,
            std_snn=res.std_snn,
        ))

    # S4: Holm--Bonferroni across P + A = 6 hypothesis tests (Section 5.2)
    # Per main.tex: correction across all loss tests (3 primary + 3 ablation)
    # Latency tests are corrected in a separate family.
    loss_indices_t = [i for i, r in enumerate(enriched_t) if r.metric == "loss"]
    lat_indices_t = [i for i, r in enumerate(enriched_t) if r.metric != "loss"]

    adjusted_t = list(enriched_t)

    # Loss family: all P + A loss tests together
    if loss_indices_t:
        loss_family = [enriched_t[i] for i in loss_indices_t]
        corrected = holm_bonferroni(loss_family, alpha)
        for local_idx, global_idx in enumerate(loss_indices_t):
            adjusted_t[global_idx] = corrected[local_idx]

    # Latency family: all latency tests together
    if lat_indices_t:
        lat_family = [enriched_t[i] for i in lat_indices_t]
        corrected = holm_bonferroni(lat_family, alpha)
        for local_idx, global_idx in enumerate(lat_indices_t):
            adjusted_t[global_idx] = corrected[local_idx]

    # Same for Wilcoxon
    loss_indices_w = [i for i, r in enumerate(raw_w) if r.metric == "loss"]
    lat_indices_w = [i for i, r in enumerate(raw_w) if r.metric != "loss"]

    adjusted_w = list(raw_w)

    if loss_indices_w:
        loss_family = [raw_w[i] for i in loss_indices_w]
        corrected = holm_bonferroni(loss_family, alpha)
        for local_idx, global_idx in enumerate(loss_indices_w):
            adjusted_w[global_idx] = corrected[local_idx]

    if lat_indices_w:
        lat_family = [raw_w[i] for i in lat_indices_w]
        corrected = holm_bonferroni(lat_family, alpha)
        for local_idx, global_idx in enumerate(lat_indices_w):
            adjusted_w[global_idx] = corrected[local_idx]

    report = AnalysisReport(
        t_tests=adjusted_t,
        wilcoxon_tests=adjusted_w,
        effect_sizes=effect_sizes,
    )

    n_loss = len(loss_indices_t)
    n_lat = len(lat_indices_t)
    logger.info(
        "Statistical analysis complete: %d comparisons (%d loss + %d latency)",
        len(comparisons), n_loss, n_lat,
    )

    return report


# ---------------------------------------------------------------------------
# S6 -- Output
# ---------------------------------------------------------------------------

def save_analysis_csv(report: AnalysisReport, output_path: Path | str) -> None:
    """Save statistical analysis to CSV.

    Parameters
    ----------
    report : AnalysisReport
        Completed analysis.
    output_path : Path or str
        Output CSV file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_name", "condition", "masking_policy", "metric",
        "mean_ann", "std_ann", "mean_snn", "std_snn",
        "statistic", "p_value", "p_adjusted", "reject",
        "cohens_d", "d_ci_lower", "d_ci_upper",
        "n_seeds",
    ]

    all_results = report.t_tests + report.wilcoxon_tests

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "test_name": r.test_name,
                "condition": r.condition,
                "masking_policy": r.masking_policy,
                "metric": r.metric,
                "mean_ann": f"{r.mean_ann:.6f}",
                "std_ann": f"{r.std_ann:.6f}",
                "mean_snn": f"{r.mean_snn:.6f}",
                "std_snn": f"{r.std_snn:.6f}",
                "statistic": f"{r.statistic:.6f}",
                "p_value": f"{r.p_value:.6e}",
                "p_adjusted": f"{r.p_adjusted:.6e}" if r.p_adjusted is not None else "",
                "reject": str(r.reject) if r.reject is not None else "",
                "cohens_d": f"{r.effect_size.d:.4f}" if r.effect_size else "",
                "d_ci_lower": f"{r.effect_size.ci_lower:.4f}" if r.effect_size else "",
                "d_ci_upper": f"{r.effect_size.ci_upper:.4f}" if r.effect_size else "",
                "n_seeds": r.n_seeds,
            }
            writer.writerow(row)

    logger.info("Saved analysis CSV to %s", output_path)


def generate_latex_table(report: AnalysisReport) -> str:
    r"""Generate LaTeX table fragment for Table 1 of the publication.

    Produces a tabular environment with columns for each masking policy
    and rows for ANN/SNN mean +/- std, p-value, and Cohen's d.

    Parameters
    ----------
    report : AnalysisReport
        Completed analysis.

    Returns
    -------
    str
        LaTeX code.
    """
    lines = []
    lines.append(r"\begin{tabular}{l l " + "c " * 3 + "}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & Substrate & Future-block & Random-drop & Multi-target \\"
    )
    lines.append(r"\midrule")

    # Group t-test results by metric
    primary_t = [r for r in report.t_tests if r.condition == "primary"]
    metrics = sorted(set(r.metric for r in primary_t))
    policies = ["future_block", "random_drop", "multi_target"]

    for metric in metrics:
        # Build lookup
        lookup: Dict[str, TestResult] = {}
        for r in primary_t:
            if r.metric == metric:
                lookup[r.masking_policy] = r

        # ANN row
        ann_cells = []
        for p in policies:
            if p in lookup:
                r = lookup[p]
                ann_cells.append(f"${r.mean_ann:.4f} \\pm {r.std_ann:.4f}$")
            else:
                ann_cells.append("--")

        lines.append(
            f"{metric} & ANN & " + " & ".join(ann_cells) + r" \\"
        )

        # SNN row
        snn_cells = []
        for p in policies:
            if p in lookup:
                r = lookup[p]
                snn_cells.append(f"${r.mean_snn:.4f} \\pm {r.std_snn:.4f}$")
            else:
                snn_cells.append("--")

        lines.append(
            f" & SNN & " + " & ".join(snn_cells) + r" \\"
        )

        # p-value row
        p_cells = []
        for p in policies:
            if p in lookup:
                r = lookup[p]
                p_adj = r.p_adjusted if r.p_adjusted is not None else r.p_value
                star = "*" if r.reject else ""
                p_cells.append(f"${p_adj:.3e}{star}$")
            else:
                p_cells.append("--")

        lines.append(
            f" & $p_{{adj}}$ & " + " & ".join(p_cells) + r" \\"
        )

        # Cohen's d row
        d_cells = []
        for p in policies:
            key = f"primary/{p}/{metric}"
            if key in report.effect_sizes:
                es = report.effect_sizes[key]
                d_cells.append(
                    f"${es.d:.2f}\\;[{es.ci_lower:.2f},\\,{es.ci_upper:.2f}]$"
                )
            else:
                d_cells.append("--")

        lines.append(
            f" & $d$ & " + " & ".join(d_cells) + r" \\"
        )
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def save_latex_table(report: AnalysisReport, output_path: Path | str) -> None:
    """Write LaTeX table to file.

    Parameters
    ----------
    report : AnalysisReport
        Completed analysis.
    output_path : Path or str
        Output .tex file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex = generate_latex_table(report)
    output_path.write_text(tex, encoding="utf-8")
    logger.info("Saved LaTeX table to %s", output_path)
