"""Run statistical analysis on completed experiment results.

Reads per-window results.csv from each run directory and produces:
  - Statistical results CSV (paired t-test, Wilcoxon, Cohen's d)
  - LaTeX Table 1 fragment
  - Summary plots
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.loader import load_config
from analysis.stats import (
    ComparisonSpec,
    run_full_analysis,
    save_analysis_csv,
    save_latex_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def collect_per_seed_means(
    output_root: Path,
    seeds: list,
    policy: str,
    condition: str,
    metric: str = "loss",
) -> tuple:
    """Collect per-seed mean metric values for ANN and SNN.

    Returns (ann_means, snn_means) as numpy arrays,
    or (None, None) if data is incomplete.
    """
    ann_means = []
    snn_means = []

    for seed in seeds:
        for substrate, target_list in [("ann", ann_means), ("snn", snn_means)]:
            run_id = f"s{seed}_{policy}_{substrate}_{condition}"
            csv_path = output_root / run_id / "results.csv"

            # Also try dataset-prefixed layout: results/<dataset>/<dataset>_<run_id>/
            if not csv_path.exists():
                for subdir in output_root.iterdir():
                    if subdir.is_dir() and subdir.name not in ("analysis",):
                        alt = subdir / f"{subdir.name}_{run_id}" / "results.csv"
                        if alt.exists():
                            csv_path = alt
                            break

            if not csv_path.exists():
                logger.warning("Missing: %s", csv_path)
                return None, None

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                values = [float(row[metric]) for row in reader]

            target_list.append(float(np.mean(values)))

    return np.array(ann_means), np.array(snn_means)


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "config" / "experiment.yaml"))
    output_root = PROJECT_ROOT / "results"
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg.seeds.init_seeds
    policies = cfg.experiment.policies
    ablations = cfg.experiment.ablations

    comparisons = []

    # Primary comparisons
    for policy in policies:
        ann_means, snn_means = collect_per_seed_means(
            output_root, seeds, policy, "primary",
        )
        if ann_means is not None:
            comparisons.append(ComparisonSpec(
                metric="loss",
                masking_policy=policy,
                condition="primary",
                ann_values=ann_means,
                snn_values=snn_means,
            ))
            logger.info(
                "Primary %s: ANN=%.4f +/- %.4f, SNN=%.4f +/- %.4f",
                policy,
                np.mean(ann_means), np.std(ann_means),
                np.mean(snn_means), np.std(snn_means),
            )

    # Latency comparisons
    for policy in policies:
        ann_lat, snn_lat = collect_per_seed_means(
            output_root, seeds, policy, "primary", metric="forward_ms",
        )
        if ann_lat is not None:
            comparisons.append(ComparisonSpec(
                metric="latency_ms",
                masking_policy=policy,
                condition="primary",
                ann_values=ann_lat,
                snn_values=snn_lat,
            ))

    # Ablation comparisons
    for ablation in ablations:
        ann_means, snn_means = collect_per_seed_means(
            output_root, seeds, "future_block", ablation,
        )
        if ann_means is not None:
            comparisons.append(ComparisonSpec(
                metric="loss",
                masking_policy="future_block",
                condition=ablation,
                ann_values=ann_means,
                snn_values=snn_means,
            ))
            logger.info(
                "Ablation %s: ANN=%.4f +/- %.4f, SNN=%.4f +/- %.4f",
                ablation,
                np.mean(ann_means), np.std(ann_means),
                np.mean(snn_means), np.std(snn_means),
            )

    if not comparisons:
        logger.error("No complete comparisons found. Check that experiments have finished.")
        return

    logger.info("Running analysis on %d comparisons...", len(comparisons))
    report = run_full_analysis(comparisons, alpha=0.05)

    save_analysis_csv(report, analysis_dir / "statistical_results.csv")
    save_latex_table(report, analysis_dir / "table1.tex")

    # Print summary to console
    logger.info("=" * 70)
    logger.info("STATISTICAL RESULTS SUMMARY")
    logger.info("=" * 70)

    for r in report.t_tests:
        sig = "***" if r.reject else "n.s."
        logger.info(
            "%s | %s | %s | p=%.4f (adj=%.4f) | d=%.3f [%.3f, %.3f] | %s",
            r.condition.ljust(15),
            r.masking_policy.ljust(15),
            r.metric.ljust(12),
            r.p_value,
            r.p_adjusted if r.p_adjusted is not None else -1,
            r.effect_size.d if r.effect_size else 0,
            r.effect_size.ci_lower if r.effect_size else 0,
            r.effect_size.ci_upper if r.effect_size else 0,
            sig,
        )

    logger.info("Results saved to %s", analysis_dir)

    # Try to generate plots
    try:
        from analysis.plots import generate_all_figures
        generate_all_figures(output_root, analysis_dir)
        logger.info("Plots saved to %s", analysis_dir)
    except Exception:
        logger.exception("Plot generation failed (non-critical)")


if __name__ == "__main__":
    main()
