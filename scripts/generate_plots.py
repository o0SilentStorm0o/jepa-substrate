"""Generate all publication plots from experiment results.

Produces:
  - Fig. 2: Loss comparison violin plots (ANN vs SNN, per policy)
  - Fig. 3: Latency CDF
  - Fig. 6: Ablation bar chart (when ablation data available)
  - Summary violin plots for primary + ablation
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Style
STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(STYLE)

COLOR_ANN = "#1f77b4"
COLOR_SNN = "#d62728"


def load_per_seed_means(
    results_dir: Path,
    seeds: list,
    policy: str,
    condition: str,
    metric: str = "loss",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load per-seed mean for ANN and SNN."""
    ann_vals, snn_vals = [], []
    for seed in seeds:
        for substrate, target in [("ann", ann_vals), ("snn", snn_vals)]:
            run_id = f"s{seed}_{policy}_{substrate}_{condition}"
            csv_path = results_dir / run_id / "results.csv"
            if not csv_path.exists():
                return np.array([]), np.array([])
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                vals = [float(row[metric]) for row in reader]
            target.append(float(np.mean(vals)))
    return np.array(ann_vals), np.array(snn_vals)


def main():
    from config.loader import load_config
    cfg = load_config(str(PROJECT_ROOT / "config" / "experiment.yaml"))
    results_dir = PROJECT_ROOT / "results"
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg.seeds.init_seeds
    policies = cfg.experiment.policies
    ablations = cfg.experiment.ablations

    # ── Fig. 2: Primary loss comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=False)
    policy_labels = {
        "future_block": "Future Block",
        "random_drop": "Random Drop",
        "multi_target": "Multi Target",
    }

    for ax_idx, policy in enumerate(policies):
        ax = axes[ax_idx]
        ann_means, snn_means = load_per_seed_means(
            results_dir, seeds, policy, "primary", "loss",
        )
        if len(ann_means) == 0:
            continue

        data = [ann_means, snn_means]
        positions = [1, 2]
        parts = ax.violinplot(data, positions, showmeans=True, showmedians=True)
        for pc, color in zip(parts["bodies"], [COLOR_ANN, COLOR_SNN]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)

        # Overlay individual seeds
        for x_pos, vals, color in [(1, ann_means, COLOR_ANN), (2, snn_means, COLOR_SNN)]:
            ax.scatter(
                np.full_like(vals, x_pos) + np.random.uniform(-0.05, 0.05, len(vals)),
                vals, c=color, s=12, alpha=0.7, zorder=5, edgecolors="white", linewidths=0.3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(["ANN", "SNN"])
        ax.set_title(policy_labels.get(policy, policy))
        ax.set_ylabel("JEPA Loss" if ax_idx == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate means
        ax.text(1, ann_means.mean(), f"{ann_means.mean():.3f}", ha="center",
                va="bottom", fontsize=7, color=COLOR_ANN)
        ax.text(2, snn_means.mean(), f"{snn_means.mean():.3f}", ha="center",
                va="bottom", fontsize=7, color=COLOR_SNN)

    fig.suptitle("Primary: ANN vs SNN JEPA Loss (10 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig2_loss_comparison.pdf"))
    fig.savefig(str(output_dir / "fig2_loss_comparison.png"))
    plt.close(fig)
    logger.info("Saved fig2_loss_comparison")

    # ── Fig. 3: Latency CDF ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    all_ann_lat, all_snn_lat = [], []

    for policy in policies:
        ann_lat, snn_lat = load_per_seed_means(
            results_dir, seeds, policy, "primary", "forward_ms",
        )
        if len(ann_lat) > 0:
            all_ann_lat.extend(ann_lat)
            all_snn_lat.extend(snn_lat)

    if all_ann_lat:
        for vals, color, label in [
            (np.array(all_ann_lat), COLOR_ANN, "ANN"),
            (np.array(all_snn_lat), COLOR_SNN, "SNN"),
        ]:
            sorted_v = np.sort(vals)
            cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            ax.step(sorted_v, cdf, where="post", color=color, label=label, linewidth=1.5)

        ax.set_xlabel("Forward latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("Per-seed mean forward latency: ANN vs SNN")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / "fig3_latency_cdf.pdf"))
        fig.savefig(str(output_dir / "fig3_latency_cdf.png"))
        plt.close(fig)
        logger.info("Saved fig3_latency_cdf")

    # ── Fig. 4: Delta loss (SNN - ANN) per policy ───────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    delta_data = {}
    for policy in policies:
        ann_m, snn_m = load_per_seed_means(results_dir, seeds, policy, "primary", "loss")
        if len(ann_m) > 0:
            delta_data[policy] = snn_m - ann_m

    if delta_data:
        labels = [policy_labels.get(p, p) for p in delta_data.keys()]
        data = list(delta_data.values())

        parts = ax.violinplot(data, range(1, len(data) + 1), showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#9467bd")
            pc.set_alpha(0.6)

        for i, (key, vals) in enumerate(delta_data.items()):
            ax.scatter(
                np.full_like(vals, i + 1) + np.random.uniform(-0.05, 0.05, len(vals)),
                vals, c="#9467bd", s=12, alpha=0.7, zorder=5,
            )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"$\Delta\mathcal{L}$ = SNN − ANN")
        ax.set_title(r"Paired difference $\Delta\mathcal{L}$ per policy (10 seeds)")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(output_dir / "fig4_delta_loss.pdf"))
        fig.savefig(str(output_dir / "fig4_delta_loss.png"))
        plt.close(fig)
        logger.info("Saved fig4_delta_loss")

    # ── Fig. 6: Ablation bar chart ──────────────────────────────────
    ablation_data = {}
    for abl in ablations:
        ann_m, snn_m = load_per_seed_means(results_dir, seeds, "future_block", abl, "loss")
        if len(ann_m) > 0:
            ablation_data[abl] = {
                "ANN": (float(np.mean(ann_m)), float(np.std(ann_m, ddof=1))),
                "SNN": (float(np.mean(snn_m)), float(np.std(snn_m, ddof=1))),
            }

    # Also include primary future_block as "Full" baseline
    ann_fb, snn_fb = load_per_seed_means(results_dir, seeds, "future_block", "primary", "loss")
    if len(ann_fb) > 0:
        ablation_data_with_baseline = {
            "Full (baseline)": {
                "ANN": (float(np.mean(ann_fb)), float(np.std(ann_fb, ddof=1))),
                "SNN": (float(np.mean(snn_fb)), float(np.std(snn_fb, ddof=1))),
            },
        }
        ablation_data_with_baseline.update(ablation_data)

        if len(ablation_data_with_baseline) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            conditions = list(ablation_data_with_baseline.keys())
            x = np.arange(len(conditions))
            width = 0.35

            ann_means = [ablation_data_with_baseline[c]["ANN"][0] for c in conditions]
            ann_stds = [ablation_data_with_baseline[c]["ANN"][1] for c in conditions]
            snn_means = [ablation_data_with_baseline[c]["SNN"][0] for c in conditions]
            snn_stds = [ablation_data_with_baseline[c]["SNN"][1] for c in conditions]

            ax.bar(x - width / 2, ann_means, width, yerr=ann_stds,
                   color=COLOR_ANN, alpha=0.8, label="ANN", capsize=4)
            ax.bar(x + width / 2, snn_means, width, yerr=snn_stds,
                   color=COLOR_SNN, alpha=0.8, label="SNN", capsize=4)

            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=15, ha="right")
            ax.set_ylabel("JEPA Loss")
            ax.set_title("Ablation Study: Future-block policy (10 seeds)")
            ax.legend(frameon=False)
            ax.grid(True, axis="y", alpha=0.3)

            fig.tight_layout()
            fig.savefig(str(output_dir / "fig6_ablation.pdf"))
            fig.savefig(str(output_dir / "fig6_ablation.png"))
            plt.close(fig)
            logger.info("Saved fig6_ablation")

    # ── Summary table to console ────────────────────────────────────
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Policy':<15} {'Substrate':<10} {'Loss (mean±std)':<25} {'Latency ms (mean±std)':<25}")
    print("-" * 75)
    for policy in policies:
        ann_loss, snn_loss = load_per_seed_means(results_dir, seeds, policy, "primary", "loss")
        ann_lat, snn_lat = load_per_seed_means(results_dir, seeds, policy, "primary", "forward_ms")
        if len(ann_loss) > 0:
            print(f"{policy:<15} {'ANN':<10} {ann_loss.mean():.4f} ± {ann_loss.std(ddof=1):.4f}          {ann_lat.mean():.3f} ± {ann_lat.std(ddof=1):.3f}")
            print(f"{'':15} {'SNN':<10} {snn_loss.mean():.4f} ± {snn_loss.std(ddof=1):.4f}          {snn_lat.mean():.3f} ± {snn_lat.std(ddof=1):.3f}")
            delta = snn_loss - ann_loss
            print(f"{'':15} {'Δ(S-A)':<10} {delta.mean():.4f} ± {delta.std(ddof=1):.4f}")
            print()

    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)

    # Read statistical_results.csv
    csv_path = output_dir / "statistical_results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        t_tests = [r for r in rows if r["test_name"] == "paired_t"]
        w_tests = [r for r in rows if r["test_name"] == "wilcoxon"]

        print(f"\n{'Cond':<15} {'Policy':<15} {'Metric':<12} {'t-stat':>8} {'p-val':>12} {'p-adj':>12} {'Reject':>8} {'Cohen d':>10} {'d CI':>25}")
        print("-" * 120)
        for r in t_tests:
            ci = f"[{r['d_ci_lower']}, {r['d_ci_upper']}]" if r["d_ci_lower"] else ""
            print(f"{r['condition']:<15} {r['masking_policy']:<15} {r['metric']:<12} {float(r['statistic']):>8.3f} {r['p_value']:>12} {r['p_adjusted']:>12} {r['reject']:>8} {r['cohens_d']:>10} {ci:>25}")

        print(f"\n\nWilcoxon signed-rank tests:")
        print(f"{'Cond':<15} {'Policy':<15} {'Metric':<12} {'W-stat':>8} {'p-val':>12} {'p-adj':>12} {'Reject':>8}")
        print("-" * 85)
        for r in w_tests:
            print(f"{r['condition']:<15} {r['masking_policy']:<15} {r['metric']:<12} {float(r['statistic']):>8.3f} {r['p_value']:>12} {r['p_adjusted']:>12} {r['reject']:>8}")

    # Ablation status
    print("\n\n" + "=" * 80)
    print("ABLATION STATUS")
    print("=" * 80)
    for abl in ablations:
        ann_m, snn_m = load_per_seed_means(results_dir, seeds, "future_block", abl, "loss")
        if len(ann_m) > 0:
            print(f"{abl:<20} ANN={ann_m.mean():.4f}±{ann_m.std(ddof=1):.4f}  SNN={snn_m.mean():.4f}±{snn_m.std(ddof=1):.4f}")
        else:
            print(f"{abl:<20} [not yet complete]")

    print("\nDone.")


if __name__ == "__main__":
    main()
