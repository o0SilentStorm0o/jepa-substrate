"""Visualization for LavaJEPA experiment results.

Generates all figures described in the specification:
  - Fig. 2: Per-window loss comparison (ANN vs SNN) per masking policy.
  - Fig. 3: Latency CDF comparison.
  - Fig. 4: Spike raster / membrane trace for representative window.
  - Fig. 5: Energy proxy distribution (SNN only).
  - Fig. 6: Ablation bar chart.
  - Fig. S1: Training curves (loss vs step).
  - Fig. S2: Embedding norm evolution across windows.

All figures use matplotlib with a consistent style suitable for publication.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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
    "savefig.pad_inches": 0.05,
}

COLOR_ANN = "#1f77b4"
COLOR_SNN = "#d62728"
COLOR_ABLATION = "#2ca02c"


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Fig. 2 -- Per-window loss comparison
# ---------------------------------------------------------------------------

def plot_loss_comparison(
    ann_losses: Dict[str, np.ndarray],
    snn_losses: Dict[str, np.ndarray],
    output_path: Path | str,
    figsize: Tuple[float, float] = (7.0, 2.5),
) -> None:
    """Plot per-window loss distributions for ANN vs SNN, one panel per policy.

    Parameters
    ----------
    ann_losses : dict
        ``{policy_name: losses_array}`` for ANN, each shape ``(n_windows,)``.
    snn_losses : dict
        ``{policy_name: losses_array}`` for SNN.
    output_path : Path or str
        Where to save the figure (PDF).
    figsize : tuple
        Figure size in inches.
    """
    _apply_style()
    policies = sorted(ann_losses.keys())
    n = len(policies)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        ann = ann_losses[policy]
        snn = snn_losses[policy]

        bins = np.linspace(
            min(ann.min(), snn.min()),
            max(ann.max(), snn.max()),
            40,
        )

        ax.hist(ann, bins=bins, alpha=0.6, color=COLOR_ANN, label="ANN", density=True)
        ax.hist(snn, bins=bins, alpha=0.6, color=COLOR_SNN, label="SNN", density=True)
        ax.set_xlabel("JEPA loss")
        ax.set_title(policy.replace("_", " ").title())
        ax.legend(frameon=False)

    axes[0].set_ylabel("Density")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 3 -- Latency CDF
# ---------------------------------------------------------------------------

def plot_latency_cdf(
    ann_latencies: np.ndarray,
    snn_latencies: np.ndarray,
    output_path: Path | str,
    figsize: Tuple[float, float] = (4.0, 3.0),
) -> None:
    """Plot empirical CDF of per-window latency for ANN vs SNN.

    Parameters
    ----------
    ann_latencies : np.ndarray
        ANN latencies in ms, shape ``(n_windows,)``.
    snn_latencies : np.ndarray
        SNN latencies in ms.
    output_path : Path or str
        Output PDF path.
    figsize : tuple
        Figure size.
    """
    _apply_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for data, color, label in [
        (ann_latencies, COLOR_ANN, "ANN"),
        (snn_latencies, COLOR_SNN, "SNN"),
    ]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.step(sorted_data, cdf, where="post", color=color, label=label, linewidth=1.2)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 4 -- Spike raster and membrane trace
# ---------------------------------------------------------------------------

def plot_spike_raster(
    enc_spikes: np.ndarray,
    pred_spikes: np.ndarray,
    membrane: np.ndarray,
    mask: np.ndarray,
    output_path: Path | str,
    neuron_range: Tuple[int, int] = (0, 32),
    figsize: Tuple[float, float] = (7.0, 5.0),
) -> None:
    """Plot spike raster for encoder/predictor and membrane trace.

    Parameters
    ----------
    enc_spikes : np.ndarray
        Encoder spikes, shape ``(T, H)``.
    pred_spikes : np.ndarray
        Predictor spikes, shape ``(T, H)``.
    membrane : np.ndarray
        Encoder membrane potential, shape ``(T, H)``.
    mask : np.ndarray
        Binary mask, shape ``(T,)``.
    output_path : Path or str
        Output PDF path.
    neuron_range : tuple
        Range of neuron indices to display.
    figsize : tuple
        Figure size.
    """
    _apply_style()
    n0, n1 = neuron_range
    T = enc_spikes.shape[0]
    t_axis = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Encoder spike raster
    ax = axes[0]
    enc_sub = enc_spikes[:, n0:n1]
    spike_t, spike_n = np.where(enc_sub > 0)
    ax.scatter(spike_t, spike_n + n0, s=0.3, c="k", marker="|", linewidths=0.5)
    _shade_mask(ax, mask, T)
    ax.set_ylabel("Neuron (enc)")
    ax.set_title("Encoder spike raster")

    # Predictor spike raster
    ax = axes[1]
    pred_sub = pred_spikes[:, n0:n1]
    spike_t, spike_n = np.where(pred_sub > 0)
    ax.scatter(spike_t, spike_n + n0, s=0.3, c="k", marker="|", linewidths=0.5)
    _shade_mask(ax, mask, T)
    ax.set_ylabel("Neuron (pred)")
    ax.set_title("Predictor spike raster")

    # Membrane trace (3 example neurons)
    ax = axes[2]
    example_neurons = [n0, n0 + (n1 - n0) // 3, n0 + 2 * (n1 - n0) // 3]
    for ni in example_neurons:
        if ni < membrane.shape[1]:
            ax.plot(t_axis, membrane[:, ni], linewidth=0.8, label=f"n={ni}")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, label="$v_{th}$")
    _shade_mask(ax, mask, T)
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("$v_t$")
    ax.set_title("Encoder membrane potential")
    ax.legend(frameon=False, ncol=4, loc="upper right")

    fig.tight_layout()
    _save_fig(fig, output_path)


def _shade_mask(ax, mask: np.ndarray, T: int) -> None:
    """Shade target (masked) regions on a plot axis."""
    in_mask = False
    start = 0
    for t in range(T):
        if mask[t] > 0.5 and not in_mask:
            start = t
            in_mask = True
        elif mask[t] < 0.5 and in_mask:
            ax.axvspan(start, t, alpha=0.15, color="red", linewidth=0)
            in_mask = False
    if in_mask:
        ax.axvspan(start, T, alpha=0.15, color="red", linewidth=0)


# ---------------------------------------------------------------------------
# Fig. 5 -- Energy proxy distribution (SNN)
# ---------------------------------------------------------------------------

def plot_energy_distribution(
    energy_values: Dict[str, np.ndarray],
    output_path: Path | str,
    figsize: Tuple[float, float] = (5.0, 3.0),
) -> None:
    """Plot energy proxy distribution per masking policy (SNN only).

    Parameters
    ----------
    energy_values : dict
        ``{policy_name: energy_array}`` in pJ, each shape ``(n_windows,)``.
    output_path : Path or str
        Output PDF path.
    figsize : tuple
        Figure size.
    """
    _apply_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    policies = sorted(energy_values.keys())
    data = [energy_values[p] * 1e12 for p in policies]  # Convert J -> pJ
    labels = [p.replace("_", " ").title() for p in policies]

    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(policies) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy proxy (pJ)")
    ax.set_title("SNN energy proxy per masking policy")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 6 -- Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    output_path: Path | str,
    figsize: Tuple[float, float] = (6.0, 3.5),
) -> None:
    """Plot ablation results as grouped bar chart.

    Parameters
    ----------
    results : dict
        ``{condition: {substrate: (mean, std)}}``.
        Conditions: "full", "no_ema", "no_pos", "no_mask".
        Substrates: "ANN", "SNN".
    output_path : Path or str
        Output PDF path.
    figsize : tuple
        Figure size.
    """
    _apply_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    conditions = list(results.keys())
    x = np.arange(len(conditions))
    width = 0.35

    ann_means = [results[c]["ANN"][0] for c in conditions]
    ann_stds = [results[c]["ANN"][1] for c in conditions]
    snn_means = [results[c]["SNN"][0] for c in conditions]
    snn_stds = [results[c]["SNN"][1] for c in conditions]

    ax.bar(
        x - width / 2, ann_means, width, yerr=ann_stds,
        color=COLOR_ANN, alpha=0.8, label="ANN", capsize=3,
    )
    ax.bar(
        x + width / 2, snn_means, width, yerr=snn_stds,
        color=COLOR_SNN, alpha=0.8, label="SNN", capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in conditions])
    ax.set_ylabel("JEPA loss")
    ax.set_title("Ablation study")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. S1 -- Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    ann_curves: Dict[str, List[Tuple[int, float]]],
    snn_curves: Dict[str, List[Tuple[int, float]]],
    output_path: Path | str,
    figsize: Tuple[float, float] = (7.0, 2.5),
) -> None:
    """Plot training loss vs step for ANN and SNN.

    Parameters
    ----------
    ann_curves : dict
        ``{policy: [(step, loss), ...]}``.
    snn_curves : dict
        ``{policy: [(step, loss), ...]}``.
    output_path : Path or str
        Output PDF path.
    figsize : tuple
        Figure size.
    """
    _apply_style()
    policies = sorted(ann_curves.keys())
    n = len(policies)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        if policy in ann_curves:
            steps, losses = zip(*ann_curves[policy])
            ax.plot(steps, losses, color=COLOR_ANN, label="ANN", linewidth=0.8)
        if policy in snn_curves:
            steps, losses = zip(*snn_curves[policy])
            ax.plot(steps, losses, color=COLOR_SNN, label="SNN", linewidth=0.8)
        ax.set_xlabel("Step")
        ax.set_title(policy.replace("_", " ").title())
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("JEPA loss")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. S2 -- Embedding norm evolution
# ---------------------------------------------------------------------------

def plot_embedding_norms(
    ann_norms: np.ndarray,
    snn_norms: np.ndarray,
    teacher_norms: np.ndarray,
    output_path: Path | str,
    figsize: Tuple[float, float] = (5.0, 3.0),
) -> None:
    """Plot embedding L2 norm across windows.

    Parameters
    ----------
    ann_norms : np.ndarray
        Per-window mean embedding norm for ANN, shape ``(n_windows,)``.
    snn_norms : np.ndarray
        Per-window mean embedding norm for SNN.
    teacher_norms : np.ndarray
        Per-window mean teacher embedding norm.
    output_path : Path or str
        Output PDF path.
    figsize : tuple
        Figure size.
    """
    _apply_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    window_idx = np.arange(len(ann_norms))
    ax.plot(window_idx, ann_norms, color=COLOR_ANN, label="ANN pred", linewidth=0.8)
    ax.plot(window_idx, snn_norms, color=COLOR_SNN, label="SNN pred", linewidth=0.8)
    ax.plot(
        window_idx, teacher_norms, color="gray", linestyle="--",
        label="Teacher", linewidth=0.8,
    )

    ax.set_xlabel("Window index")
    ax.set_ylabel("Mean $\\|z\\|_2$")
    ax.set_title("Embedding norm across test windows")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Combined figure generator
# ---------------------------------------------------------------------------

def generate_all_figures(
    results_dir: Path | str,
    output_dir: Path | str,
) -> None:
    """Load result CSVs and observables, generate all publication figures.

    Parameters
    ----------
    results_dir : Path or str
        Directory containing per-condition result subdirectories.
    output_dir : Path or str
        Directory for output PDFs.

    Notes
    -----
    Expected directory layout under results_dir::

        {seed}_{policy}_{substrate}/
            results.csv
            observables/
                window_*.npz
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import csv

    # Collect per-policy, per-substrate losses and latencies
    ann_losses: Dict[str, List[float]] = {}
    snn_losses: Dict[str, List[float]] = {}
    ann_latencies: List[float] = []
    snn_latencies: List[float] = []
    snn_energies: Dict[str, List[float]] = {}

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "results.csv"
        if not csv_path.exists():
            continue

        parts = subdir.name.split("_")
        if len(parts) < 3:
            continue

        # Parse: seed_policy_substrate
        substrate = parts[-1].upper()
        policy = "_".join(parts[1:-1])

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loss = float(row["loss"])
                latency = float(row["total_ms"])

                if substrate == "ANN":
                    ann_losses.setdefault(policy, []).append(loss)
                    ann_latencies.append(latency)
                elif substrate == "SNN":
                    snn_losses.setdefault(policy, []).append(loss)
                    snn_latencies.append(latency)
                    energy = float(row.get("energy_proxy", 0.0))
                    snn_energies.setdefault(policy, []).append(energy)

    # Fig. 2
    if ann_losses and snn_losses:
        plot_loss_comparison(
            {k: np.array(v) for k, v in ann_losses.items()},
            {k: np.array(v) for k, v in snn_losses.items()},
            output_dir / "fig2_loss_comparison.pdf",
        )
        logger.info("Generated fig2_loss_comparison.pdf")

    # Fig. 3
    if ann_latencies and snn_latencies:
        plot_latency_cdf(
            np.array(ann_latencies),
            np.array(snn_latencies),
            output_dir / "fig3_latency_cdf.pdf",
        )
        logger.info("Generated fig3_latency_cdf.pdf")

    # Fig. 5
    if snn_energies:
        plot_energy_distribution(
            {k: np.array(v) for k, v in snn_energies.items()},
            output_dir / "fig5_energy_distribution.pdf",
        )
        logger.info("Generated fig5_energy_distribution.pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path | str) -> None:
    """Save figure as PDF and close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="pdf")
    plt.close(fig)
