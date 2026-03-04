"""PyTorch-based SNN evaluation for LavaJEPA measurement.

Uses the trained SNN model in PyTorch for test-set measurement (loss,
spike count, energy proxy).  This is functionally equivalent to Lava
execution — verified by test V14 (exact 100% spike match, 0 mismatches).

Lava is used separately for the latency benchmark only (1 window × 1000
repeats) because Lava's internal runtime start/stop cycle (~5 s per
window) makes full test-set evaluation impractical.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from shared.harness import (
    WindowResult,
    compute_embedding_stats,
    save_observables_npz,
    save_results_csv,
)
from shared.loss import jepa_time_loss_numpy
from shared.timing import measure_latency

logger = logging.getLogger(__name__)


def run_snn_measurement(
    model: "SNNModel",
    test_windows: np.ndarray,
    test_masks: List[np.ndarray],
    teacher_targets_dir: Path | str,
    output_dir: Path | str,
    T: int = 128,
    H: int = 128,
    c_mask: float = 1.0,
    energy_alpha: float = 23.6e-12,
    energy_beta: float = 81.0e-12,
    energy_gamma: float = 26.0e-9,
    save_observables: bool = False,
) -> List[WindowResult]:
    """Run full SNN measurement on the test set via PyTorch.

    Parameters
    ----------
    model : SNNModel
        Trained SNN model (PyTorch).
    test_windows : np.ndarray
        Test windows, shape ``(N, T, D)``.
    test_masks : list of np.ndarray
        Per-window masks, each shape ``(T,)``.
    teacher_targets_dir : Path or str
        Directory with pre-computed teacher targets.
    output_dir : Path or str
        Output directory.
    T, H : int
        Window length and latent dimension.
    c_mask : float
        Mask bias current magnitude.
    energy_alpha, energy_beta, energy_gamma : float
        Energy proxy coefficients (Eq. 8).
    save_observables : bool
        Whether to save per-window observables.

    Returns
    -------
    list of WindowResult
    """
    output_dir = Path(output_dir)
    teacher_targets_dir = Path(teacher_targets_dir)
    obs_dir = output_dir / "observables"

    model.eval()
    torch.set_num_threads(1)

    n_windows = len(test_windows)
    logger.info("SNN measurement (PyTorch): %d windows", n_windows)

    results: List[WindowResult] = []

    with torch.no_grad():
        for idx in range(n_windows):
            x = torch.from_numpy(test_windows[idx]).float().unsqueeze(0)  # (1,T,D)
            mask = torch.from_numpy(test_masks[idx]).float().unsqueeze(0)  # (1,T)

            # Load pre-computed teacher target
            target_path = teacher_targets_dir / f"target_{idx:05d}.npz"
            teacher_target = np.load(str(target_path))["targets"]  # (T, H)

            # Timed forward pass
            def forward_fn():
                return model(x, mask)

            latency = measure_latency(forward_fn)

            # Full forward for outputs
            predictions_t, _, enc_spikes_t, pred_spikes_t, _ = model(x, mask)

            predictions = predictions_t.cpu().numpy()[0]   # (T, H)
            enc_spikes = enc_spikes_t.cpu().numpy()[0]     # (T, H)
            pred_spikes = pred_spikes_t.cpu().numpy()[0]   # (T, H)
            mask_np = test_masks[idx]                      # (T,)

            # Loss
            loss = jepa_time_loss_numpy(predictions, teacher_target, mask_np)

            # Spike statistics
            total_enc = int(np.sum(enc_spikes > 0))
            total_pred = int(np.sum(pred_spikes > 0))
            total_spikes = total_enc + total_pred
            n_neurons = H * 2
            spike_rate = total_spikes / (n_neurons * T) if n_neurons * T > 0 else 0.0

            # Synaptic events
            synaptic_events = 0
            for t_idx in range(T):
                enc_active = int(np.sum(enc_spikes[t_idx] > 0))
                pred_active = int(np.sum(pred_spikes[t_idx] > 0))
                synaptic_events += enc_active * H   # DensePred fan-out
                synaptic_events += pred_active * H  # DenseReadout fan-out

            # Energy proxy (Eq. 8)
            energy = (
                energy_alpha * synaptic_events
                + energy_beta * total_spikes
                + energy_gamma * T
            )

            # Embedding stats
            pred_stats = compute_embedding_stats(
                predictions, mask_np, is_teacher=False,
            )
            teacher_stats = compute_embedding_stats(
                teacher_target, mask_np, is_teacher=True,
            )
            pred_variance = float(np.var(predictions))

            result = WindowResult(
                window_index=idx,
                loss=loss,
                forward_ms=latency.forward_ms,
                teacher_ms=0.0,
                overhead_ms=latency.overhead_ms,
                total_ms=latency.total_ms,
                embedding_norm_mean=pred_stats["mean"],
                embedding_norm_std=pred_stats["std"],
                teacher_norm_mean=teacher_stats["mean"],
                teacher_norm_std=teacher_stats["std"],
                prediction_variance=pred_variance,
                total_spikes=float(total_spikes),
                spike_rate=spike_rate,
                synaptic_events=float(synaptic_events),
                energy_proxy=energy,
            )
            results.append(result)

            # Save observables
            if save_observables:
                save_observables_npz(
                    obs_dir / f"window_{idx:05d}.npz",
                    window_index=idx,
                    predictions=predictions,
                    teacher_targets=teacher_target,
                    spike_vectors_enc=enc_spikes,
                    spike_vectors_pred=pred_spikes,
                )

            if idx % 100 == 0:
                logger.info(
                    "SNN window %d/%d | loss=%.6f | spikes=%d | "
                    "rate=%.4f | fwd=%.2fms",
                    idx, n_windows, loss, total_spikes,
                    spike_rate, latency.forward_ms,
                )

    save_results_csv(results, output_dir / "results.csv")
    logger.info("SNN measurement complete: %d windows", n_windows)
    return results
