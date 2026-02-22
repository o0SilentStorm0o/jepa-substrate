"""Shared inference harness for both ANN and SNN.

Implements invariant I3: same inference regime for both substrates.
All evaluation metrics are collected at batch size B=1 in streaming mode.

Both harnesses use the same timing wrapper (shared/timing.py)
and report identical column names (invariant I4).
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Column names for the results CSV (must be identical for ANN and SNN)
RESULT_COLUMNS = [
    "window_index",
    "loss",
    "forward_ms",
    "teacher_ms",
    "overhead_ms",
    "total_ms",
    "embedding_norm_mean",
    "embedding_norm_std",
    "teacher_norm_mean",
    "teacher_norm_std",
    "prediction_variance",
    # SNN-specific (NA for ANN)
    "total_spikes",
    "spike_rate",
    "synaptic_events",
    "energy_proxy",
]


@dataclass
class WindowResult:
    """Per-window measurement result."""
    window_index: int
    loss: float
    forward_ms: float
    teacher_ms: float = 0.0
    overhead_ms: float = 0.0
    total_ms: float = 0.0
    embedding_norm_mean: float = 0.0
    embedding_norm_std: float = 0.0
    teacher_norm_mean: float = 0.0
    teacher_norm_std: float = 0.0
    prediction_variance: float = 0.0
    # SNN-specific
    total_spikes: float = float("nan")
    spike_rate: float = float("nan")
    synaptic_events: float = float("nan")
    energy_proxy: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {col: getattr(self, col) for col in RESULT_COLUMNS}


def save_results_csv(
    results: List[WindowResult],
    output_path: Path | str,
) -> None:
    """Save per-window measurement results to a CSV file.

    Both ANN and SNN harnesses call this with identical column names
    (invariant I4).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    logger.info("Results saved to %s (%d windows)", output_path, len(results))


def save_observables_npz(
    output_path: Path | str,
    window_index: int,
    student_latents: Optional[np.ndarray] = None,
    teacher_targets: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    spike_vectors_enc: Optional[np.ndarray] = None,
    spike_vectors_pred: Optional[np.ndarray] = None,
    spike_traces: Optional[np.ndarray] = None,
    membrane_potentials: Optional[np.ndarray] = None,
) -> None:
    """Save per-timestep observables to a compressed NPZ archive.

    Implements the observability points O1--O12 from the specification.

    Parameters
    ----------
    output_path : Path or str
        Output file path (one file per test window).
    window_index : int
        Index of the test window.
    student_latents : np.ndarray, optional
        O1: Student encoder output, shape ``(T, H)``.
    teacher_targets : np.ndarray, optional
        O2: Teacher encoder output, shape ``(T, H)``.
    predictions : np.ndarray, optional
        O3: Predictor output, shape ``(T, H)``.
    spike_vectors_enc : np.ndarray, optional
        O4: Encoder spike vectors, shape ``(T, H)``.
    spike_vectors_pred : np.ndarray, optional
        O4: Predictor spike vectors, shape ``(T, H)``.
    spike_traces : np.ndarray, optional
        O5: Spike traces after exponential filter, shape ``(T, H)``.
    membrane_potentials : np.ndarray, optional
        O6: Membrane potentials, shape ``(T, H)``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {"window_index": np.array(window_index)}
    if student_latents is not None:
        data["student_latents"] = student_latents.astype(np.float32)
    if teacher_targets is not None:
        data["teacher_targets"] = teacher_targets.astype(np.float32)
    if predictions is not None:
        data["predictions"] = predictions.astype(np.float32)
    if spike_vectors_enc is not None:
        data["spike_vectors_enc"] = spike_vectors_enc.astype(np.float32)
    if spike_vectors_pred is not None:
        data["spike_vectors_pred"] = spike_vectors_pred.astype(np.float32)
    if spike_traces is not None:
        data["spike_traces"] = spike_traces.astype(np.float32)
    if membrane_potentials is not None:
        data["membrane_potentials"] = membrane_potentials.astype(np.float32)

    np.savez_compressed(str(output_path), **data)


def compute_embedding_stats(
    latents: np.ndarray,
    mask: np.ndarray,
    is_teacher: bool = False,
) -> Dict[str, float]:
    """Compute embedding norm statistics for observability.

    Parameters
    ----------
    latents : np.ndarray
        Latent vectors, shape ``(T, H)``.
    mask : np.ndarray
        Binary mask, shape ``(T,)``. 1 = target.
    is_teacher : bool
        If True, compute stats over target positions; else over context.

    Returns
    -------
    dict
        Mean and std of embedding L2 norms.
    """
    if is_teacher:
        indices = np.where(mask > 0.5)[0]
    else:
        indices = np.where(mask < 0.5)[0]

    if len(indices) == 0:
        return {"mean": 0.0, "std": 0.0}

    selected = latents[indices]  # (N_selected, H)
    norms = np.linalg.norm(selected, axis=-1)  # (N_selected,)
    return {"mean": float(norms.mean()), "std": float(norms.std())}
