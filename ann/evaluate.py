"""ANN inference and measurement harness.

Implements the measurement protocol from Section 5.7 for the ANN substrate:
  - Batch size B=1, streaming mode
  - Wall-clock latency decomposition (3 components)
  - Per-window observability logging (O1--O12)
  - Latency benchmark (warm-up + timed repetitions)

Platform-neutral (primary): ANN inference on CPU, single-thread,
torch.set_num_threads(1).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ann.model import ANNModel
from shared.data import WindowDataset
from shared.harness import (
    WindowResult,
    compute_embedding_stats,
    save_observables_npz,
    save_results_csv,
)
from shared.loss import jepa_time_loss
from shared.timing import measure_latency, run_latency_benchmark, BenchmarkResult

logger = logging.getLogger(__name__)


def precompute_teacher_targets(
    model: ANNModel,
    dataset: WindowDataset,
    output_dir: Path | str,
    device: str = "cpu",
) -> Path:
    """Pre-compute all teacher targets for the test set.

    Implements Section 5.6 teacher target handling:
    Teacher targets are pre-computed and stored as float32 .npz files,
    verified by SHA-256 hash.

    Parameters
    ----------
    model : ANNModel
        Trained ANN model with teacher encoder.
    dataset : WindowDataset
        Test dataset.
    output_dir : Path or str
        Output directory for target files.
    device : str
        Device.

    Returns
    -------
    Path
        Directory containing pre-computed targets.
    """
    output_dir = Path(output_dir) / "teacher_targets"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sha = hashlib.sha256()

    with torch.no_grad():
        for idx, (x, mask, _) in enumerate(loader):
            x = x.to(device)
            teacher_out = model.teacher_encoder(x)  # (1, T, H)
            targets_np = teacher_out.cpu().numpy().astype(np.float32)

            filepath = output_dir / f"target_{idx:05d}.npz"
            np.savez(str(filepath), targets=targets_np[0])

            # Update hash for verification
            sha.update(targets_np.tobytes())

    hash_hex = sha.hexdigest()
    hash_file = output_dir / "targets_sha256.txt"
    with open(hash_file, "w", encoding="utf-8") as fh:
        fh.write(hash_hex + "\n")

    logger.info(
        "Pre-computed %d teacher targets, SHA-256: %s",
        len(dataset), hash_hex,
    )

    model.train()
    return output_dir


def load_teacher_target(
    targets_dir: Path | str,
    window_index: int,
) -> np.ndarray:
    """Load a pre-computed teacher target.

    Parameters
    ----------
    targets_dir : Path or str
        Directory containing pre-computed targets.
    window_index : int
        Window index.

    Returns
    -------
    np.ndarray
        Teacher target, shape ``(T, H)``.
    """
    filepath = Path(targets_dir) / f"target_{window_index:05d}.npz"
    data = np.load(str(filepath))
    return data["targets"]


def run_ann_measurement(
    model: ANNModel,
    test_dataset: WindowDataset,
    targets_dir: Path | str,
    output_dir: Path | str,
    device: str = "cpu",
    save_observables: bool = True,
) -> List[WindowResult]:
    """Run full ANN measurement on the test set.

    Implements the measurement protocol:
    - Batch size B=1, sequential iteration
    - Pre-computed teacher targets (component 2 = 0)
    - Per-window observability logging

    Parameters
    ----------
    model : ANNModel
        Trained ANN model.
    test_dataset : WindowDataset
        Test dataset.
    targets_dir : Path or str
        Directory with pre-computed teacher targets.
    output_dir : Path or str
        Output directory for results.
    device : str
        Device (CPU for primary comparison).
    save_observables : bool
        Whether to save per-timestep observables.

    Returns
    -------
    list of WindowResult
        Per-window measurement results.
    """
    output_dir = Path(output_dir)
    targets_dir = Path(targets_dir)
    obs_dir = output_dir / "observables"

    model.eval()
    model = model.to(device)

    # Enforce single-thread CPU for fair comparison
    torch.set_num_threads(1)

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    results: List[WindowResult] = []

    with torch.no_grad():
        for idx, (x, mask, _) in enumerate(loader):
            x = x.to(device)      # (1, T, D)
            mask = mask.to(device) # (1, T)

            # Load pre-computed teacher target
            teacher_np = load_teacher_target(targets_dir, idx)
            teacher_target = torch.from_numpy(teacher_np).unsqueeze(0).to(device)

            # Timed forward pass (component 1)
            def forward_fn():
                return model(x, mask)

            latency = measure_latency(forward_fn)

            # Re-run for actual outputs (forward_fn return is discarded by timing)
            predictions, _ = model(x, mask)
            student_latents = model.get_student_latents(x, mask)

            # Loss
            loss = jepa_time_loss(predictions, teacher_target.detach(), mask)

            # Observability stats
            pred_np = predictions.cpu().numpy()[0]    # (T, H)
            student_np = student_latents.cpu().numpy()[0]  # (T, H)
            teacher_np_full = teacher_target.cpu().numpy()[0]  # (T, H)
            mask_np = mask.cpu().numpy()[0]            # (T,)

            student_stats = compute_embedding_stats(student_np, mask_np, is_teacher=False)
            teacher_stats = compute_embedding_stats(teacher_np_full, mask_np, is_teacher=True)
            pred_variance = float(np.var(pred_np))

            result = WindowResult(
                window_index=idx,
                loss=loss.item(),
                forward_ms=latency.forward_ms,
                teacher_ms=0.0,  # pre-computed
                overhead_ms=latency.overhead_ms,
                total_ms=latency.total_ms,
                embedding_norm_mean=student_stats["mean"],
                embedding_norm_std=student_stats["std"],
                teacher_norm_mean=teacher_stats["mean"],
                teacher_norm_std=teacher_stats["std"],
                prediction_variance=pred_variance,
            )
            results.append(result)

            # Save observables (O1--O3)
            if save_observables:
                save_observables_npz(
                    obs_dir / f"window_{idx:05d}.npz",
                    window_index=idx,
                    student_latents=student_np,
                    teacher_targets=teacher_np_full,
                    predictions=pred_np,
                )

            if idx % 100 == 0:
                logger.info(
                    "ANN measurement: window %d/%d | loss=%.6f | fwd=%.2fms",
                    idx, len(test_dataset), loss.item(), latency.forward_ms,
                )

    # Save results CSV
    save_results_csv(results, output_dir / "results.csv")

    model.train()
    return results


def run_ann_latency_benchmark(
    model: ANNModel,
    test_dataset: WindowDataset,
    n_warmup: int = 5,
    n_timed: int = 1000,
    device: str = "cpu",
) -> BenchmarkResult:
    """Run ANN latency benchmark.

    Implements Section 5.7: warm-up + timed repetitions.
    Only component (1) is timed: model.forward().

    Parameters
    ----------
    model : ANNModel
        Trained ANN model.
    test_dataset : WindowDataset
        Test dataset (uses first window for benchmarking).
    n_warmup : int
        Number of warm-up calls.
    n_timed : int
        Number of timed calls.
    device : str
        Device (CPU for primary comparison).

    Returns
    -------
    BenchmarkResult
        Latency benchmark statistics.
    """
    model.eval()
    model = model.to(device)
    torch.set_num_threads(1)

    # Use first test window
    x, mask, _ = test_dataset[0]
    x = x.unsqueeze(0).to(device)      # (1, T, D)
    mask = mask.unsqueeze(0).to(device) # (1, T)

    def forward_fn():
        with torch.no_grad():
            model(x, mask)

    result = run_latency_benchmark(
        forward_fn,
        n_warmup=n_warmup,
        n_timed=n_timed,
    )

    logger.info(
        "ANN latency benchmark: median=%.3fms, mean=%.3fms, std=%.3fms",
        result.median_ms, result.mean_ms, result.std_ms,
    )

    model.train()
    return result
