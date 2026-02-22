"""Data pipeline for LavaJEPA experiment.

Implements D1--D6 from the specification:
  D1: Download (data/download.py)
  D2: Subject split (by subject ID, not by random windows)
  D3: Windowing (fixed-length T=128, stride=T, no overlap)
  D4: Normalization (per-channel z-score on training windows only)
  D5: Mask generation (deterministic per-window seed)
  D6: Iterator (WindowDataset, sequential for measurement)

Both ANN and SNN use this identical data pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from shared.masking import generate_mask

logger = logging.getLogger(__name__)


def subject_split(
    all_subject_ids: np.ndarray,
    n_train: int = 21,
    n_val: int = 4,
    n_test: int = 5,
    split_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Deterministic subject-level split.

    Subject IDs are sorted, then deterministically partitioned using
    the data-split seed. The mapping subject_id -> {train, val, test}
    is returned and should be saved as JSON.

    Parameters
    ----------
    all_subject_ids : np.ndarray
        Array of all unique subject IDs.
    n_train : int
        Number of training subjects.
    n_val : int
        Number of validation subjects.
    n_test : int
        Number of test subjects.
    split_seed : int
        Deterministic split seed (fixed across all runs).

    Returns
    -------
    dict
        Mapping from split name to array of subject IDs.
    """
    unique_ids = np.sort(np.unique(all_subject_ids))
    n_total = len(unique_ids)
    assert n_train + n_val + n_test == n_total, (
        f"Split sizes ({n_train}+{n_val}+{n_test}={n_train+n_val+n_test}) "
        f"must sum to total subjects ({n_total})"
    )

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(unique_ids)

    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train: n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:]),
    }


def save_subject_split(
    split_map: Dict[str, np.ndarray],
    output_path: Path | str,
) -> None:
    """Save subject split mapping to JSON for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.tolist() for k, v in split_map.items()}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    logger.info("Subject split saved to %s", output_path)


def load_subject_split(path: Path | str) -> Dict[str, np.ndarray]:
    """Load subject split mapping from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: np.array(v, dtype=np.int32) for k, v in raw.items()}


def compute_normalization_stats(
    train_windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std on training windows only.

    Parameters
    ----------
    train_windows : np.ndarray
        Training data of shape ``(N_train, T, D)``.

    Returns
    -------
    mu : np.ndarray
        Per-channel mean, shape ``(D,)``.
    sigma : np.ndarray
        Per-channel std, shape ``(D,)``.
    """
    # Flatten across windows and timesteps: (N*T, D)
    flat = train_windows.reshape(-1, train_windows.shape[-1])
    mu = flat.mean(axis=0).astype(np.float32)
    sigma = flat.std(axis=0).astype(np.float32)
    # Prevent division by zero
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def normalize(
    windows: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Apply per-channel z-score normalization.

    Parameters
    ----------
    windows : np.ndarray
        Data of shape ``(N, T, D)``.
    mu : np.ndarray
        Per-channel mean, shape ``(D,)``.
    sigma : np.ndarray
        Per-channel std, shape ``(D,)``.

    Returns
    -------
    np.ndarray
        Normalized data, same shape.
    """
    return ((windows - mu[np.newaxis, np.newaxis, :]) /
            sigma[np.newaxis, np.newaxis, :]).astype(np.float32)


def save_normalization_stats(
    mu: np.ndarray,
    sigma: np.ndarray,
    output_path: Path | str,
) -> None:
    """Save normalization statistics to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), mu=mu, sigma=sigma)
    logger.info("Normalization stats saved to %s", output_path)


def load_normalization_stats(
    path: Path | str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load normalization statistics from disk."""
    data = np.load(str(path))
    return data["mu"], data["sigma"]


def filter_windows_by_subjects(
    signals: np.ndarray,
    subject_ids: np.ndarray,
    target_subjects: np.ndarray,
) -> np.ndarray:
    """Filter windows belonging to specific subjects.

    Parameters
    ----------
    signals : np.ndarray
        All signals, shape ``(N, T, D)``.
    subject_ids : np.ndarray
        Subject ID per window, shape ``(N,)``.
    target_subjects : np.ndarray
        Subject IDs to keep.

    Returns
    -------
    np.ndarray
        Filtered signals.
    """
    mask = np.isin(subject_ids, target_subjects)
    return signals[mask]


class WindowDataset(Dataset):
    """Shared dataset class for both ANN and SNN.

    Yields tuples ``(x, mask, seed_w)`` in a fixed, reproducible order.
    During measurement, iteration is sequential (no shuffling).

    Implements D6 from the specification.
    """

    def __init__(
        self,
        windows: np.ndarray,
        masking_policy: str = "future_block",
        split_seed: int = 42,
        context_fraction: float = 0.75,
        target_probability: float = 0.25,
        n_blocks: int = 2,
        block_length_fraction: float = 0.125,
        min_gap_fraction: float = 0.0625,
    ) -> None:
        """
        Parameters
        ----------
        windows : np.ndarray
            Normalized windows, shape ``(N, T, D)``.
        masking_policy : str
            One of ``"future_block"``, ``"random_drop"``, ``"multi_target"``.
        split_seed : int
            Seed for deterministic mask generation.
        context_fraction : float
            For future_block policy.
        target_probability : float
            For random_drop policy.
        n_blocks : int
            For multi_target policy.
        block_length_fraction : float
            For multi_target policy.
        min_gap_fraction : float
            For multi_target policy.
        """
        self.windows = windows
        self.T = windows.shape[1]
        self.masking_policy = masking_policy
        self.split_seed = split_seed
        self.context_fraction = context_fraction
        self.target_probability = target_probability
        self.n_blocks = n_blocks
        self.block_length_fraction = block_length_fraction
        self.min_gap_fraction = min_gap_fraction

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get a single window with its mask.

        Returns
        -------
        x : torch.Tensor
            Input window, shape ``(T, D)``.
        mask : torch.Tensor
            Binary mask, shape ``(T,)``. 1 = target, 0 = context.
        seed_w : int
            Per-window seed for reproducibility.
        """
        x = torch.from_numpy(self.windows[idx])  # (T, D)

        mask, _, _ = generate_mask(
            policy=self.masking_policy,
            T=self.T,
            split_seed=self.split_seed,
            window_index=idx,
            context_fraction=self.context_fraction,
            target_probability=self.target_probability,
            n_blocks=self.n_blocks,
            block_length_fraction=self.block_length_fraction,
            min_gap_fraction=self.min_gap_fraction,
        )
        mask = torch.from_numpy(mask)  # (T,)

        # Derive per-window seed
        seed_data = f"{self.split_seed}:{idx}".encode("utf-8")
        seed_w = int(hashlib.sha256(seed_data).hexdigest()[:8], 16)

        return x, mask, seed_w


def prepare_dataset(
    data_dir: Path | str,
    split_seed: int = 42,
    n_train_subjects: int = 21,
    n_val_subjects: int = 4,
    n_test_subjects: int = 5,
    force_download: bool = False,
) -> Dict[str, Any]:
    """Full data preparation pipeline.

    Runs D1--D4: download, split, window, normalize.

    Parameters
    ----------
    data_dir : Path or str
        Root data directory.
    split_seed : int
        Data-split seed.
    n_train_subjects, n_val_subjects, n_test_subjects : int
        Number of subjects per split.
    force_download : bool
        Re-download even if present.

    Returns
    -------
    dict
        Contains ``"train"``, ``"val"``, ``"test"`` arrays (normalized),
        normalization stats, and subject split mapping.
    """
    from data.download import download_uci_har, load_raw_data

    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"

    # D1: Download
    dataset_dir = download_uci_har(raw_dir, force=force_download)

    # Load raw data
    raw = load_raw_data(dataset_dir)

    # Combine train + test from UCI (they have their own split, we re-split)
    all_signals = np.concatenate(
        [raw["train"]["signals"], raw["test"]["signals"]], axis=0
    )
    all_subjects = np.concatenate(
        [raw["train"]["subjects"], raw["test"]["subjects"]], axis=0
    )

    # D2: Subject split
    split_map = subject_split(
        all_subjects,
        n_train=n_train_subjects,
        n_val=n_val_subjects,
        n_test=n_test_subjects,
        split_seed=split_seed,
    )
    split_path = data_dir / "processed" / "subject_split.json"
    save_subject_split(split_map, split_path)

    # Filter by subjects
    train_windows = filter_windows_by_subjects(
        all_signals, all_subjects, split_map["train"]
    )
    val_windows = filter_windows_by_subjects(
        all_signals, all_subjects, split_map["val"]
    )
    test_windows = filter_windows_by_subjects(
        all_signals, all_subjects, split_map["test"]
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_windows), len(val_windows), len(test_windows),
    )

    # D4: Normalization (compute on training set only)
    mu, sigma = compute_normalization_stats(train_windows)
    norm_path = data_dir / "processed" / "norm_stats.npz"
    save_normalization_stats(mu, sigma, norm_path)

    train_norm = normalize(train_windows, mu, sigma)
    val_norm = normalize(val_windows, mu, sigma)
    test_norm = normalize(test_windows, mu, sigma)

    return {
        "train": train_norm,
        "val": val_norm,
        "test": test_norm,
        "mu": mu,
        "sigma": sigma,
        "split_map": split_map,
        "n_train": len(train_norm),
        "n_val": len(val_norm),
        "n_test": len(test_norm),
    }
