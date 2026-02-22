"""Temporal masking strategies for JEPA-style prediction.

Implements three masking policies from Section 2.2 of the specification:
  (i)   Future-block prediction (baseline)
  (ii)  Random drop-in-window
  (iii) Multi-target blocks

All mask indices are generated deterministically from the per-window seed
so that ANN and SNN receive identical masks.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Tuple

import numpy as np


def _derive_seed(split_seed: int, window_index: int) -> int:
    """Derive a per-window deterministic seed.

    seed_w = hash(split_seed, window_index)

    Uses SHA-256 truncated to 32 bits for reproducibility across platforms.
    """
    data = struct.pack(">qq", split_seed, window_index)
    digest = hashlib.sha256(data).digest()
    return struct.unpack(">I", digest[:4])[0]


def future_block_mask(
    T: int,
    context_fraction: float = 0.75,
    split_seed: int = 42,
    window_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Future-block prediction masking (baseline policy).

    Context comprises the first ``context_fraction`` of the window,
    target the remaining portion.

    t0 = floor(context_fraction * T)
    C = {1, ..., t0}, T_set = {t0+1, ..., T}

    Parameters
    ----------
    T : int
        Window length (number of timesteps).
    context_fraction : float
        Fraction of the window used as context.
    split_seed : int
        Data-split seed (for deterministic seed derivation).
    window_index : int
        Index of the current window.

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape ``(T,)``. ``mask[t] = 1`` iff ``t`` is a target.
    context_indices : np.ndarray
        Sorted array of context timestep indices.
    target_indices : np.ndarray
        Sorted array of target timestep indices.
    """
    t0 = int(np.floor(context_fraction * T))
    mask = np.zeros(T, dtype=np.float32)
    mask[t0:] = 1.0
    context_indices = np.arange(0, t0, dtype=np.int64)
    target_indices = np.arange(t0, T, dtype=np.int64)
    return mask, context_indices, target_indices


def random_drop_mask(
    T: int,
    target_probability: float = 0.25,
    split_seed: int = 42,
    window_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random drop-in-window masking.

    Each timestep is independently assigned to the target set with
    probability ``target_probability``.

    Parameters
    ----------
    T : int
        Window length.
    target_probability : float
        Probability of each timestep being a target.
    split_seed : int
        Data-split seed.
    window_index : int
        Index of the current window.

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape ``(T,)``.
    context_indices : np.ndarray
        Sorted context indices.
    target_indices : np.ndarray
        Sorted target indices.
    """
    seed = _derive_seed(split_seed, window_index)
    rng = np.random.RandomState(seed)
    is_target = rng.rand(T) < target_probability
    # Ensure at least one context and one target
    if not np.any(is_target):
        is_target[T - 1] = True
    if np.all(is_target):
        is_target[0] = False
    mask = is_target.astype(np.float32)
    context_indices = np.where(~is_target)[0].astype(np.int64)
    target_indices = np.where(is_target)[0].astype(np.int64)
    return mask, context_indices, target_indices


def multi_target_mask(
    T: int,
    n_blocks: int = 2,
    block_length_fraction: float = 0.125,
    min_gap_fraction: float = 0.0625,
    split_seed: int = 42,
    window_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-target blocks masking.

    Two contiguous target blocks of length ``floor(block_length_fraction * T)``
    each, placed uniformly at random with a minimum gap of
    ``floor(min_gap_fraction * T)`` timesteps between them.

    Parameters
    ----------
    T : int
        Window length.
    n_blocks : int
        Number of target blocks.
    block_length_fraction : float
        Fraction of T for each block length.
    min_gap_fraction : float
        Fraction of T for minimum gap between blocks.
    split_seed : int
        Data-split seed.
    window_index : int
        Index of the current window.

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape ``(T,)``.
    context_indices : np.ndarray
        Sorted context indices.
    target_indices : np.ndarray
        Sorted target indices.
    """
    block_len = int(np.floor(block_length_fraction * T))
    min_gap = int(np.floor(min_gap_fraction * T))
    seed = _derive_seed(split_seed, window_index)
    rng = np.random.RandomState(seed)

    mask = np.zeros(T, dtype=np.float32)

    # Total space needed: n_blocks * block_len + (n_blocks - 1) * min_gap
    total_needed = n_blocks * block_len + (n_blocks - 1) * min_gap
    if total_needed > T:
        raise ValueError(
            f"Cannot fit {n_blocks} blocks of length {block_len} with "
            f"min gap {min_gap} in window of length {T}"
        )

    # Place blocks greedily with random offsets
    slack = T - total_needed
    # Distribute slack among (n_blocks + 1) gaps (before, between, after)
    # Sample n_blocks + 1 non-negative integers summing to slack
    if slack > 0:
        cuts = sorted(rng.choice(slack + n_blocks, n_blocks, replace=False))
        gaps = []
        prev = 0
        for i, c in enumerate(cuts):
            gaps.append(c - prev - i)
            prev = c
        gaps.append(slack + n_blocks - 1 - prev - (n_blocks - 1))
        # Fix: use simpler distribution
        gaps = []
        raw = rng.dirichlet(np.ones(n_blocks + 1))
        raw_int = np.floor(raw * slack).astype(int)
        remainder = slack - raw_int.sum()
        for i in range(int(remainder)):
            raw_int[i % (n_blocks + 1)] += 1
        gaps = raw_int.tolist()
    else:
        gaps = [0] * (n_blocks + 1)

    # Ensure minimum gap between blocks
    pos = gaps[0]
    block_starts = []
    for i in range(n_blocks):
        block_starts.append(pos)
        mask[pos: pos + block_len] = 1.0
        pos += block_len
        if i < n_blocks - 1:
            pos += min_gap + gaps[i + 1]

    context_indices = np.where(mask == 0)[0].astype(np.int64)
    target_indices = np.where(mask == 1)[0].astype(np.int64)
    return mask, context_indices, target_indices


def generate_mask(
    policy: str,
    T: int,
    split_seed: int = 42,
    window_index: int = 0,
    context_fraction: float = 0.75,
    target_probability: float = 0.25,
    n_blocks: int = 2,
    block_length_fraction: float = 0.125,
    min_gap_fraction: float = 0.0625,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a mask according to the specified policy.

    This is the unified entry point used by both ANN and SNN pipelines.
    Deterministic output for identical inputs is guaranteed.

    Parameters
    ----------
    policy : str
        One of ``"future_block"``, ``"random_drop"``, ``"multi_target"``.
    T : int
        Window length.
    split_seed : int
        Data-split seed.
    window_index : int
        Window index for per-window seed derivation.
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

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape ``(T,)``, ``mask[t] = 1`` iff target.
    context_indices : np.ndarray
        Sorted context indices.
    target_indices : np.ndarray
        Sorted target indices.
    """
    if policy == "future_block":
        return future_block_mask(T, context_fraction, split_seed, window_index)
    elif policy == "random_drop":
        return random_drop_mask(T, target_probability, split_seed, window_index)
    elif policy == "multi_target":
        return multi_target_mask(
            T, n_blocks, block_length_fraction, min_gap_fraction,
            split_seed, window_index,
        )
    else:
        raise ValueError(f"Unknown masking policy: {policy}")
