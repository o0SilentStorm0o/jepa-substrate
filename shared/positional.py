"""Sinusoidal time-position encoding.

Implements Eq. (7) from the specification:
    e_{t,2k}   = sin(t / 10000^{2k/H})
    e_{t,2k+1} = cos(t / 10000^{2k/H})

The same numerical values are used by both ANN and SNN predictors.
"""

from __future__ import annotations

import numpy as np
import torch


def sinusoidal_position_encoding(T: int, H: int) -> torch.Tensor:
    """Compute sinusoidal time-position encodings.

    Parameters
    ----------
    T : int
        Number of timesteps.
    H : int
        Latent dimension (must be even).

    Returns
    -------
    torch.Tensor
        Position encoding matrix of shape ``(T, H)``, dtype ``float32``.
        Row *t* contains ``pos(t)`` as defined in the specification.
    """
    if H % 2 != 0:
        raise ValueError(f"Latent dimension H must be even, got {H}")

    positions = np.arange(T, dtype=np.float64)
    dims = np.arange(0, H, 2, dtype=np.float64)

    # 10000^{2k/H}
    freq_denominator = np.power(10000.0, dims / H)

    # (T, H//2)
    angles = positions[:, np.newaxis] / freq_denominator[np.newaxis, :]

    encoding = np.zeros((T, H), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles).astype(np.float32)
    encoding[:, 1::2] = np.cos(angles).astype(np.float32)

    return torch.from_numpy(encoding)


def sinusoidal_position_encoding_numpy(T: int, H: int) -> np.ndarray:
    """Compute sinusoidal time-position encodings as a NumPy array.

    Same computation as :func:`sinusoidal_position_encoding` but returns
    a NumPy array, suitable for Lava ``RingBuffer`` preloading.

    Parameters
    ----------
    T : int
        Number of timesteps.
    H : int
        Latent dimension (must be even).

    Returns
    -------
    np.ndarray
        Position encoding matrix of shape ``(T, H)``, dtype ``float32``.
    """
    return sinusoidal_position_encoding(T, H).numpy()
