"""JEPA-time loss function.

Implements Eq. (5) from the specification:
    L_JEPA-time(theta, phi) = (1/|T|) * sum_{t in T} d(z_hat_t, sg(z_t^(t)))

where d(.,.) is MSE and sg(.) is the stop-gradient operator.
Both ANN and SNN use this identical loss function.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_time_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the JEPA-time prediction loss.

    Parameters
    ----------
    predictions : torch.Tensor
        Predictor outputs, shape ``(B, T, H)``.
    targets : torch.Tensor
        Teacher encoder outputs (stop-gradient applied externally),
        shape ``(B, T, H)``.
    mask : torch.Tensor
        Binary mask, shape ``(B, T)``. ``mask[b, t] = 1`` iff ``t`` is a
        target timestep.

    Returns
    -------
    torch.Tensor
        Scalar loss: mean squared error averaged over target positions.

    Notes
    -----
    The stop-gradient operator must be applied to ``targets`` before
    calling this function (i.e., ``targets = targets.detach()``).
    """
    # Expand mask to (B, T, 1) for broadcasting
    mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)

    # Compute per-element squared error
    sq_error = (predictions - targets) ** 2  # (B, T, H)

    # Mask: only target positions contribute
    masked_sq_error = sq_error * mask_expanded  # (B, T, H)

    # Sum over H dimension, then average over target positions
    per_position_loss = masked_sq_error.sum(dim=-1)  # (B, T)

    # Count target positions per batch element
    n_targets = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)

    # Average over target positions, then over batch
    loss = (per_position_loss.sum(dim=-1) / n_targets.squeeze(-1)).mean()

    return loss


def jepa_time_loss_numpy(
    predictions,
    targets,
    mask,
) -> float:
    """Compute the JEPA-time loss using NumPy (for Lava inference).

    Parameters
    ----------
    predictions : np.ndarray
        Shape ``(T, H)`` or ``(B, T, H)``.
    targets : np.ndarray
        Shape ``(T, H)`` or ``(B, T, H)``.
    mask : np.ndarray
        Shape ``(T,)`` or ``(B, T)``.

    Returns
    -------
    float
        Scalar loss value.
    """
    import numpy as np

    if predictions.ndim == 2:
        predictions = predictions[np.newaxis]
        targets = targets[np.newaxis]
        mask = mask[np.newaxis]

    mask_expanded = mask[:, :, np.newaxis]  # (B, T, 1)
    sq_error = (predictions - targets) ** 2  # (B, T, H)
    masked_sq_error = sq_error * mask_expanded
    per_position_loss = masked_sq_error.sum(axis=-1)  # (B, T)
    n_targets = mask.sum(axis=-1, keepdims=True).clip(min=1.0)
    loss = (per_position_loss.sum(axis=-1) / n_targets.squeeze(-1)).mean()
    return float(loss)
