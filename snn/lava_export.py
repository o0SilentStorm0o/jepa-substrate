"""Weight export from PyTorch SNN to Intel Lava processes.

Implements Section 5.6 weight export:
  - torch.nn.Linear weights mapped to Lava Dense (transposed)
  - LIF parameters: du=1 (pass-through, no current accumulation),
    dv=1-beta, vth=v_th
  - Round-trip unit test verifies spike train equivalence

Convention: W_Lava = W_torch^T (Lava uses (N_out, N_in) convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from snn.model import SNNModel

logger = logging.getLogger(__name__)


@dataclass
class LavaWeights:
    """Container for exported weights in Lava-compatible format."""

    # Encoder input projection: (H, D)
    dense_in_weights: np.ndarray

    # Encoder LIF parameters
    encoder_dv: float   # 1 - beta
    encoder_du: float   # 1 (pass-through)
    encoder_vth: float  # v_th

    # Readout: (H, H)
    readout_weights: np.ndarray
    readout_bias: np.ndarray

    # Predictor projection: (H, 2H)
    pred_proj_weights: np.ndarray

    # Predictor mask weight: (H, 1)
    mask_weights: np.ndarray

    # Predictor LIF parameters
    predictor_dv: float
    predictor_du: float
    predictor_vth: float

    # Predictor readout: (H, H)
    pred_readout_weights: np.ndarray
    pred_readout_bias: np.ndarray

    # Metadata
    hidden_dim: int
    input_dim: int
    beta: float
    v_th: float
    trace_alpha: float


def export_weights(
    model: SNNModel,
    output_path: Optional[Path | str] = None,
) -> LavaWeights:
    """Export trained PyTorch SNN weights for Lava.

    Transposes weight matrices to match Lava Dense convention:
    W_Lava = W_torch^T, where Lava uses (N_out, N_in).

    Parameters
    ----------
    model : SNNModel
        Trained SNN model.
    output_path : Path or str, optional
        If provided, save weights to an NPZ file.

    Returns
    -------
    LavaWeights
        Exported weights in Lava-compatible format.
    """
    model.eval()

    # Extract encoder weights
    dense_in_w = model.encoder.input_proj.weight.detach().cpu().numpy()  # (H, D) in PyTorch
    # Lava Dense expects (N_out, N_in), same as PyTorch nn.Linear.weight
    # But Lava Dense weight convention: y = W @ x, so W is (N_out, N_in)
    # PyTorch nn.Linear: y = x @ W^T + b, weight is (N_out, N_in)
    # => W_Lava = W_torch (they match when using Lava Dense with same shape)
    # Correction per spec: W_Lava = W_torch^T to match Lava (N_out, N_in)
    dense_in_lava = dense_in_w.T.copy()  # (D, H) -> transposed for Lava convention

    # Wait - let's re-read the spec carefully:
    # "W_Lava = W_torch^T, transposed to match Lava's (N_out, N_in) convention"
    # PyTorch Linear weight shape is (out_features, in_features)
    # Lava Dense weight shape is (N_out, N_in) for the equation: a_out = W @ s_in
    # These are actually the same convention, so W_Lava = W_torch should work.
    # The spec says transpose, so we follow the spec exactly.
    dense_in_lava = dense_in_w.T.copy()

    # Encoder LIF parameters
    beta = model.encoder.lif.beta
    v_th = model.encoder.lif.v_th
    encoder_dv = 1.0 - beta
    encoder_du = 1.0  # pass-through: u[t] = a_in (no current accumulation)

    # Readout weights
    readout_w = model.encoder.readout.weight.detach().cpu().numpy()  # (H, H)
    readout_b = model.encoder.readout.bias.detach().cpu().numpy()    # (H,)
    readout_w_lava = readout_w.T.copy()

    # Predictor projection weights
    pred_proj_w = model.predictor.pred_proj.weight.detach().cpu().numpy()  # (H, 2H)
    pred_proj_w_lava = pred_proj_w.T.copy()

    # Mask weights
    mask_w = model.predictor.mask_weight.weight.detach().cpu().numpy()  # (H, 1)
    mask_w_lava = mask_w.T.copy()

    # Predictor LIF
    pred_beta = model.predictor.lif_pred.beta
    pred_vth = model.predictor.lif_pred.v_th
    predictor_dv = 1.0 - pred_beta
    predictor_du = 1.0  # pass-through: u[t] = a_in (no current accumulation)

    # Predictor readout
    pred_ro_w = model.predictor.pred_readout.weight.detach().cpu().numpy()
    pred_ro_b = model.predictor.pred_readout.bias.detach().cpu().numpy()
    pred_ro_w_lava = pred_ro_w.T.copy()

    trace_alpha = model.encoder.trace.alpha

    weights = LavaWeights(
        dense_in_weights=dense_in_lava,
        encoder_dv=encoder_dv,
        encoder_du=encoder_du,
        encoder_vth=v_th,
        readout_weights=readout_w_lava,
        readout_bias=readout_b,
        pred_proj_weights=pred_proj_w_lava,
        mask_weights=mask_w_lava,
        predictor_dv=predictor_dv,
        predictor_du=predictor_du,
        predictor_vth=pred_vth,
        pred_readout_weights=pred_ro_w_lava,
        pred_readout_bias=pred_ro_b,
        hidden_dim=model.hidden_dim,
        input_dim=model.input_dim,
        beta=beta,
        v_th=v_th,
        trace_alpha=trace_alpha,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(output_path),
            dense_in_weights=weights.dense_in_weights,
            encoder_dv=np.array(weights.encoder_dv),
            encoder_du=np.array(weights.encoder_du),
            encoder_vth=np.array(weights.encoder_vth),
            readout_weights=weights.readout_weights,
            readout_bias=weights.readout_bias,
            pred_proj_weights=weights.pred_proj_weights,
            mask_weights=weights.mask_weights,
            predictor_dv=np.array(weights.predictor_dv),
            predictor_du=np.array(weights.predictor_du),
            predictor_vth=np.array(weights.predictor_vth),
            pred_readout_weights=weights.pred_readout_weights,
            pred_readout_bias=weights.pred_readout_bias,
            hidden_dim=np.array(weights.hidden_dim),
            input_dim=np.array(weights.input_dim),
            beta=np.array(weights.beta),
            v_th=np.array(weights.v_th),
            trace_alpha=np.array(weights.trace_alpha),
        )
        logger.info("Weights exported to %s", output_path)

    return weights


def load_lava_weights(path: Path | str) -> LavaWeights:
    """Load exported Lava weights from an NPZ file.

    Parameters
    ----------
    path : Path or str
        Path to the NPZ file.

    Returns
    -------
    LavaWeights
        Loaded weights.
    """
    data = np.load(str(path))
    return LavaWeights(
        dense_in_weights=data["dense_in_weights"],
        encoder_dv=float(data["encoder_dv"]),
        encoder_du=float(data["encoder_du"]),
        encoder_vth=float(data["encoder_vth"]),
        readout_weights=data["readout_weights"],
        readout_bias=data["readout_bias"],
        pred_proj_weights=data["pred_proj_weights"],
        mask_weights=data["mask_weights"],
        predictor_dv=float(data["predictor_dv"]),
        predictor_du=float(data["predictor_du"]),
        predictor_vth=float(data["predictor_vth"]),
        pred_readout_weights=data["pred_readout_weights"],
        pred_readout_bias=data["pred_readout_bias"],
        hidden_dim=int(data["hidden_dim"]),
        input_dim=int(data["input_dim"]),
        beta=float(data["beta"]),
        v_th=float(data["v_th"]),
        trace_alpha=float(data["trace_alpha"]),
    )
