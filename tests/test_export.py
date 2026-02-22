"""Acceptance tests: export round-trip tests T-E1 through T-E3.

T-E1: Spike-train round-trip -- PyTorch LIF and Lava-equivalent LIF
      produce <= 1 spike mismatch per neuron over 1000 steps.
T-E2: Trace round-trip -- spike traces from both frameworks agree
      with max absolute difference < 1e-4.
T-E3: Readout round-trip -- z_hat values agree with max abs diff < 1e-3.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snn.model import LIFNeuronLayer, SpikeTraceComputer


# -----------------------------------------------------------------------
# Helpers: reference LIF implementation matching Lava floating-point mode
# -----------------------------------------------------------------------

def _run_pytorch_lif(
    input_current: torch.Tensor,
    weight: torch.Tensor,
    beta: float,
    v_th: float,
    T: int,
    H: int,
) -> torch.Tensor:
    """Run PyTorch LIF neurons and return spike tensor (T, H).

    Uses the same discrete LIF dynamics as snn/model.py:
      v[t] = beta * v[t-1] * (1 - s[t-1]) + I[t]
      s[t] = Heaviside(v[t] - v_th)
    """
    lif = LIFNeuronLayer(n_neurons=H, beta=beta, v_th=v_th, surrogate_k=25.0)
    lif.eval()

    # Input current through a linear layer
    # input_current: (1, T, D) -> projected: (1, T, H)
    D = input_current.shape[2]
    proj = torch.nn.Linear(D, H, bias=False)
    with torch.no_grad():
        proj.weight.copy_(weight)  # (H, D)

    projected = proj(input_current)  # (1, T, H)
    spikes, membrane, _ = lif(projected)  # (1, T, H) each + v_final

    return spikes.squeeze(0)  # (T, H)


def _run_reference_lif_numpy(
    input_current: np.ndarray,
    weight: np.ndarray,
    beta: float,
    v_th: float,
    T: int,
    H: int,
) -> np.ndarray:
    """Run reference LIF in NumPy (matching Lava floating-pt semantics).

    Parameters
    ----------
    input_current : np.ndarray
        Shape ``(T, D)``.
    weight : np.ndarray
        Shape ``(H, D)``, same as Lava Dense weight.
    beta, v_th : float
        LIF parameters.
    T, H : int
        Timesteps and hidden dim.

    Returns
    -------
    np.ndarray
        Spike tensor, shape ``(T, H)``.
    """
    D = input_current.shape[1]
    v = np.zeros(H, dtype=np.float64)
    spikes = np.zeros((T, H), dtype=np.float64)

    for t in range(T):
        # Input current through Dense
        I_t = weight @ input_current[t]  # (H,)
        # Hard reset: multiply by (1 - s[t-1])
        if t > 0:
            v = beta * v * (1.0 - spikes[t - 1]) + I_t
        else:
            v = I_t.copy()
        # Spike
        spikes[t] = (v >= v_th).astype(np.float64)

    return spikes


# -----------------------------------------------------------------------
# T-E1: Spike-train round-trip
# -----------------------------------------------------------------------

class TestE1SpikeTrainRoundTrip:
    """T-E1: PyTorch LIF and reference NumPy LIF (matching Lava floating-pt)
    produce at most 1 spike mismatch per neuron over 1000 steps."""

    def test_spike_roundtrip(self) -> None:
        T = 1000
        D = 6
        H = 128
        beta = 0.904837
        v_th = 1.0

        rng = np.random.default_rng(42)
        input_np = rng.standard_normal((T, D)).astype(np.float32) * 0.3
        weight_np = rng.standard_normal((H, D)).astype(np.float32) * 0.1

        # PyTorch path
        input_pt = torch.from_numpy(input_np).unsqueeze(0)  # (1, T, D)
        weight_pt = torch.from_numpy(weight_np)  # (H, D)
        spikes_pt = _run_pytorch_lif(input_pt, weight_pt, beta, v_th, T, H)
        spikes_pt_np = spikes_pt.detach().numpy()

        # NumPy reference path
        spikes_ref = _run_reference_lif_numpy(
            input_np.astype(np.float64),
            weight_np.astype(np.float64),
            beta, v_th, T, H,
        )

        # Compare: at most 1 mismatch per neuron
        mismatches_per_neuron = np.sum(
            np.abs(spikes_pt_np - spikes_ref) > 0.5, axis=0,
        )  # (H,)

        max_mismatches = int(np.max(mismatches_per_neuron))
        assert max_mismatches <= 1, (
            f"Spike mismatch: max {max_mismatches} mismatches per neuron "
            f"(threshold: 1). "
            f"Mean mismatches: {np.mean(mismatches_per_neuron):.2f}"
        )


# -----------------------------------------------------------------------
# T-E2: Trace round-trip
# -----------------------------------------------------------------------

class TestE2TraceRoundTrip:
    """T-E2: Spike traces from PyTorch and NumPy reference agree
    with max absolute difference < 1e-4."""

    def test_trace_roundtrip(self) -> None:
        T = 1000
        H = 128
        alpha = 0.9

        rng = np.random.default_rng(42)
        # Use a deterministic spike train
        spikes_np = (rng.random((T, H)) > 0.95).astype(np.float32)

        # PyTorch path
        tracer = SpikeTraceComputer(alpha=alpha)
        spikes_pt = torch.from_numpy(spikes_np).unsqueeze(0)  # (1, T, H)
        trace_pt = tracer(spikes_pt).squeeze(0).detach().numpy()  # (T, H)

        # NumPy reference
        trace_ref = np.zeros((T, H), dtype=np.float64)
        for t in range(T):
            if t == 0:
                trace_ref[t] = (1.0 - alpha) * spikes_np[t]
            else:
                trace_ref[t] = alpha * trace_ref[t - 1] + (1.0 - alpha) * spikes_np[t]

        max_diff = float(np.max(np.abs(trace_pt.astype(np.float64) - trace_ref)))
        assert max_diff < 1e-4, (
            f"Trace mismatch: max absolute difference = {max_diff:.2e} "
            f"(threshold: 1e-4)"
        )


# -----------------------------------------------------------------------
# T-E3: Readout round-trip
# -----------------------------------------------------------------------

class TestE3ReadoutRoundTrip:
    """T-E3: Readout z_hat = W_readout @ trace + b_readout agrees
    between PyTorch and NumPy with max abs diff < 1e-3."""

    def test_readout_roundtrip(self) -> None:
        T = 100
        H = 128
        alpha = 0.9

        rng = np.random.default_rng(42)
        spikes_np = (rng.random((T, H)) > 0.92).astype(np.float32)

        # Compute trace
        trace_np = np.zeros((T, H), dtype=np.float64)
        for t in range(T):
            if t == 0:
                trace_np[t] = spikes_np[t]
            else:
                trace_np[t] = alpha * trace_np[t - 1] + spikes_np[t]

        # Random readout weights
        W = rng.standard_normal((H, H)).astype(np.float32) * 0.1
        b = rng.standard_normal(H).astype(np.float32) * 0.01

        # NumPy readout
        z_hat_np = trace_np @ W.T + b  # (T, H)

        # PyTorch readout
        readout = torch.nn.Linear(H, H)
        with torch.no_grad():
            readout.weight.copy_(torch.from_numpy(W))
            readout.bias.copy_(torch.from_numpy(b))

        trace_pt = torch.from_numpy(trace_np.astype(np.float32))
        z_hat_pt = readout(trace_pt).detach().numpy()  # (T, H)

        max_diff = float(np.max(np.abs(z_hat_pt.astype(np.float64) - z_hat_np)))
        assert max_diff < 1e-3, (
            f"Readout mismatch: max absolute difference = {max_diff:.2e} "
            f"(threshold: 1e-3)"
        )
