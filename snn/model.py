"""SNN model for LavaJEPA (PyTorch training version).

Implements the SNN baseline from the specification:
  - Recurrent LIF population, H=128 neurons
  - Surrogate-gradient training (fast sigmoid, k=25)
  - Spike trace: exponential low-pass filter (Eq. 9)
  - Readout: linear projection from spike trace (Eq. 10)
  - Predictor: 2-layer spiking/hybrid MLP
  - Teacher: EMA copy of encoder weights

Neuron model (Eq. 14, Section 5.6):
  V_i[t] = beta * V_i[t-1] * (1 - s_i[t-1]) + I_i[t]
  s_i[t] = Theta(V_i[t] - v_th)

Surrogate gradient: Fast sigmoid
  Theta'(x) = 1 / (1 + k|x|)^2, k=25
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.positional import sinusoidal_position_encoding


class FastSigmoidSurrogate(torch.autograd.Function):
    """Fast sigmoid surrogate gradient for the Heaviside step function.

    Forward: Heaviside(x - v_th)
    Backward: 1 / (1 + k * |x - v_th|)^2
    """

    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, v_th: float, k: float) -> torch.Tensor:
        ctx.save_for_backward(membrane_potential)
        ctx.v_th = v_th
        ctx.k = k
        return (membrane_potential >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane_potential,) = ctx.saved_tensors
        v_shifted = membrane_potential - ctx.v_th
        grad = 1.0 / (1.0 + ctx.k * torch.abs(v_shifted)) ** 2
        return grad_output * grad, None, None


def spike_function(
    membrane_potential: torch.Tensor,
    v_th: float = 1.0,
    k: float = 25.0,
) -> torch.Tensor:
    """Apply spike function with surrogate gradient.

    Parameters
    ----------
    membrane_potential : torch.Tensor
        Membrane potential.
    v_th : float
        Firing threshold.
    k : float
        Surrogate gradient steepness.

    Returns
    -------
    torch.Tensor
        Binary spike tensor.
    """
    return FastSigmoidSurrogate.apply(membrane_potential, v_th, k)


class LIFNeuronLayer(nn.Module):
    """Leaky Integrate-and-Fire neuron layer with surrogate gradients.

    Implements discrete-time LIF dynamics (Eq. 14):
      V_i[t] = beta * V_i[t-1] * (1 - s_i[t-1]) + I_i[t]
      s_i[t] = Theta(V_i[t] - v_th)

    With hard reset (multiplicative: V -> 0 on spike).

    Parameters
    ----------
    n_neurons : int
        Number of neurons (H).
    beta : float
        Membrane decay constant, exp(-dt/tau_mem).
    v_th : float
        Firing threshold.
    surrogate_k : float
        Surrogate gradient steepness.
    """

    def __init__(
        self,
        n_neurons: int,
        beta: float = 0.904837,
        v_th: float = 1.0,
        surrogate_k: float = 25.0,
    ) -> None:
        super().__init__()
        self.n_neurons = n_neurons
        self.beta = beta
        self.v_th = v_th
        self.surrogate_k = surrogate_k

    def forward(
        self,
        input_current: torch.Tensor,
        v_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a sequence of input currents.

        Parameters
        ----------
        input_current : torch.Tensor
            Input currents, shape ``(B, T, H)``.
        v_init : torch.Tensor, optional
            Initial membrane potential, shape ``(B, H)``.
            Defaults to zeros.

        Returns
        -------
        spikes : torch.Tensor
            Spike outputs, shape ``(B, T, H)``.
        membrane : torch.Tensor
            Membrane potentials, shape ``(B, T, H)``.
        v_final : torch.Tensor
            Final membrane potential, shape ``(B, H)``.
        """
        B, T, H = input_current.shape

        if v_init is None:
            v = torch.zeros(B, H, device=input_current.device, dtype=input_current.dtype)
        else:
            v = v_init

        spikes_list = []
        membrane_list = []
        s_prev = torch.zeros_like(v)

        for t in range(T):
            # LIF dynamics (Eq. 14)
            # Hard reset: multiply by (1 - s_prev)
            v = self.beta * v * (1.0 - s_prev) + input_current[:, t, :]

            # Spike generation with surrogate gradient
            s = spike_function(v, self.v_th, self.surrogate_k)

            spikes_list.append(s)
            membrane_list.append(v)
            s_prev = s

        spikes = torch.stack(spikes_list, dim=1)     # (B, T, H)
        membrane = torch.stack(membrane_list, dim=1)  # (B, T, H)

        return spikes, membrane, v


class SpikeTraceComputer(nn.Module):
    """Exponential low-pass filter for spike traces.

    Implements Eq. (9):
      r_t = alpha * r_{t-1} + (1 - alpha) * s_t

    Parameters
    ----------
    alpha : float
        Trace decay constant.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Compute spike traces from spike trains.

        Parameters
        ----------
        spikes : torch.Tensor
            Binary spike tensor, shape ``(B, T, H)``.

        Returns
        -------
        torch.Tensor
            Spike traces, shape ``(B, T, H)``.
        """
        B, T, H = spikes.shape
        r = torch.zeros(B, H, device=spikes.device, dtype=spikes.dtype)
        traces = []

        for t in range(T):
            r = self.alpha * r + (1.0 - self.alpha) * spikes[:, t, :]
            traces.append(r)

        return torch.stack(traces, dim=1)  # (B, T, H)


class SNNEncoder(nn.Module):
    """SNN encoder with input projection + LIF neurons + spike trace + readout.

    Architecture:
      Input (B, T, D) -> Dense (D -> H) -> LIF (H neurons) ->
      Spike Trace -> Readout (H -> H) -> Latent (B, T, H)

    Parameters
    ----------
    input_dim : int
        Input dimension D.
    hidden_dim : int
        Hidden/latent dimension H.
    beta : float
        LIF membrane decay.
    v_th : float
        LIF firing threshold.
    surrogate_k : float
        Surrogate gradient steepness.
    trace_alpha : float
        Spike trace decay.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        beta: float = 0.904837,
        v_th: float = 1.0,
        surrogate_k: float = 25.0,
        trace_alpha: float = 0.9,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection: Dense (D -> H)
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # LIF neuron layer
        self.lif = LIFNeuronLayer(hidden_dim, beta, v_th, surrogate_k)

        # Spike trace computer
        self.trace = SpikeTraceComputer(trace_alpha)

        # Readout: linear projection from spike trace to latent space (Eq. 10)
        self.readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input through SNN.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape ``(B, T, D)``.
        context_mask : torch.Tensor, optional
            Binary mask, shape ``(B, T)``. Target positions (mask=1) are
            zeroed before feeding the encoder.

        Returns
        -------
        latents : torch.Tensor
            Readout latents z_t = W_ro * r_t + b_ro, shape ``(B, T, H)``.
        spikes : torch.Tensor
            Spike outputs, shape ``(B, T, H)``.
        traces : torch.Tensor
            Spike traces, shape ``(B, T, H)``.
        membrane : torch.Tensor
            Membrane potentials, shape ``(B, T, H)``.
        """
        if context_mask is not None:
            gate = (1.0 - context_mask.unsqueeze(-1))
            x = x * gate

        # Input projection
        current = self.input_proj(x)  # (B, T, H)

        # LIF dynamics
        spikes, membrane, _ = self.lif(current)

        # Spike trace
        traces = self.trace(spikes)  # (B, T, H)

        # Readout (Eq. 10)
        latents = self.readout(traces)  # (B, T, H)

        return latents, spikes, traces, membrane


class SNNPredictor(nn.Module):
    """2-layer spiking/hybrid MLP predictor.

    Same topology as ANN predictor but uses LIF neurons.

    Input: student readout + positional encoding + mask signal
    Architecture: (H + H) -> LIF(H) -> Dense(H -> H)

    Parameters
    ----------
    hidden_dim : int
        Latent dimension H.
    max_seq_len : int
        Maximum sequence length T.
    beta : float
        LIF membrane decay.
    v_th : float
        LIF firing threshold.
    surrogate_k : float
        Surrogate gradient steepness.
    trace_alpha : float
        Spike trace decay.
    c_mask : float
        Mask bias current magnitude.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 128,
        beta: float = 0.904837,
        v_th: float = 1.0,
        surrogate_k: float = 25.0,
        trace_alpha: float = 0.9,
        c_mask: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.c_mask = c_mask

        # Predictor layer 1: Dense (2H -> H) -> LIF
        self.pred_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.lif_pred = LIFNeuronLayer(hidden_dim, beta, v_th, surrogate_k)
        self.trace_pred = SpikeTraceComputer(trace_alpha)

        # Predictor layer 2: Dense readout (H -> H)
        self.pred_readout = nn.Linear(hidden_dim, hidden_dim)

        # Mask weight: broadcasts scalar c_mask to all H neurons
        # SNN analogue of ANN's learned [MASK] embedding
        self.mask_weight = nn.Linear(1, hidden_dim, bias=False)

        # Pre-compute positional encodings
        self.register_buffer(
            "pos_encoding",
            sinusoidal_position_encoding(max_seq_len, hidden_dim),
        )

    def forward(
        self,
        encoder_latents: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict target latents.

        Parameters
        ----------
        encoder_latents : torch.Tensor
            Encoder readout, shape ``(B, T, H)``.
        mask : torch.Tensor
            Binary mask, shape ``(B, T)``. 1 = target.

        Returns
        -------
        predictions : torch.Tensor
            Predicted latents, shape ``(B, T, H)``.
        pred_spikes : torch.Tensor
            Predictor spike outputs, shape ``(B, T, H)``.
        pred_traces : torch.Tensor
            Predictor spike traces, shape ``(B, T, H)``.
        """
        B, T, H = encoder_latents.shape

        # Positional signal: identical numerical values to ANN
        pos = self.pos_encoding[:T].unsqueeze(0).expand(B, -1, -1)  # (B, T, H)

        # Mask signal: deterministic constant bias current
        mask_signal = mask.unsqueeze(-1) * self.c_mask  # (B, T, 1)
        mask_current = self.mask_weight(mask_signal)     # (B, T, H)

        # Combine encoder latents with positional encoding
        combined = torch.cat([encoder_latents, pos], dim=-1)  # (B, T, 2H)

        # Predictor layer 1: Dense -> add mask current -> LIF
        pred_current = self.pred_proj(combined)  # (B, T, H)
        pred_current = pred_current + mask_current  # Add mask signal

        pred_spikes, _, _ = self.lif_pred(pred_current)
        pred_traces = self.trace_pred(pred_spikes)

        # Predictor layer 2: Dense readout
        predictions = self.pred_readout(pred_traces)  # (B, T, H)

        return predictions, pred_spikes, pred_traces


class SNNModel(nn.Module):
    """Complete SNN JEPA model with encoder, predictor, and EMA teacher.

    Parameters
    ----------
    input_dim : int
        Input feature dimension D.
    hidden_dim : int
        Latent dimension H.
    max_seq_len : int
        Maximum sequence length T.
    tau_ema : float
        EMA momentum for teacher update.
    beta : float
        LIF membrane decay.
    v_th : float
        Firing threshold.
    surrogate_k : float
        Surrogate gradient steepness.
    trace_alpha : float
        Spike trace decay.
    c_mask : float
        Mask bias current magnitude.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        max_seq_len: int = 128,
        tau_ema: float = 0.996,
        beta: float = 0.904837,
        v_th: float = 1.0,
        surrogate_k: float = 25.0,
        trace_alpha: float = 0.9,
        c_mask: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_ema = tau_ema

        # Student encoder
        self.encoder = SNNEncoder(
            input_dim, hidden_dim, beta, v_th, surrogate_k, trace_alpha,
        )

        # Predictor
        self.predictor = SNNPredictor(
            hidden_dim, max_seq_len, beta, v_th, surrogate_k, trace_alpha, c_mask,
        )

        # Teacher encoder (EMA copy, no gradients)
        self.teacher_encoder = SNNEncoder(
            input_dim, hidden_dim, beta, v_th, surrogate_k, trace_alpha,
        )
        self._copy_encoder_to_teacher()
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

    def _copy_encoder_to_teacher(self) -> None:
        """Copy encoder parameters to teacher."""
        for tp, sp in zip(
            self.teacher_encoder.parameters(),
            self.encoder.parameters(),
        ):
            tp.data.copy_(sp.data)

    @torch.no_grad()
    def update_teacher(self) -> None:
        """Update teacher parameters by EMA (Eq. 6)."""
        for tp, sp in zip(
            self.teacher_encoder.parameters(),
            self.encoder.parameters(),
        ):
            tp.data.mul_(self.tau_ema).add_(sp.data, alpha=1.0 - self.tau_ema)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the complete SNN JEPA model.

        Parameters
        ----------
        x : torch.Tensor
            Input time-series, shape ``(B, T, D)``.
        mask : torch.Tensor
            Binary mask, shape ``(B, T)``. 1 = target.

        Returns
        -------
        predictions : torch.Tensor
            Predictor output latents, shape ``(B, T, H)``.
        teacher_targets : torch.Tensor
            Teacher encoder latents (detached), shape ``(B, T, H)``.
        enc_spikes : torch.Tensor
            Encoder spike outputs, shape ``(B, T, H)``.
        pred_spikes : torch.Tensor
            Predictor spike outputs, shape ``(B, T, H)``.
        enc_traces : torch.Tensor
            Encoder spike traces, shape ``(B, T, H)``.
        """
        # Student encoder
        student_latents, enc_spikes, enc_traces, enc_membrane = self.encoder(
            x, context_mask=mask,
        )

        # Predictor
        predictions, pred_spikes, pred_traces = self.predictor(
            student_latents, mask,
        )

        # Teacher encoder (stop-gradient)
        with torch.no_grad():
            teacher_latents, _, _, _ = self.teacher_encoder(x)

        return predictions, teacher_latents, enc_spikes, pred_spikes, enc_traces

    def get_encoder_outputs(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get student encoder outputs for observability.

        Returns latents, spikes, traces, membrane potentials.
        """
        return self.encoder(x, context_mask=mask)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_spike_stats(
        self,
        enc_spikes: torch.Tensor,
        pred_spikes: torch.Tensor,
    ) -> dict:
        """Compute spike statistics for event metrics.

        Parameters
        ----------
        enc_spikes : torch.Tensor
            Encoder spikes, shape ``(B, T, H)``.
        pred_spikes : torch.Tensor
            Predictor spikes, shape ``(B, T, H)``.

        Returns
        -------
        dict
            Spike count, spike rate, etc.
        """
        total_spikes = int(enc_spikes.sum().item() + pred_spikes.sum().item())
        n_neurons = enc_spikes.shape[-1] + pred_spikes.shape[-1]
        n_steps = enc_spikes.shape[1]
        spike_rate = total_spikes / (n_neurons * n_steps) if n_neurons * n_steps > 0 else 0.0

        return {
            "total_spikes": total_spikes,
            "enc_spikes": int(enc_spikes.sum().item()),
            "pred_spikes": int(pred_spikes.sum().item()),
            "spike_rate": spike_rate,
            "n_neurons": n_neurons,
            "n_steps": n_steps,
        }
