"""ANN baseline model for LavaJEPA.

Implements the ANN factorization from the specification:
  - Encoder: 1-layer GRU, hidden size H=128
  - Predictor: 2-layer MLP (H -> H -> H) with time-position embedding
  - Teacher: EMA copy of encoder, tau=0.996

Latent tap point (Section 5.3):
  ANN: z_t = h_t (GRU hidden state at step t), no projection head.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from shared.positional import sinusoidal_position_encoding


class GRUEncoder(nn.Module):
    """1-layer GRU encoder producing per-timestep latent vectors.

    Parameters
    ----------
    input_dim : int
        Input feature dimension D.
    hidden_dim : int
        Latent dimension H.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input time-series.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape ``(B, T, D)``.
        context_mask : torch.Tensor, optional
            Binary mask, shape ``(B, T)``. If provided, input at target
            positions (mask=1) is zeroed before feeding the encoder.
            This implements the context-only constraint.

        Returns
        -------
        torch.Tensor
            Per-timestep latent vectors, shape ``(B, T, H)``.
        """
        if context_mask is not None:
            # Zero out target positions (mask=1 means target)
            # Expand mask to (B, T, 1) for broadcasting
            gate = (1.0 - context_mask.unsqueeze(-1))  # 0 at targets, 1 at context
            x = x * gate

        output, _ = self.gru(x)  # (B, T, H)
        return output


class MLPPredictor(nn.Module):
    """2-layer MLP predictor with time-position embedding.

    The predictor receives student latents concatenated with positional
    encoding and produces target latent predictions.

    Parameters
    ----------
    hidden_dim : int
        Latent dimension H.
    max_seq_len : int
        Maximum sequence length T (for positional encoding).
    """

    def __init__(self, hidden_dim: int, max_seq_len: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Learned mask token embedding (Section 5.4)
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # 2-layer MLP: (H + H) -> H -> H
        # Input is: latent (or mask token) + positional encoding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pre-compute positional encodings
        self.register_buffer(
            "pos_encoding",
            sinusoidal_position_encoding(max_seq_len, hidden_dim),
        )

    def forward(
        self,
        student_latents: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target latents.

        Parameters
        ----------
        student_latents : torch.Tensor
            Student encoder outputs, shape ``(B, T, H)``.
        mask : torch.Tensor
            Binary mask, shape ``(B, T)``. 1 = target position.

        Returns
        -------
        torch.Tensor
            Predicted latents, shape ``(B, T, H)``.
            Only target positions contain meaningful predictions.
        """
        B, T, H = student_latents.shape

        # Replace target positions with learned mask token
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        mask_token_expanded = self.mask_token.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        predictor_input = (
            student_latents * (1.0 - mask_expanded) +
            mask_token_expanded * mask_expanded
        )

        # Add positional encoding
        pos = self.pos_encoding[:T].unsqueeze(0).expand(B, -1, -1)  # (B, T, H)
        combined = torch.cat([predictor_input, pos], dim=-1)  # (B, T, 2H)

        predictions = self.mlp(combined)  # (B, T, H)
        return predictions


class ANNModel(nn.Module):
    """Complete ANN JEPA model with encoder, predictor, and EMA teacher.

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
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        max_seq_len: int = 128,
        tau_ema: float = 0.996,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_ema = tau_ema

        # Student encoder
        self.encoder = GRUEncoder(input_dim, hidden_dim)

        # Predictor
        self.predictor = MLPPredictor(hidden_dim, max_seq_len)

        # Teacher encoder (EMA copy, no gradients)
        self.teacher_encoder = GRUEncoder(input_dim, hidden_dim)
        self._copy_encoder_to_teacher()
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

    def _copy_encoder_to_teacher(self) -> None:
        """Copy encoder parameters to teacher (initial sync)."""
        for tp, sp in zip(
            self.teacher_encoder.parameters(),
            self.encoder.parameters(),
        ):
            tp.data.copy_(sp.data)

    @torch.no_grad()
    def update_teacher(self) -> None:
        """Update teacher parameters by EMA (Eq. 6).

        theta_bar <- tau * theta_bar + (1 - tau) * theta
        """
        for tp, sp in zip(
            self.teacher_encoder.parameters(),
            self.encoder.parameters(),
        ):
            tp.data.mul_(self.tau_ema).add_(sp.data, alpha=1.0 - self.tau_ema)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the complete JEPA model.

        Parameters
        ----------
        x : torch.Tensor
            Input time-series, shape ``(B, T, D)``.
        mask : torch.Tensor
            Binary mask, shape ``(B, T)``. 1 = target.

        Returns
        -------
        predictions : torch.Tensor
            Predictor outputs, shape ``(B, T, H)``.
        teacher_targets : torch.Tensor
            Teacher encoder outputs (detached), shape ``(B, T, H)``.
        """
        # Student encoder: sees only context (target positions zeroed)
        student_latents = self.encoder(x, context_mask=mask)

        # Predictor: predicts target latents
        predictions = self.predictor(student_latents, mask)

        # Teacher encoder: processes full window (stop-gradient)
        with torch.no_grad():
            teacher_targets = self.teacher_encoder(x)

        return predictions, teacher_targets

    def get_student_latents(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get student encoder outputs for observability logging.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape ``(B, T, D)``.
        mask : torch.Tensor
            Binary mask, shape ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Student latents, shape ``(B, T, H)``.
        """
        return self.encoder(x, context_mask=mask)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
