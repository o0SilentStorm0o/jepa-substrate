"""SNN training loop for LavaJEPA.

Implements SNN training via surrogate-gradient BPTT in PyTorch.
After training, weights are exported to Lava for inference/measurement.

Training protocol mirrors the ANN training:
  - Cosine schedule with linear warm-up
  - AdamW optimizer
  - EMA teacher update
  - Same validation procedure
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.data import WindowDataset
from shared.gates import FailureGateChecker
from shared.loss import jepa_time_loss
from snn.model import SNNModel

logger = logging.getLogger(__name__)


def cosine_lr_schedule(
    step: int,
    total_steps: int,
    lr_max: float,
    warmup_steps: int,
) -> float:
    """Cosine learning rate schedule with linear warm-up."""
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_snn(
    model: SNNModel,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    n_train_steps: int = 5000,
    lr_max: float = 1e-3,
    warmup_fraction: float = 0.05,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    val_every: int = 500,
    output_dir: Path | str = "outputs/snn",
    device: str = "cpu",
    seed: int = 1,
    gate_checker: Optional[FailureGateChecker] = None,
) -> Dict[str, Any]:
    """Train the SNN model with surrogate-gradient BPTT.

    Parameters
    ----------
    model : SNNModel
        The SNN JEPA model.
    train_dataset : WindowDataset
        Training dataset.
    val_dataset : WindowDataset
        Validation dataset.
    n_train_steps : int
        Total training steps.
    lr_max : float
        Peak learning rate.
    warmup_fraction : float
        Warmup fraction.
    weight_decay : float
        Weight decay.
    betas : tuple
        AdamW betas.
    val_every : int
        Validation frequency.
    output_dir : Path or str
        Output directory.
    device : str
        Device.
    seed : int
        Random seed.
    gate_checker : FailureGateChecker, optional
        Failure-mode gate checker.

    Returns
    -------
    dict
        Training history and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    model.train()

    # Optimizer: only encoder and predictor
    trainable_params = list(
        p for p in model.encoder.parameters() if p.requires_grad
    ) + list(
        p for p in model.predictor.parameters() if p.requires_grad
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr_max,
        betas=betas,
        weight_decay=weight_decay,
    )

    warmup_steps = int(warmup_fraction * n_train_steps)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True,
    )
    train_iter = iter(train_loader)

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "step": [],
        "embedding_norm_student": [],
        "embedding_norm_teacher": [],
        "spike_rate": [],
        "total_spikes": [],
    }

    best_val_loss = float("inf")
    gate_failed = False

    t_start = time.time()

    for step in range(n_train_steps):
        try:
            x, mask, seed_w = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, mask, seed_w = next(train_iter)

        x = x.to(device)
        mask = mask.to(device)

        lr = cosine_lr_schedule(step, n_train_steps, lr_max, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass
        predictions, teacher_targets, enc_spikes, pred_spikes, enc_traces = model(x, mask)

        # Loss
        loss = jepa_time_loss(predictions, teacher_targets.detach(), mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA teacher update
        model.update_teacher()

        train_loss = loss.item()
        history["train_loss"].append(train_loss)
        history["lr"].append(lr)
        history["step"].append(step)

        # Spike statistics
        spike_stats = model.get_spike_stats(enc_spikes.detach(), pred_spikes.detach())
        history["spike_rate"].append(spike_stats["spike_rate"])
        history["total_spikes"].append(spike_stats["total_spikes"])

        # Failure gate checks
        if gate_checker is not None and step > 0 and step % val_every == 0:
            # F1: Collapse
            collapse_result = gate_checker.check_collapse(predictions.detach())
            if not collapse_result.passed:
                logger.error("GATE FAILED at step %d: %s", step, collapse_result.message)
                gate_failed = True
                break

            # F2: Saturation
            enc_spike_counts = enc_spikes.detach().sum(dim=1).cpu().numpy()[0]
            sat_result = gate_checker.check_saturation(
                enc_spike_counts,
                spike_stats["n_neurons"],
                spike_stats["n_steps"],
            )
            if not sat_result.passed:
                logger.error("GATE FAILED at step %d: %s", step, sat_result.message)
                gate_failed = True
                break

        # Embedding norms
        with torch.no_grad():
            student_norm = predictions.norm(dim=-1).mean().item()
            teacher_norm = teacher_targets.norm(dim=-1).mean().item()
            history["embedding_norm_student"].append(student_norm)
            history["embedding_norm_teacher"].append(teacher_norm)

        # Validation
        if step > 0 and step % val_every == 0:
            val_loss = evaluate_snn_val(model, val_dataset, device)
            history["val_loss"].append({"step": step, "loss": val_loss})

            logger.info(
                "Step %d/%d | loss=%.6f | val=%.6f | lr=%.6f | "
                "spikes=%d | rate=%.4f | s_norm=%.3f | t_norm=%.3f",
                step, n_train_steps, train_loss, val_loss, lr,
                spike_stats["total_spikes"], spike_stats["spike_rate"],
                student_norm, teacher_norm,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    output_dir / "checkpoint_best.pt",
                )

    elapsed = time.time() - t_start

    torch.save(
        {
            "step": n_train_steps - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
        },
        output_dir / "checkpoint_final.pt",
    )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    result = {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "n_steps": len(history["train_loss"]),
        "elapsed_seconds": elapsed,
        "gate_failed": gate_failed,
        "n_params": model.count_parameters(),
        "final_spike_rate": history["spike_rate"][-1] if history["spike_rate"] else 0.0,
    }

    logger.info(
        "SNN Training complete: %d steps in %.1fs | best_val=%.6f | params=%d | spike_rate=%.4f",
        result["n_steps"], elapsed, best_val_loss, result["n_params"],
        result["final_spike_rate"],
    )

    return result


def evaluate_snn_val(
    model: SNNModel,
    val_dataset: WindowDataset,
    device: str = "cpu",
) -> float:
    """Evaluate SNN on validation set.

    Parameters
    ----------
    model : SNNModel
        The SNN model.
    val_dataset : WindowDataset
        Validation dataset.
    device : str
        Device.

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    total_loss = 0.0
    n_windows = 0

    with torch.no_grad():
        for x, mask, _ in val_loader:
            x = x.to(device)
            mask = mask.to(device)
            predictions, teacher_targets, _, _, _ = model(x, mask)
            loss = jepa_time_loss(predictions, teacher_targets, mask)
            total_loss += loss.item()
            n_windows += 1

    model.train()
    return total_loss / max(n_windows, 1)
