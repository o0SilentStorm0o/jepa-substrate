"""ANN training loop for LavaJEPA.

Implements the training protocol from Section 5.6:
  - Cosine schedule with linear warm-up (5% of N_train)
  - AdamW optimizer (lr_max, betas=(0.9, 0.999), weight_decay=1e-4)
  - EMA teacher update after each step (tau=0.996, fixed)
  - Validation every 500 steps
  - Checkpointing at best validation loss and final step

All training uses the shared WindowDataset and loss function.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ann.model import ANNModel
from shared.data import WindowDataset
from shared.gates import FailureGateChecker
from shared.loss import jepa_time_loss

logger = logging.getLogger(__name__)


def cosine_lr_schedule(
    step: int,
    total_steps: int,
    lr_max: float,
    warmup_steps: int,
) -> float:
    """Cosine learning rate schedule with linear warm-up.

    Parameters
    ----------
    step : int
        Current training step (0-indexed).
    total_steps : int
        Total number of training steps.
    lr_max : float
        Peak learning rate.
    warmup_steps : int
        Number of linear warm-up steps.

    Returns
    -------
    float
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_ann(
    model: ANNModel,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    n_train_steps: int = 5000,
    lr_max: float = 1e-3,
    warmup_fraction: float = 0.05,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    val_every: int = 500,
    output_dir: Path | str = "outputs/ann",
    device: str = "cpu",
    seed: int = 1,
    gate_checker: Optional[FailureGateChecker] = None,
) -> Dict[str, Any]:
    """Train the ANN model.

    Parameters
    ----------
    model : ANNModel
        The ANN JEPA model.
    train_dataset : WindowDataset
        Training dataset.
    val_dataset : WindowDataset
        Validation dataset.
    n_train_steps : int
        Total number of training steps.
    lr_max : float
        Peak learning rate.
    warmup_fraction : float
        Fraction of steps for linear warm-up.
    weight_decay : float
        AdamW weight decay.
    betas : tuple
        AdamW beta parameters.
    val_every : int
        Validate every N steps.
    output_dir : Path or str
        Output directory for checkpoints and logs.
    device : str
        Device to train on.
    seed : int
        Random seed for training order.
    gate_checker : FailureGateChecker, optional
        Failure-mode gate checker.

    Returns
    -------
    dict
        Training history and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for training order
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    model.train()

    # Optimizer: AdamW
    # Only optimize encoder and predictor (teacher is EMA)
    trainable_params = [
        p for p in model.encoder.parameters() if p.requires_grad
    ] + [
        p for p in model.predictor.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr_max,
        betas=betas,
        weight_decay=weight_decay,
    )

    warmup_steps = int(warmup_fraction * n_train_steps)

    # DataLoader with shuffling for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    train_iter = iter(train_loader)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "step": [],
        "embedding_norm_student": [],
        "embedding_norm_teacher": [],
    }

    best_val_loss = float("inf")
    gate_failed = False

    t_start = time.time()

    for step in range(n_train_steps):
        # Get next batch (cycle through dataset)
        try:
            x, mask, seed_w = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, mask, seed_w = next(train_iter)

        x = x.to(device)        # (1, T, D)
        mask = mask.to(device)   # (1, T)

        # Update learning rate
        lr = cosine_lr_schedule(step, n_train_steps, lr_max, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass
        predictions, teacher_targets = model(x, mask)

        # Loss (stop-gradient on teacher targets)
        loss = jepa_time_loss(predictions, teacher_targets.detach(), mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA teacher update (Eq. 6)
        model.update_teacher()

        train_loss = loss.item()
        history["train_loss"].append(train_loss)
        history["lr"].append(lr)
        history["step"].append(step)

        # Collapse gate check (F1)
        if gate_checker is not None and step > 0 and step % val_every == 0:
            gate_result = gate_checker.check_collapse(predictions.detach())
            if not gate_result.passed:
                logger.error(
                    "GATE FAILED at step %d: %s", step, gate_result.message
                )
                gate_failed = True
                break

        # Log embedding norms for collapse detection
        with torch.no_grad():
            student_norm = predictions.norm(dim=-1).mean().item()
            teacher_norm = teacher_targets.norm(dim=-1).mean().item()
            history["embedding_norm_student"].append(student_norm)
            history["embedding_norm_teacher"].append(teacher_norm)

        # Validation
        if step > 0 and step % val_every == 0:
            val_loss = evaluate_ann_val(model, val_dataset, device)
            history["val_loss"].append({"step": step, "loss": val_loss})

            logger.info(
                "Step %d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.6f | "
                "s_norm=%.3f | t_norm=%.3f",
                step, n_train_steps, train_loss, val_loss, lr,
                student_norm, teacher_norm,
            )

            # Checkpoint at best validation loss (R3)
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

    # Save final checkpoint (R3)
    torch.save(
        {
            "step": n_train_steps - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
        },
        output_dir / "checkpoint_final.pt",
    )

    # Save training history
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    result = {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "n_steps": len(history["train_loss"]),
        "elapsed_seconds": elapsed,
        "gate_failed": gate_failed,
        "n_params": model.count_parameters(),
    }

    logger.info(
        "Training complete: %d steps in %.1fs | best_val=%.6f | params=%d",
        result["n_steps"], elapsed, best_val_loss, result["n_params"],
    )

    return result


def evaluate_ann_val(
    model: ANNModel,
    val_dataset: WindowDataset,
    device: str = "cpu",
) -> float:
    """Evaluate on validation set. Returns mean loss.

    Parameters
    ----------
    model : ANNModel
        The ANN model (set to eval mode internally).
    val_dataset : WindowDataset
        Validation dataset.
    device : str
        Device.

    Returns
    -------
    float
        Mean validation loss over all windows.
    """
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    total_loss = 0.0
    n_windows = 0

    with torch.no_grad():
        for x, mask, _ in val_loader:
            x = x.to(device)
            mask = mask.to(device)
            predictions, teacher_targets = model(x, mask)
            loss = jepa_time_loss(predictions, teacher_targets, mask)
            total_loss += loss.item()
            n_windows += 1

    model.train()
    return total_loss / max(n_windows, 1)
