"""Optimized runner: all 60 primary experiments (30 ANN + 30 SNN).

Prepares data once, then iterates over seeds/policies/substrates.
SNN runs use PyTorch surrogate-gradient training + measurement
(Lava inference is skipped on Apple Silicon -- no Loihi HW).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

from config.loader import load_config, Config
from shared.data import prepare_dataset, WindowDataset
from shared.masking import generate_mask
from shared.env_log import save_environment_info
from shared.timing import measure_latency
from shared.loss import jepa_time_loss
from shared.harness import WindowResult, save_results_csv

from ann.model import ANNModel
from ann.train import train_ann
from ann.evaluate import precompute_teacher_targets, run_ann_measurement

from snn.model import SNNModel
from snn.train import train_snn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_snn_pytorch_measurement(
    model: SNNModel,
    test_dataset: WindowDataset,
    targets_dir: Path,
    output_dir: Path,
) -> List[WindowResult]:
    """Measure SNN using PyTorch (CPU) instead of Lava.

    Same protocol as ANN: batch_size=1, sequential, pre-computed targets.
    """
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.set_num_threads(1)

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    results: List[WindowResult] = []

    with torch.no_grad():
        for idx, (x, mask, _) in enumerate(loader):
            # Load teacher target
            target_path = targets_dir / f"target_{idx:05d}.npz"
            teacher_np = np.load(str(target_path))["targets"]
            teacher_target = torch.from_numpy(teacher_np).unsqueeze(0)

            # Timed forward
            def forward_fn():
                return model(x, mask)

            latency = measure_latency(forward_fn)

            # Get predictions
            predictions, _, _, _, _ = model(x, mask)

            # Loss
            loss = jepa_time_loss(predictions, teacher_target.detach(), mask)

            wr = WindowResult(
                window_index=idx,
                loss=float(loss.item()),
                forward_ms=latency.forward_ms,
                energy_proxy=0.0,  # No energy proxy without Lava/Loihi
            )
            results.append(wr)

            if idx % 200 == 0:
                logger.info(
                    "SNN measurement: window %d/%d | loss=%.6f | fwd=%.2fms",
                    idx, len(test_dataset), wr.loss, wr.forward_ms,
                )

    save_results_csv(results, output_dir / "results.csv")
    return results


def precompute_snn_teacher_targets(
    model: SNNModel,
    test_windows: np.ndarray,
    output_dir: Path,
) -> Path:
    """Pre-compute SNN teacher targets."""
    targets_dir = output_dir / "teacher_targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    sha = hashlib.sha256()

    with torch.no_grad():
        for idx in range(len(test_windows)):
            x = torch.from_numpy(test_windows[idx]).float().unsqueeze(0)
            latents, _, _, _ = model.teacher_encoder(x)
            targets_np = latents.squeeze(0).numpy().astype(np.float32)

            np.savez(
                str(targets_dir / f"target_{idx:05d}.npz"),
                targets=targets_np,
            )
            sha.update(targets_np.tobytes())

    hash_file = targets_dir / "targets_sha256.txt"
    hash_file.write_text(sha.hexdigest() + "\n")
    logger.info("Pre-computed %d SNN teacher targets", len(test_windows))
    return targets_dir


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "config" / "experiment.yaml"))
    data_dir = PROJECT_ROOT / "data" / "uci_har"
    output_root = PROJECT_ROOT / "results"
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- Prepare data ONCE ----
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(data_dir=data_dir, split_seed=cfg.seeds.data_split)
    train_np = dataset["train"]
    val_np = dataset["val"]
    test_np = dataset["test"]
    logger.info(
        "Data ready: train=%d, val=%d, test=%d",
        len(train_np), len(val_np), len(test_np),
    )

    seeds = cfg.seeds.init_seeds
    policies = cfg.experiment.policies
    substrates = ["ann", "snn"]

    total_runs = len(seeds) * len(policies) * len(substrates)
    logger.info("Total primary runs: %d", total_runs)

    summaries = []
    run_idx = 0

    for seed in seeds:
        for policy in policies:
            for substrate in substrates:
                run_idx += 1
                run_id = f"s{seed}_{policy}_{substrate}_primary"
                run_dir = output_root / run_id

                # Skip if already completed
                if (run_dir / "results.csv").exists():
                    logger.info(
                        "[%d/%d] SKIP (already done): %s",
                        run_idx, total_runs, run_id,
                    )
                    summaries.append({"run_id": run_id, "status": "skipped"})
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "[%d/%d] START: %s",
                    run_idx, total_runs, run_id,
                )
                t0 = time.time()

                save_environment_info(run_dir / "environment.json")
                (run_dir / "run_spec.json").write_text(json.dumps({
                    "seed": seed,
                    "masking_policy": policy,
                    "substrate": substrate,
                    "condition": "primary",
                }, indent=2))

                torch.manual_seed(seed)
                np.random.seed(seed)

                # Build datasets
                ds_kwargs = dict(
                    masking_policy=policy,
                    split_seed=seed,
                    context_fraction=cfg.masking.future_block.context_fraction,
                    target_probability=cfg.masking.random_drop.target_probability,
                    n_blocks=cfg.masking.multi_target.n_blocks,
                    block_length_fraction=cfg.masking.multi_target.block_length_fraction,
                    min_gap_fraction=cfg.masking.multi_target.min_gap_fraction,
                )
                train_dataset = WindowDataset(windows=train_np, **ds_kwargs)
                val_dataset = WindowDataset(windows=val_np, **ds_kwargs)
                test_dataset = WindowDataset(windows=test_np, **ds_kwargs)

                try:
                    if substrate == "ann":
                        model = ANNModel(
                            input_dim=cfg.data.D,
                            hidden_dim=cfg.model.H,
                            tau_ema=cfg.model.tau_ema,
                        )
                        train_ann(
                            model=model,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            n_train_steps=cfg.training.N_train,
                            lr_max=cfg.training.lr_max,
                            warmup_fraction=cfg.training.warmup_fraction,
                            weight_decay=cfg.training.weight_decay,
                            betas=tuple(cfg.training.betas),
                            val_every=cfg.training.val_every,
                            output_dir=run_dir,
                            seed=seed,
                        )
                        targets_dir = precompute_teacher_targets(
                            model=model,
                            dataset=test_dataset,
                            output_dir=run_dir,
                        )
                        results = run_ann_measurement(
                            model=model,
                            test_dataset=test_dataset,
                            targets_dir=targets_dir,
                            output_dir=run_dir,
                        )

                    elif substrate == "snn":
                        model = SNNModel(
                            input_dim=cfg.data.D,
                            hidden_dim=cfg.model.H,
                            beta=cfg.model.beta,
                            v_th=cfg.model.v_th,
                            surrogate_k=cfg.model.surrogate_k,
                            trace_alpha=cfg.model.trace_alpha,
                            tau_ema=cfg.model.tau_ema,
                        )
                        train_snn(
                            model=model,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            n_train_steps=cfg.training.N_train,
                            lr_max=cfg.training.lr_max,
                            warmup_fraction=cfg.training.warmup_fraction,
                            weight_decay=cfg.training.weight_decay,
                            betas=tuple(cfg.training.betas),
                            val_every=cfg.training.val_every,
                            output_dir=run_dir,
                            seed=seed,
                        )
                        targets_dir = precompute_snn_teacher_targets(
                            model, test_np, run_dir,
                        )
                        results = run_snn_pytorch_measurement(
                            model, test_dataset, targets_dir, run_dir,
                        )

                    elapsed = time.time() - t0
                    mean_loss = float(np.mean([r.loss for r in results]))
                    summaries.append({
                        "run_id": run_id,
                        "seed": seed,
                        "policy": policy,
                        "substrate": substrate,
                        "mean_loss": mean_loss,
                        "elapsed_s": elapsed,
                        "n_windows": len(results),
                    })
                    logger.info(
                        "[%d/%d] DONE: %s | loss=%.4f | %.1fs",
                        run_idx, total_runs, run_id, mean_loss, elapsed,
                    )

                except Exception:
                    logger.exception("[%d/%d] FAILED: %s", run_idx, total_runs, run_id)
                    summaries.append({"run_id": run_id, "status": "FAILED"})

    # Save summary
    summary_path = output_root / "primary_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str))
    n_ok = sum(1 for s in summaries if "mean_loss" in s)
    logger.info("Primary experiments complete: %d/%d successful", n_ok, total_runs)


if __name__ == "__main__":
    main()
