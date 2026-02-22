"""Micro-benchmark: time budget estimation.

Measures the wall-clock cost of each atomic operation to validate
the time budget model from Section 5.8 of the specification:

  - t_step_ANN: single ANN training step
  - t_step_SNN: single SNN training step
  - t_inf_ANN: single ANN inference (B=1)
  - t_build_Lava: Lava graph build time
  - t_run_Lava: Lava RunSteps execution time
  - t_mon_Lava: Lava Monitor read time

Results are printed as a table and saved to JSON.

Usage:
  python scripts/benchmark_time_budget.py [--config config/experiment.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.loader import load_config
from shared.timing import measure_latency, run_latency_benchmark

logger = logging.getLogger(__name__)


def benchmark_ann_step(cfg, n_warmup: int = 5, n_timed: int = 100) -> dict:
    """Benchmark a single ANN training step."""
    from ann.model import ANNModel

    model = ANNModel(
        input_dim=cfg.data.D,
        hidden_dim=cfg.model.H,
        tau_ema=cfg.model.tau_ema,
    )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_max,
        weight_decay=cfg.training.weight_decay,
    )

    # Dummy data
    x = torch.randn(1, cfg.data.T, cfg.data.D)
    mask = torch.zeros(1, cfg.data.T)
    mask[:, -20:] = 1.0

    times = []

    for i in range(n_warmup + n_timed):
        t0 = time.perf_counter()

        optimizer.zero_grad()
        z_hat, z_teacher = model(x, mask)
        from shared.loss import jepa_time_loss
        loss = jepa_time_loss(z_hat, z_teacher, mask)
        loss.backward()
        optimizer.step()
        model.update_teacher()

        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append((t1 - t0) * 1000.0)

    return {
        "component": "t_step_ANN",
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "n_timed": n_timed,
    }


def benchmark_snn_step(cfg, n_warmup: int = 5, n_timed: int = 100) -> dict:
    """Benchmark a single SNN training step."""
    from snn.model import SNNModel

    model = SNNModel(
        input_dim=cfg.data.D,
        hidden_dim=cfg.model.H,
        beta=cfg.model.beta,
        v_th=cfg.model.v_th,
        surrogate_k=cfg.model.surrogate_k,
        alpha_trace=cfg.model.alpha_trace,
        tau_ema=cfg.model.tau_ema,
    )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_max,
        weight_decay=cfg.training.weight_decay,
    )

    x = torch.randn(1, cfg.data.T, cfg.data.D)
    mask = torch.zeros(1, cfg.data.T)
    mask[:, -20:] = 1.0

    times = []

    for i in range(n_warmup + n_timed):
        t0 = time.perf_counter()

        optimizer.zero_grad()
        z_hat, z_teacher = model(x, mask)
        from shared.loss import jepa_time_loss
        loss = jepa_time_loss(z_hat, z_teacher, mask)
        loss.backward()
        optimizer.step()
        model.update_teacher()

        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append((t1 - t0) * 1000.0)

    return {
        "component": "t_step_SNN",
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "n_timed": n_timed,
    }


def benchmark_ann_inference(cfg, n_warmup: int = 5, n_timed: int = 1000) -> dict:
    """Benchmark ANN single-window inference (B=1)."""
    from ann.model import ANNModel

    model = ANNModel(
        input_dim=cfg.data.D,
        hidden_dim=cfg.model.H,
        tau_ema=cfg.model.tau_ema,
    )
    model.eval()

    x = torch.randn(1, cfg.data.T, cfg.data.D)

    def forward():
        with torch.no_grad():
            model.teacher_encoder(x)

    result = run_latency_benchmark(forward, n_warmup=n_warmup, n_timed=n_timed)

    return {
        "component": "t_inf_ANN",
        "median_ms": result.median_ms,
        "mean_ms": result.mean_ms,
        "std_ms": result.std_ms,
        "n_timed": n_timed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Time budget micro-benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "experiment.yaml"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "time_budget.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)

    results = []

    logger.info("Benchmarking ANN training step...")
    results.append(benchmark_ann_step(cfg))

    logger.info("Benchmarking SNN training step...")
    results.append(benchmark_snn_step(cfg))

    logger.info("Benchmarking ANN inference...")
    results.append(benchmark_ann_inference(cfg))

    # Print table
    print("\n" + "=" * 60)
    print(f"{'Component':<20} {'Median (ms)':>12} {'Mean (ms)':>12} {'Std (ms)':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['component']:<20} {r['median_ms']:>12.3f} "
            f"{r['mean_ms']:>12.3f} {r['std_ms']:>10.3f}"
        )
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
