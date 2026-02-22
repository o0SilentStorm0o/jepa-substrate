"""Micro-benchmark: Lava process graph latency decomposition.

Measures the three components of Lava SNN inference latency:
  (1) Graph build: Lava process instantiation and wiring
  (2) RunSteps: actual simulation execution
  (3) Monitor read: spike/membrane data retrieval

Also measures the end-to-end latency for the benchmark
(build + run only, no monitor).

Usage:
  python scripts/benchmark_lava.py [--config config/experiment.yaml]
                                   [--n-warmup 5]
                                   [--n-timed 100]
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

from config.loader import load_config
from shared.positional import sinusoidal_position_encoding_numpy

logger = logging.getLogger(__name__)


def benchmark_lava_components(
    weights_path: str,
    cfg,
    n_warmup: int = 5,
    n_timed: int = 100,
) -> dict:
    """Benchmark individual Lava process graph components.

    Parameters
    ----------
    weights_path : str
        Path to exported Lava weights NPZ.
    cfg : Config
        Experiment configuration.
    n_warmup : int
        Warm-up iterations.
    n_timed : int
        Timed iterations.

    Returns
    -------
    dict
        Per-component timing results.
    """
    from snn.lava_export import load_lava_weights
    from snn.lava_inference import _build_lava_graph

    weights = load_lava_weights(weights_path)
    T = cfg.data.T
    H = cfg.model.H

    rng = np.random.default_rng(42)
    x_window = rng.standard_normal((T, cfg.data.D)).astype(np.float32) * 0.1
    mask = np.zeros(T, dtype=np.float32)
    mask[-20:] = 1.0
    mask_signal = (mask.reshape(-1, 1) * cfg.model.c_mask).astype(np.float32)
    pos_encoding = sinusoidal_position_encoding_numpy(T, H)

    build_times = []
    run_times = []
    monitor_times = []
    total_times = []

    for i in range(n_warmup + n_timed):
        # (1) Build
        t_build_start = time.perf_counter()
        graph = _build_lava_graph(
            weights, x_window, mask_signal, pos_encoding, T,
            attach_monitor=True,
        )
        t_build_end = time.perf_counter()

        # (2) Run
        t_run_start = time.perf_counter()
        graph["lif_enc"].run(
            condition=graph["run_cond"],
            run_cfg=graph["run_cfg"],
        )
        t_run_end = time.perf_counter()

        # (3) Monitor read
        t_mon_start = time.perf_counter()
        _ = graph["output_sink"].data.get()
        if graph["monitors"]:
            for mon in graph["monitors"].values():
                _ = mon.get_data()
        t_mon_end = time.perf_counter()

        # Stop
        graph["lif_enc"].stop()

        t_total = t_mon_end - t_build_start

        if i >= n_warmup:
            build_times.append((t_build_end - t_build_start) * 1000.0)
            run_times.append((t_run_end - t_run_start) * 1000.0)
            monitor_times.append((t_mon_end - t_mon_start) * 1000.0)
            total_times.append(t_total * 1000.0)

    return {
        "build": {
            "median_ms": float(np.median(build_times)),
            "mean_ms": float(np.mean(build_times)),
            "std_ms": float(np.std(build_times)),
        },
        "run": {
            "median_ms": float(np.median(run_times)),
            "mean_ms": float(np.mean(run_times)),
            "std_ms": float(np.std(run_times)),
        },
        "monitor": {
            "median_ms": float(np.median(monitor_times)),
            "mean_ms": float(np.mean(monitor_times)),
            "std_ms": float(np.std(monitor_times)),
        },
        "total": {
            "median_ms": float(np.median(total_times)),
            "mean_ms": float(np.mean(total_times)),
            "std_ms": float(np.std(total_times)),
        },
        "n_timed": n_timed,
    }


def benchmark_lava_no_monitor(
    weights_path: str,
    cfg,
    n_warmup: int = 5,
    n_timed: int = 1000,
) -> dict:
    """Benchmark Lava inference without monitors (benchmark mode).

    This measures only build + run, which is the latency reported
    as the SNN component (1) in the time budget.
    """
    from snn.lava_inference import run_snn_latency_benchmark

    rng = np.random.default_rng(42)
    x_window = rng.standard_normal((cfg.data.T, cfg.data.D)).astype(np.float32) * 0.1
    mask = np.zeros(cfg.data.T, dtype=np.float32)
    mask[-20:] = 1.0

    result = run_snn_latency_benchmark(
        weights_path=weights_path,
        x_window=x_window,
        mask=mask,
        T=cfg.data.T,
        H=cfg.model.H,
        c_mask=cfg.model.c_mask,
        n_warmup=n_warmup,
        n_timed=n_timed,
    )

    return {
        "component": "t_inf_SNN_no_monitor",
        "median_ms": result.median_ms,
        "mean_ms": result.mean_ms,
        "std_ms": result.std_ms,
        "n_timed": n_timed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lava latency benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "experiment.yaml"),
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to exported Lava weights NPZ file.",
    )
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-timed", type=int, default=100)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "lava_benchmark.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)

    logger.info("Benchmarking Lava components (with monitor)...")
    component_results = benchmark_lava_components(
        args.weights, cfg, args.n_warmup, args.n_timed,
    )

    logger.info("Benchmarking Lava inference (no monitor)...")
    no_mon_result = benchmark_lava_no_monitor(
        args.weights, cfg, args.n_warmup, min(args.n_timed, 100),
    )

    # Print table
    print("\n" + "=" * 60)
    print("Lava Component Benchmark")
    print("-" * 60)
    for name, data in component_results.items():
        if isinstance(data, dict) and "median_ms" in data:
            print(
                f"  {name:<15} median={data['median_ms']:8.3f}ms  "
                f"mean={data['mean_ms']:8.3f}ms  "
                f"std={data['std_ms']:7.3f}ms"
            )
    print("-" * 60)
    print(
        f"  {'no-monitor':<15} median={no_mon_result['median_ms']:8.3f}ms  "
        f"mean={no_mon_result['mean_ms']:8.3f}ms  "
        f"std={no_mon_result['std_ms']:7.3f}ms"
    )
    print("=" * 60)

    combined = {
        "components": component_results,
        "no_monitor": no_mon_result,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined, indent=2), encoding="utf-8",
    )
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
