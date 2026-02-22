"""Latency measurement utilities.

Both ANN and SNN harnesses use this identical timing wrapper
to ensure the same latency definition (invariant I4).

Latency is decomposed into three components:
  (1) Encoder + predictor forward (primary metric)
  (2) Teacher forward (zero if pre-computed)
  (3) Runtime overhead (data loading, monitor, orchestration)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class LatencyResult:
    """Container for latency measurement results."""

    # Component (1): encoder + predictor forward
    forward_ms: float = 0.0
    # Component (2): teacher forward (0 if pre-computed)
    teacher_ms: float = 0.0
    # Component (3): runtime overhead
    overhead_ms: float = 0.0
    # Total wall-clock
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "forward_ms": self.forward_ms,
            "teacher_ms": self.teacher_ms,
            "overhead_ms": self.overhead_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class BenchmarkResult:
    """Container for latency benchmark results over multiple calls."""

    # Per-call forward latencies (ms)
    forward_latencies_ms: List[float] = field(default_factory=list)
    # Summary statistics
    median_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p5_ms: float = 0.0
    p95_ms: float = 0.0
    iqr_ms: float = 0.0
    n_warmup: int = 0
    n_timed: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "p5_ms": self.p5_ms,
            "p95_ms": self.p95_ms,
            "iqr_ms": self.iqr_ms,
            "n_warmup": self.n_warmup,
            "n_timed": self.n_timed,
        }


def measure_latency(
    forward_fn: Callable[[], None],
    teacher_fn: Optional[Callable[[], None]] = None,
    setup_fn: Optional[Callable[[], None]] = None,
    teardown_fn: Optional[Callable[[], None]] = None,
) -> LatencyResult:
    """Measure wall-clock latency for a single inference call.

    Parameters
    ----------
    forward_fn : Callable
        The encoder + predictor forward pass (component 1).
    teacher_fn : Callable, optional
        Teacher forward pass (component 2). None if pre-computed.
    setup_fn : Callable, optional
        Setup operations (part of component 3).
    teardown_fn : Callable, optional
        Teardown operations (part of component 3).

    Returns
    -------
    LatencyResult
        Decomposed latency measurements.
    """
    result = LatencyResult()
    total_start = time.perf_counter()

    # Component (3) setup
    if setup_fn is not None:
        setup_fn()

    # Component (1) forward
    fwd_start = time.perf_counter()
    forward_fn()
    fwd_end = time.perf_counter()
    result.forward_ms = (fwd_end - fwd_start) * 1000.0

    # Component (2) teacher
    if teacher_fn is not None:
        teacher_start = time.perf_counter()
        teacher_fn()
        teacher_end = time.perf_counter()
        result.teacher_ms = (teacher_end - teacher_start) * 1000.0

    # Component (3) teardown
    if teardown_fn is not None:
        teardown_fn()

    total_end = time.perf_counter()
    result.total_ms = (total_end - total_start) * 1000.0
    result.overhead_ms = result.total_ms - result.forward_ms - result.teacher_ms

    return result


def run_latency_benchmark(
    forward_fn: Callable[[], None],
    n_warmup: int = 5,
    n_timed: int = 1000,
    setup_fn: Optional[Callable[[], None]] = None,
    teardown_fn: Optional[Callable[[], None]] = None,
) -> BenchmarkResult:
    """Run a latency benchmark with warm-up and timed repetitions.

    Follows the no-hidden-compute rule: only ``forward_fn`` is timed.
    Warm-up includes setup so that JIT compilation / process-model
    compilation is completed before timing.

    Parameters
    ----------
    forward_fn : Callable
        The forward pass to benchmark (component 1 only).
    n_warmup : int
        Number of untimed warm-up calls.
    n_timed : int
        Number of timed calls.
    setup_fn : Callable, optional
        Per-call setup (excluded from timing).
    teardown_fn : Callable, optional
        Per-call teardown (excluded from timing).

    Returns
    -------
    BenchmarkResult
        Benchmark statistics.
    """
    # Warm-up phase
    for _ in range(n_warmup):
        if setup_fn is not None:
            setup_fn()
        forward_fn()
        if teardown_fn is not None:
            teardown_fn()

    # Timed phase
    latencies = []
    for _ in range(n_timed):
        if setup_fn is not None:
            setup_fn()

        start = time.perf_counter()
        forward_fn()
        end = time.perf_counter()

        if teardown_fn is not None:
            teardown_fn()

        latencies.append((end - start) * 1000.0)

    arr = np.array(latencies, dtype=np.float64)
    result = BenchmarkResult(
        forward_latencies_ms=latencies,
        median_ms=float(np.median(arr)),
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr, ddof=1)),
        p5_ms=float(np.percentile(arr, 5)),
        p95_ms=float(np.percentile(arr, 95)),
        iqr_ms=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        n_warmup=n_warmup,
        n_timed=n_timed,
    )
    return result
