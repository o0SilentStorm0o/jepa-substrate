"""Lava-based SNN inference for LavaJEPA measurement.

Implements the Lava process graph from Section 5.5 (Figure 1):
  InputSource -> DenseIn -> LIF_enc -> DensePred -> LIF_pred -> DenseReadout -> OutputSink
  MaskSource -> DenseMask -----> (+) LIF_pred
  PosSource  -> DensePos  -----> (+) LIF_pred

Execution proceeds as a single RunSteps(num_steps=T) call using
Loihi2SimCfg(select_tag="floating_pt").

After the run, data is retrieved from sinks and monitors, then stop() is called.

Window inference is sequential; Lava's internal multiprocessing
(SharedMemoryManager) is incompatible with multiprocessing.Pool.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.harness import (
    WindowResult,
    compute_embedding_stats,
    save_observables_npz,
    save_results_csv,
)
from shared.loss import jepa_time_loss_numpy
from shared.positional import sinusoidal_position_encoding_numpy
from shared.timing import measure_latency, run_latency_benchmark, BenchmarkResult
from snn.lava_export import LavaWeights, load_lava_weights

logger = logging.getLogger(__name__)


def _build_lava_graph(
    weights: LavaWeights,
    x_window: np.ndarray,
    mask_signal: np.ndarray,
    pos_encoding: np.ndarray,
    T: int,
    attach_monitor: bool = True,
) -> dict:
    """Build the Lava process graph for a single window.

    Parameters
    ----------
    weights : LavaWeights
        Exported model weights.
    x_window : np.ndarray
        Input window, shape ``(T, D)``.
    mask_signal : np.ndarray
        Mask signal, shape ``(T, 1)``. c_mask for target positions, 0 otherwise.
    pos_encoding : np.ndarray
        Positional encoding, shape ``(T, H)``.
    T : int
        Number of timesteps.
    attach_monitor : bool
        Whether to attach spike monitors (for observability, not for benchmark).

    Returns
    -------
    dict
        Dictionary containing all Lava processes and run config.
    """
    from lava.proc.dense.process import Dense
    from lava.proc.io.source import RingBuffer as SendProcess
    from lava.proc.io.sink import RingBuffer as RecvProcess
    from lava.proc.lif.process import LIF
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg

    H = weights.hidden_dim
    D = weights.input_dim

    # --- Input sources (RingBuffer) ---
    # InputSource: shape (D,), sends one timestep per step, cycling through T
    input_data = x_window.T.astype(np.float32)  # (D, T) for RingBuffer
    input_source = SendProcess(data=input_data)

    # MaskSource: shape (1,), mask signal
    mask_data = mask_signal.T.astype(np.float32)  # (1, T)
    mask_source = SendProcess(data=mask_data)

    # PosSource: shape (H,), positional encoding
    pos_data = pos_encoding.T.astype(np.float32)  # (H, T)
    pos_source = SendProcess(data=pos_data)

    # --- Dense layers ---
    # DenseIn: input projection (H x D)
    # Receives graded float signals from RingBuffer -> num_message_bits=24
    dense_in = Dense(
        weights=weights.dense_in_weights.T.astype(np.float32),
        num_message_bits=24,
    )

    # DenseMask: mask channel weight (H x 1)
    # Receives graded float from RingBuffer -> num_message_bits=24
    dense_mask = Dense(
        weights=weights.mask_weights.T.astype(np.float32),
        num_message_bits=24,
    )

    # DensePos: identity pass-through (H x H)
    # Receives graded float from RingBuffer -> num_message_bits=24
    dense_pos = Dense(
        weights=np.eye(H, dtype=np.float32),
        num_message_bits=24,
    )

    # --- Encoder LIF ---
    lif_enc = LIF(
        shape=(H,),
        du=weights.encoder_du,
        dv=weights.encoder_dv,
        vth=weights.encoder_vth,
        bias_mant=0,
    )

    # --- Predictor Dense ---
    # The predictor expects concatenated input (2H) from encoder spikes + pos
    # But in Lava, we need to handle this differently since ports are additive
    # We split: DensePred receives encoder spikes, DensePos receives pos signal
    # Both connect to LIF_pred.a_in (additive)
    # For the concatenation approach in the spec, we use a single Dense(H, H)
    # that receives encoder spikes, and add pos and mask as separate currents
    dense_pred = Dense(weights=weights.pred_proj_weights[:H, :H].T.astype(np.float32))

    # Additional dense for positional part of predictor projection
    # Receives graded float from DensePos.a_out -> num_message_bits=24
    dense_pos_pred = Dense(
        weights=weights.pred_proj_weights[H:, :H].T.astype(np.float32),
        num_message_bits=24,
    )

    # --- Predictor LIF ---
    lif_pred = LIF(
        shape=(H,),
        du=weights.predictor_du,
        dv=weights.predictor_dv,
        vth=weights.predictor_vth,
        bias_mant=0,
    )

    # --- Readout Dense ---
    dense_readout = Dense(
        weights=weights.pred_readout_weights.T.astype(np.float32),
        bias=weights.pred_readout_bias.astype(np.float32),
    )

    # --- Output Sink ---
    output_sink = RecvProcess(shape=(H,), buffer=T)

    # --- Wiring ---
    # InputSource -> DenseIn -> LIF_enc
    input_source.s_out.connect(dense_in.s_in)
    dense_in.a_out.connect(lif_enc.a_in)

    # LIF_enc -> DensePred -> LIF_pred (encoder spikes path)
    lif_enc.s_out.connect(dense_pred.s_in)
    dense_pred.a_out.connect(lif_pred.a_in)

    # PosSource -> DensePos -> DensePosPred -> LIF_pred (positional path)
    pos_source.s_out.connect(dense_pos.s_in)
    dense_pos.a_out.connect(dense_pos_pred.s_in)
    dense_pos_pred.a_out.connect(lif_pred.a_in)

    # MaskSource -> DenseMask -> LIF_pred (mask path)
    mask_source.s_out.connect(dense_mask.s_in)
    dense_mask.a_out.connect(lif_pred.a_in)

    # LIF_pred -> DenseReadout -> OutputSink
    lif_pred.s_out.connect(dense_readout.s_in)
    dense_readout.a_out.connect(output_sink.a_in)

    # --- Monitors (for observability) ---
    # Only spike monitors; membrane monitor omitted to reduce overhead
    monitors = {}
    if attach_monitor:
        mon_enc = Monitor()
        mon_enc.probe(lif_enc.s_out, T)
        monitors["enc_spikes"] = mon_enc

        mon_pred = Monitor()
        mon_pred.probe(lif_pred.s_out, T)
        monitors["pred_spikes"] = mon_pred

    # Run config
    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    run_cond = RunSteps(num_steps=T)

    return {
        "input_source": input_source,
        "mask_source": mask_source,
        "pos_source": pos_source,
        "dense_in": dense_in,
        "dense_mask": dense_mask,
        "dense_pos": dense_pos,
        "dense_pred": dense_pred,
        "dense_pos_pred": dense_pos_pred,
        "lif_enc": lif_enc,
        "lif_pred": lif_pred,
        "dense_readout": dense_readout,
        "output_sink": output_sink,
        "monitors": monitors,
        "run_cfg": run_cfg,
        "run_cond": run_cond,
    }


def run_lava_inference_window(
    weights: LavaWeights,
    x_window: np.ndarray,
    mask: np.ndarray,
    pos_encoding: np.ndarray,
    T: int,
    c_mask: float = 1.0,
    attach_monitor: bool = True,
) -> Dict[str, Any]:
    """Run Lava inference for a single window.

    Parameters
    ----------
    weights : LavaWeights
        Exported model weights.
    x_window : np.ndarray
        Input window, shape ``(T, D)``.
    mask : np.ndarray
        Binary mask, shape ``(T,)``. 1 = target.
    pos_encoding : np.ndarray
        Positional encoding, shape ``(T, H)``.
    T : int
        Number of timesteps.
    c_mask : float
        Mask bias current magnitude.
    attach_monitor : bool
        Whether to attach monitors.

    Returns
    -------
    dict
        Inference results including predictions, spike data, and latency.
    """
    # Prepare mask signal: c_mask at target positions, 0 elsewhere
    mask_signal = (mask.reshape(-1, 1) * c_mask).astype(np.float32)  # (T, 1)

    # Build process graph
    graph = _build_lava_graph(
        weights, x_window, mask_signal, pos_encoding, T, attach_monitor,
    )

    # Timed execution: RunSteps(T)
    t_build = time.perf_counter()
    # Graph is already built above

    t_run_start = time.perf_counter()
    graph["lif_enc"].run(
        condition=graph["run_cond"],
        run_cfg=graph["run_cfg"],
    )
    t_run_end = time.perf_counter()

    # Retrieve outputs
    t_read_start = time.perf_counter()
    predictions = graph["output_sink"].data.get().T  # (T, H)

    # Retrieve monitor data
    enc_spikes = None
    pred_spikes = None
    if attach_monitor and graph["monitors"]:
        enc_spikes = graph["monitors"]["enc_spikes"].get_data()[
            graph["lif_enc"].name
        ]["s_out"]  # (T, H) — Monitor returns (T, H) natively
        pred_spikes = graph["monitors"]["pred_spikes"].get_data()[
            graph["lif_pred"].name
        ]["s_out"]  # (T, H)

    t_read_end = time.perf_counter()

    # Stop the runtime
    graph["lif_enc"].stop()

    run_ms = (t_run_end - t_run_start) * 1000.0
    build_ms = (t_run_start - t_build) * 1000.0
    monitor_ms = (t_read_end - t_read_start) * 1000.0

    # Spike statistics
    total_spikes = 0
    spike_rate = 0.0
    synaptic_events = 0
    if enc_spikes is not None and pred_spikes is not None:
        total_enc = int(np.sum(enc_spikes))
        total_pred = int(np.sum(pred_spikes))
        total_spikes = total_enc + total_pred
        n_neurons = weights.hidden_dim * 2
        spike_rate = total_spikes / (n_neurons * T)

        # Synaptic events: sum_t ||s_t||_0 * N_fan_out for each Dense layer
        for t_idx in range(T):
            enc_active = int(np.sum(enc_spikes[t_idx] > 0))
            pred_active = int(np.sum(pred_spikes[t_idx] > 0))
            synaptic_events += enc_active * weights.hidden_dim  # DensePred fan-out
            synaptic_events += pred_active * weights.hidden_dim  # DenseReadout fan-out

    return {
        "predictions": predictions,
        "enc_spikes": enc_spikes,
        "pred_spikes": pred_spikes,
        "total_spikes": total_spikes,
        "spike_rate": spike_rate,
        "synaptic_events": synaptic_events,
        "run_ms": run_ms,
        "build_ms": build_ms,
        "monitor_ms": monitor_ms,
    }


def _process_single_window(
    weights_path_str: str,
    x_window: np.ndarray,
    mask: np.ndarray,
    teacher_targets_dir_str: str,
    output_dir_str: str,
    idx: int,
    T: int,
    H: int,
    c_mask: float,
    energy_alpha: float,
    energy_beta: float,
    energy_gamma: float,
    save_observables: bool,
) -> WindowResult:
    """Process a single window in an isolated process (for multiprocessing).

    This is a module-level function so it can be pickled by
    ``multiprocessing.Pool.starmap``.  Each worker loads its own copy
    of the weights and builds a fresh Lava graph, avoiding shared-memory
    leaks across windows.
    """
    weights = load_lava_weights(weights_path_str)
    pos_encoding = sinusoidal_position_encoding_numpy(T, H)
    teacher_targets_dir = Path(teacher_targets_dir_str)
    output_dir = Path(output_dir_str)

    # Load pre-computed teacher target
    target_path = teacher_targets_dir / f"target_{idx:05d}.npz"
    teacher_target = np.load(str(target_path))["targets"]  # (T, H)

    # Run Lava inference
    t_total_start = time.perf_counter()
    lava_result = run_lava_inference_window(
        weights, x_window, mask, pos_encoding, T, c_mask,
        attach_monitor=save_observables,
    )
    t_total_end = time.perf_counter()

    # Compute loss
    loss = jepa_time_loss_numpy(
        lava_result["predictions"], teacher_target, mask,
    )

    # Energy proxy (Eq. 8)
    energy = (
        energy_alpha * lava_result["synaptic_events"]
        + energy_beta * lava_result["total_spikes"]
        + energy_gamma * T
    )

    # Embedding stats
    pred_stats = compute_embedding_stats(
        lava_result["predictions"], mask, is_teacher=False,
    )
    teacher_stats = compute_embedding_stats(
        teacher_target, mask, is_teacher=True,
    )
    pred_variance = float(np.var(lava_result["predictions"]))

    result = WindowResult(
        window_index=idx,
        loss=loss,
        forward_ms=lava_result["run_ms"],
        teacher_ms=0.0,
        overhead_ms=lava_result["build_ms"] + lava_result["monitor_ms"],
        total_ms=(t_total_end - t_total_start) * 1000.0,
        embedding_norm_mean=pred_stats["mean"],
        embedding_norm_std=pred_stats["std"],
        teacher_norm_mean=teacher_stats["mean"],
        teacher_norm_std=teacher_stats["std"],
        prediction_variance=pred_variance,
        total_spikes=float(lava_result["total_spikes"]),
        spike_rate=lava_result["spike_rate"],
        synaptic_events=float(lava_result["synaptic_events"]),
        energy_proxy=energy,
    )

    # Save observables (no membrane — monitor removed in production)
    if save_observables:
        obs_dir = output_dir / "observables"
        save_observables_npz(
            obs_dir / f"window_{idx:05d}.npz",
            window_index=idx,
            predictions=lava_result["predictions"],
            teacher_targets=teacher_target,
            spike_vectors_enc=lava_result["enc_spikes"],
            spike_vectors_pred=lava_result["pred_spikes"],
        )

    if idx % 100 == 0:
        logger.info(
            "SNN window %d | loss=%.6f | spikes=%d | rate=%.4f | fwd=%.2fms",
            idx, loss, lava_result["total_spikes"],
            lava_result["spike_rate"], lava_result["run_ms"],
        )

    return result


def run_snn_measurement(
    weights_path: Path | str,
    test_windows: np.ndarray,
    test_masks: List[np.ndarray],
    teacher_targets_dir: Path | str,
    output_dir: Path | str,
    T: int = 128,
    H: int = 128,
    c_mask: float = 1.0,
    energy_alpha: float = 23.6e-12,
    energy_beta: float = 81.0e-12,
    energy_gamma: float = 26.0e-9,
    save_observables: bool = False,
) -> List[WindowResult]:
    """Run full SNN measurement on the test set via Lava.

    Parameters
    ----------
    weights_path : Path or str
        Path to exported weights NPZ file.
    test_windows : np.ndarray
        Test windows, shape ``(N, T, D)``.
    test_masks : list of np.ndarray
        Per-window masks, each shape ``(T,)``.
    teacher_targets_dir : Path or str
        Directory with pre-computed teacher targets.
    output_dir : Path or str
        Output directory.
    T : int
        Window length.
    H : int
        Latent dimension.
    c_mask : float
        Mask bias current magnitude.
    energy_alpha, energy_beta, energy_gamma : float
        Energy proxy coefficients (Eq. 8).
    save_observables : bool
        Whether to save per-timestep observables.

    Returns
    -------
    list of WindowResult
        Per-window measurement results.
    """
    output_dir = Path(output_dir)
    teacher_targets_dir = Path(teacher_targets_dir)

    n_windows = len(test_windows)
    logger.info(
        "SNN measurement: %d windows, sequential execution, "
        "save_observables=%s",
        n_windows, save_observables,
    )

    # Sequential execution — Lava's internal multiprocessing (SharedMemoryManager)
    # is incompatible with multiprocessing.Pool (daemon child limitation).
    results: List[WindowResult] = []
    for idx in range(n_windows):
        result = _process_single_window(
            str(weights_path),
            test_windows[idx],       # (T, D)
            test_masks[idx],         # (T,)
            str(teacher_targets_dir),
            str(output_dir),
            idx,
            T,
            H,
            c_mask,
            energy_alpha,
            energy_beta,
            energy_gamma,
            save_observables,
        )
        results.append(result)
        if (idx + 1) % 100 == 0:
            logger.info(
                "SNN measurement: window %d/%d", idx + 1, n_windows,
            )

    save_results_csv(results, output_dir / "results.csv")
    logger.info("SNN measurement complete: %d windows", n_windows)
    return results


def run_snn_latency_benchmark(
    weights_path: Path | str,
    x_window: np.ndarray,
    mask: np.ndarray,
    T: int = 128,
    H: int = 128,
    c_mask: float = 1.0,
    n_warmup: int = 5,
    n_timed: int = 1000,
) -> BenchmarkResult:
    """Run SNN latency benchmark via Lava.

    No monitors are attached during the benchmark (only component 1 latency).

    Parameters
    ----------
    weights_path : Path or str
        Path to exported weights.
    x_window : np.ndarray
        Single input window, shape ``(T, D)``.
    mask : np.ndarray
        Binary mask, shape ``(T,)``.
    T : int
        Window length.
    H : int
        Latent dimension.
    c_mask : float
        Mask current magnitude.
    n_warmup : int
        Warm-up calls.
    n_timed : int
        Timed calls.

    Returns
    -------
    BenchmarkResult
        Benchmark statistics.
    """
    weights = load_lava_weights(weights_path)
    pos_encoding = sinusoidal_position_encoding_numpy(T, H)

    mask_signal = (mask.reshape(-1, 1) * c_mask).astype(np.float32)

    def forward_fn():
        """Single Lava forward pass (build + run + stop, no monitor)."""
        graph = _build_lava_graph(
            weights, x_window, mask_signal, pos_encoding, T,
            attach_monitor=False,
        )
        graph["lif_enc"].run(
            condition=graph["run_cond"],
            run_cfg=graph["run_cfg"],
        )
        # Retrieve output (minimal)
        _ = graph["output_sink"].data.get()
        graph["lif_enc"].stop()

    result = run_latency_benchmark(
        forward_fn,
        n_warmup=n_warmup,
        n_timed=n_timed,
    )

    logger.info(
        "SNN latency benchmark: median=%.3fms, mean=%.3fms, std=%.3fms",
        result.median_ms, result.mean_ms, result.std_ms,
    )

    return result
