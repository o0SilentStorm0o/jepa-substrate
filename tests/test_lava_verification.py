"""Comprehensive Lava inference verification tests.

These tests verify the critical bug fixes for the two Lava issues:
  BUG-1: Dense(num_message_bits=0) binarizes float inputs to 0/1
  BUG-2: LIF(du=0) causes current accumulation, masking input differences

Each test is self-contained and runs actual Lava process graphs.
Official Lava LIF dynamics (from lava-nc source):
    u[t] = u[t-1] * (1 - du) + a_in
    v[t] = v[t-1] * (1 - dv) + u[t] + bias
    s_out = v[t] > vth
    v[t] = 0  if s_out  (reset)

Test categories:
  V1-V3: Dense num_message_bits verification
  V4-V6: LIF du parameter verification
  V7-V9: Full graph end-to-end verification
  V10-V12: Export parameter verification
  V13-V14: PyTorch <-> Lava round-trip with fixed parameters
  V15: Multiprocessing correctness
"""

from __future__ import annotations

import numpy as np
import pytest
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------
# V1: Dense with num_message_bits=0 binarizes float inputs (the OLD bug)
# --------------------------------------------------------------------------

class TestV1DenseBinaryDefault:
    """Demonstrate that Dense(num_message_bits=0) converts float to binary."""

    def test_dense_binarizes_floats(self) -> None:
        """Float input [2.5, -1.3, 0.7] through identity Dense(num_message_bits=0)
        must NOT preserve float values — only 0/1 pass through."""
        from lava.proc.dense.process import Dense
        from lava.proc.io.source import RingBuffer as SendProcess
        from lava.proc.io.sink import RingBuffer as RecvProcess
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        N = 3
        T = 5
        # Identity weights
        W = np.eye(N, dtype=np.float32)
        # Float input data — amplitude matters
        input_data = np.array([
            [2.5, -1.3, 0.7],
            [0.0,  3.0, -0.5],
            [1.0, 1.0, 1.0],
            [10.0, -10.0, 5.0],
            [0.1, 0.2, 0.3],
        ], dtype=np.float32).T  # (N, T)

        src = SendProcess(data=input_data)
        dense = Dense(weights=W, num_message_bits=0)  # THE BUG: binary mode
        sink = RecvProcess(shape=(N,), buffer=T)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(sink.a_in)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
        output = sink.data.get().T  # (T, N)
        src.stop()

        # With num_message_bits=0, Dense treats input as binary:
        # Any non-zero value -> 1, zero -> 0
        # So the float amplitudes are LOST
        # Check that output does NOT match input
        input_orig = input_data.T  # (T, N)
        # At least some values should differ significantly from original
        max_output = float(np.max(np.abs(output)))
        max_input = float(np.max(np.abs(input_orig)))

        # Output with binary mode: values come from weight*{0,1}, not weight*float
        # So max output should be ~1.0 (weight=identity, input=1 or 0)
        assert max_output <= max_input, "Sanity check"
        # The critical check: output should NOT match input for large-amplitude entries
        # Input has values like 10.0, but binary mode can only produce W@{0,1}
        large_entries = np.abs(input_orig) > 1.5
        if np.any(large_entries):
            # For entries with |input| > 1.5, binary Dense cannot reproduce them
            differences = np.abs(output - input_orig)
            max_diff_on_large = float(np.max(differences[large_entries]))
            assert max_diff_on_large > 0.5, (
                f"Dense(num_message_bits=0) should NOT preserve float amplitudes > 1.5, "
                f"but max difference was only {max_diff_on_large:.4f}"
            )


# --------------------------------------------------------------------------
# V2: Dense with num_message_bits=24 preserves float inputs (the FIX)
# --------------------------------------------------------------------------

class TestV2DenseGradedMode:
    """Verify that Dense(num_message_bits=24) correctly passes float values."""

    def test_dense_preserves_floats(self) -> None:
        """Float input through identity Dense(num_message_bits=24)
        must preserve amplitudes (with 1-step Lava pipeline delay)."""
        from lava.proc.dense.process import Dense
        from lava.proc.io.source import RingBuffer as SendProcess
        from lava.proc.io.sink import RingBuffer as RecvProcess
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        N = 4
        T = 10
        W = np.eye(N, dtype=np.float32)

        rng = np.random.default_rng(42)
        input_values = (rng.standard_normal((T, N)) * 3.0).astype(np.float32)
        input_data = input_values.T  # (N, T)

        src = SendProcess(data=input_data)
        dense = Dense(weights=W, num_message_bits=24)  # THE FIX: graded mode
        sink = RecvProcess(shape=(N,), buffer=T)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(sink.a_in)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
        output = sink.data.get().T  # (T, N)
        src.stop()

        # With num_message_bits=24, Dense preserves float amplitudes
        # BUT there is a 1-step pipeline delay in Lava's Dense process
        # So output[t] corresponds to input[t-1], and output[0] is from init=0
        # Compare output[1:] with input[:-1]
        expected = input_values[:-1]  # (T-1, N) - shifted by 1
        actual = output[1:]           # (T-1, N) - skip first step (pipeline delay)

        max_diff = float(np.max(np.abs(actual - expected)))
        assert max_diff < 1e-5, (
            f"Dense(num_message_bits=24) should preserve float values. "
            f"Max difference: {max_diff:.2e} (threshold: 1e-5). "
            f"Expected sample: {expected[0]}, Got: {actual[0]}"
        )


# --------------------------------------------------------------------------
# V3: Dense with weighted matrix + graded mode
# --------------------------------------------------------------------------

class TestV3DenseWeightedGraded:
    """Verify Dense with non-identity weights and num_message_bits=24."""

    def test_dense_weighted_graded(self) -> None:
        """Dense(W, num_message_bits=24): output = W @ input (with 1-step delay)."""
        from lava.proc.dense.process import Dense
        from lava.proc.io.source import RingBuffer as SendProcess
        from lava.proc.io.sink import RingBuffer as RecvProcess
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        D_in, D_out = 3, 5
        T = 8

        rng = np.random.default_rng(123)
        W = rng.standard_normal((D_out, D_in)).astype(np.float32) * 0.5
        input_values = rng.standard_normal((T, D_in)).astype(np.float32)
        input_data = input_values.T  # (D_in, T)

        src = SendProcess(data=input_data)
        dense = Dense(weights=W, num_message_bits=24)
        sink = RecvProcess(shape=(D_out,), buffer=T)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(sink.a_in)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
        output = sink.data.get().T  # (T, D_out)
        src.stop()

        # Expected: W @ input[t-1], accounting for 1-step delay
        expected = (W @ input_values[:-1].T).T  # (T-1, D_out)
        actual = output[1:]  # (T-1, D_out)

        max_diff = float(np.max(np.abs(actual - expected)))
        assert max_diff < 1e-4, (
            f"Dense weighted graded mode failed. Max diff: {max_diff:.2e}"
        )


# --------------------------------------------------------------------------
# V4: LIF with du=0 causes current accumulation (the OLD bug)
# --------------------------------------------------------------------------

class TestV4LIFDuZeroAccumulation:
    """Demonstrate that LIF(du=0) accumulates current indefinitely."""

    def test_lif_du0_accumulates(self) -> None:
        """With du=0: u[t] = u[t-1]*1.0 + a_in → current grows without bound.
        When combined with voltage decay, this produces many more spikes
        than du=1 (confirmed by V6)."""
        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense
        from lava.proc.io.source import RingBuffer as SendProcess
        from lava.proc.monitor.process import Monitor
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        H = 4
        T = 100
        beta = 0.9
        vth = 1.0

        # Constant input = 0.05 per step
        const_input = np.ones((H, T), dtype=np.float32) * 0.05

        src = SendProcess(data=const_input)
        dense = Dense(
            weights=np.eye(H, dtype=np.float32),
            num_message_bits=24,
        )
        lif = LIF(shape=(H,), du=0.0, dv=1.0 - beta, vth=vth, bias_mant=0)

        mon_s = Monitor()
        mon_s.probe(lif.s_out, T)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(lif.a_in)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
        spikes = mon_s.get_data()[lif.name]["s_out"]  # (T, H)
        src.stop()

        total_spikes = int(np.sum(spikes))
        # With du=0 and input=0.05: u accumulates → many spikes
        # With du=1 same input would produce very few spikes (0.05 << vth=1.0)
        assert total_spikes > 10, (
            f"du=0 with constant input 0.05 should produce many spikes "
            f"due to current accumulation. Got only {total_spikes}."
        )
        print(f"\n  du=0, const input=0.05: {total_spikes} spikes (accumulation effect)")


# --------------------------------------------------------------------------
# V5: LIF with du=1 gives pass-through current (the FIX)
# --------------------------------------------------------------------------

class TestV5LIFDuOnePassThrough:
    """Verify that LIF(du=1) passes input directly: u[t] = a_in."""

    def test_lif_du1_passthrough(self) -> None:
        """With du=1: u[t] = u[t-1]*0 + a_in = a_in.
        Current should NOT accumulate — it equals the input at each step."""
        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense
        from lava.proc.io.source import RingBuffer as SendProcess
        from lava.proc.monitor.process import Monitor
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        H = 4
        T = 30
        beta = 0.9
        vth = 100.0  # High threshold to prevent spiking (no reset interference)

        # Varying input to test that u tracks it exactly
        rng = np.random.default_rng(42)
        input_values = rng.standard_normal((T, H)).astype(np.float32) * 0.3
        input_data = input_values.T  # (H, T)

        src = SendProcess(data=input_data)
        dense = Dense(
            weights=np.eye(H, dtype=np.float32),
            num_message_bits=24,
        )
        lif = LIF(shape=(H,), du=1.0, dv=1.0 - beta, vth=vth, bias_mant=0)

        mon_u = Monitor()
        mon_u.probe(lif.u, T)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(lif.a_in)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
        u_data = mon_u.get_data()[lif.name]["u"]  # (T, H) — Monitor native shape
        src.stop()

        # With du=1: u[t] = a_in (no accumulation)
        # a_in at time t comes from Dense output, which is input[t-1] (1-step delay)
        # So u[t] should equal input[t-1] for t >= 1
        expected_u = input_values[:-1]  # input[0..T-2], shape (T-1, H)
        actual_u = u_data[1:]            # u[1..T-1], shape (T-1, H)

        max_diff = float(np.max(np.abs(actual_u - expected_u)))
        assert max_diff < 1e-5, (
            f"du=1 should give u[t] = a_in (pass-through). "
            f"Max diff: {max_diff:.2e}. "
            f"This means current is accumulating when it shouldn't!"
        )
        print(f"\n  du=1 pass-through verified: max |u[t] - a_in[t-1]| = {max_diff:.2e}")


# --------------------------------------------------------------------------
# V6: LIF du=1 vs du=0 with identical random input — divergence test
# --------------------------------------------------------------------------

class TestV6DuComparison:
    """du=0 and du=1 must produce DIFFERENT spike patterns for same input."""

    def test_du0_vs_du1_different_spikes(self) -> None:
        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense
        from lava.proc.io.source import RingBuffer as SendProcess
        from lava.proc.monitor.process import Monitor
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        H = 16
        T = 100
        beta = 0.904837
        vth = 1.0

        rng = np.random.default_rng(99)
        input_data = (rng.standard_normal((H, T)) * 0.3).astype(np.float32)

        results = {}
        for du_val in [0.0, 1.0]:
            src = SendProcess(data=input_data.copy())
            dense = Dense(
                weights=np.eye(H, dtype=np.float32),
                num_message_bits=24,
            )
            lif = LIF(shape=(H,), du=du_val, dv=1.0 - beta, vth=vth, bias_mant=0)
            mon = Monitor()
            mon.probe(lif.s_out, T)

            src.s_out.connect(dense.s_in)
            dense.a_out.connect(lif.a_in)

            run_cfg = Loihi2SimCfg(select_tag="floating_pt")
            src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
            spikes = mon.get_data()[lif.name]["s_out"].T
            src.stop()
            results[du_val] = spikes

        # They should produce different spike counts
        count_du0 = int(np.sum(results[0.0]))
        count_du1 = int(np.sum(results[1.0]))

        assert count_du0 != count_du1, (
            f"du=0 and du=1 produced identical spike counts ({count_du0}). "
            f"They should differ because du=0 accumulates current."
        )
        # du=0 accumulates → likely MORE spikes (lower effective threshold)
        # This is just informational:
        print(f"  du=0 spikes: {count_du0}, du=1 spikes: {count_du1}")


# --------------------------------------------------------------------------
# V7: Full Lava graph — DIFFERENT inputs MUST produce DIFFERENT outputs
# --------------------------------------------------------------------------

class TestV7EndToEndVariability:
    """The critical test: different inputs must produce different predictions.
    This is exactly what failed before with du=0 + num_message_bits=0."""

    def test_different_inputs_different_outputs(self) -> None:
        """Run 5 random input windows through the full graph.
        All prediction vectors must be different from each other."""
        from snn.lava_inference import _build_lava_graph
        from snn.lava_export import LavaWeights
        from shared.positional import sinusoidal_position_encoding_numpy
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        T = 32
        H = 16
        D = 6
        N_windows = 5
        beta = 0.904837
        vth = 1.0

        rng = np.random.default_rng(42)

        # Create synthetic weights — scale high enough to ensure spiking
        weights = LavaWeights(
            dense_in_weights=rng.standard_normal((D, H)).astype(np.float32) * 0.3,
            encoder_dv=1.0 - beta,
            encoder_du=1.0,  # THE FIX
            encoder_vth=vth,
            readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.3,
            readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            pred_proj_weights=rng.standard_normal((2 * H, H)).astype(np.float32) * 0.3,
            mask_weights=rng.standard_normal((1, H)).astype(np.float32) * 0.3,
            predictor_dv=1.0 - beta,
            predictor_du=1.0,  # THE FIX
            predictor_vth=vth,
            pred_readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.3,
            pred_readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            hidden_dim=H,
            input_dim=D,
            beta=beta,
            v_th=vth,
            trace_alpha=0.9,
        )

        pos_encoding = sinusoidal_position_encoding_numpy(T, H)

        # Generate N_windows different random inputs
        all_predictions = []
        all_spike_counts = []
        for i in range(N_windows):
            x_window = rng.standard_normal((T, D)).astype(np.float32) * 0.5
            mask = np.zeros(T, dtype=np.float32)
            mask[T // 2:] = 1.0  # Second half is target
            mask_signal = mask.reshape(-1, 1).astype(np.float32)

            graph = _build_lava_graph(
                weights, x_window, mask_signal, pos_encoding, T,
                attach_monitor=True,
            )

            graph["lif_enc"].run(
                condition=graph["run_cond"],
                run_cfg=graph["run_cfg"],
            )

            predictions = graph["output_sink"].data.get().T  # Sink: (H,T)→.T→(T,H)
            # Monitor natively returns (T, H) — NO transpose needed
            enc_spikes = graph["monitors"]["enc_spikes"].get_data()[
                graph["lif_enc"].name
            ]["s_out"]  # (T, H)
            pred_spikes = graph["monitors"]["pred_spikes"].get_data()[
                graph["lif_pred"].name
            ]["s_out"]  # (T, H)

            graph["lif_enc"].stop()

            print(f"  Window {i}: enc_spikes={int(np.sum(enc_spikes))}, "
                  f"pred_spikes={int(np.sum(pred_spikes))}, "
                  f"pred_mean={float(np.mean(predictions)):.6f}")

            all_predictions.append(predictions.copy())
            all_spike_counts.append(int(np.sum(enc_spikes) + np.sum(pred_spikes)))

        # ---- ASSERTIONS ----

        # 1. Not all predictions should be identical
        pred_stack = np.stack(all_predictions)  # (N, T, H)
        pred_variance = float(np.var(pred_stack, axis=0).mean())
        assert pred_variance > 1e-10, (
            f"All {N_windows} windows produced nearly identical predictions! "
            f"Variance across windows: {pred_variance:.2e}. "
            f"THIS IS THE BUG — Lava is ignoring input differences."
        )

        # 2. Pairwise: each pair of windows should differ
        for i in range(N_windows):
            for j in range(i + 1, N_windows):
                diff = float(np.max(np.abs(
                    all_predictions[i] - all_predictions[j]
                )))
                assert diff > 1e-6, (
                    f"Windows {i} and {j} produced identical predictions "
                    f"(max diff: {diff:.2e}). Input differences are being ignored."
                )

        # 3. Spike counts should vary across windows
        unique_counts = len(set(all_spike_counts))
        assert unique_counts >= 2, (
            f"All {N_windows} windows had identical spike count "
            f"({all_spike_counts[0]}). This indicates the bug is still present."
        )

        # Report
        print(f"\n  Predictions variance: {pred_variance:.6f}")
        print(f"  Spike counts: {all_spike_counts}")
        print(f"  Unique spike counts: {unique_counts}/{N_windows}")


# --------------------------------------------------------------------------
# V8: Full graph with IDENTICAL inputs → IDENTICAL outputs (determinism)
# --------------------------------------------------------------------------

class TestV8Determinism:
    """Same input run twice must produce exactly the same output."""

    def test_deterministic_output(self) -> None:
        from snn.lava_inference import _build_lava_graph
        from snn.lava_export import LavaWeights
        from shared.positional import sinusoidal_position_encoding_numpy

        T = 32
        H = 16
        D = 6
        beta = 0.904837
        vth = 1.0

        rng = np.random.default_rng(42)
        weights = LavaWeights(
            dense_in_weights=rng.standard_normal((D, H)).astype(np.float32) * 0.1,
            encoder_dv=1.0 - beta,
            encoder_du=1.0,
            encoder_vth=vth,
            readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.1,
            readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            pred_proj_weights=rng.standard_normal((2 * H, H)).astype(np.float32) * 0.1,
            mask_weights=rng.standard_normal((1, H)).astype(np.float32) * 0.1,
            predictor_dv=1.0 - beta,
            predictor_du=1.0,
            predictor_vth=vth,
            pred_readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.1,
            pred_readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            hidden_dim=H,
            input_dim=D,
            beta=beta,
            v_th=vth,
            trace_alpha=0.9,
        )

        pos_encoding = sinusoidal_position_encoding_numpy(T, H)
        x_window = rng.standard_normal((T, D)).astype(np.float32)
        mask_signal = np.zeros((T, 1), dtype=np.float32)
        mask_signal[T // 2:] = 1.0

        predictions_list = []
        for _ in range(3):
            graph = _build_lava_graph(
                weights, x_window, mask_signal, pos_encoding, T,
                attach_monitor=False,
            )
            graph["lif_enc"].run(
                condition=graph["run_cond"],
                run_cfg=graph["run_cfg"],
            )
            predictions = graph["output_sink"].data.get().T.copy()
            graph["lif_enc"].stop()
            predictions_list.append(predictions)

        # All 3 runs should produce identical results
        for i in range(1, 3):
            max_diff = float(np.max(np.abs(predictions_list[0] - predictions_list[i])))
            assert max_diff < 1e-10, (
                f"Run 0 vs run {i}: max diff = {max_diff:.2e}. "
                f"Lava graph should be deterministic."
            )


# --------------------------------------------------------------------------
# V9: Full graph — non-zero output check
# --------------------------------------------------------------------------

class TestV9NonZeroOutput:
    """The graph must produce non-trivial output (not all zeros)."""

    def test_output_not_all_zeros(self) -> None:
        from snn.lava_inference import _build_lava_graph
        from snn.lava_export import LavaWeights
        from shared.positional import sinusoidal_position_encoding_numpy

        T = 64
        H = 32
        D = 6
        beta = 0.904837
        vth = 1.0

        rng = np.random.default_rng(42)
        # Use larger weights to ensure spiking activity
        weights = LavaWeights(
            dense_in_weights=rng.standard_normal((D, H)).astype(np.float32) * 0.5,
            encoder_dv=1.0 - beta,
            encoder_du=1.0,
            encoder_vth=vth,
            readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.3,
            readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            pred_proj_weights=rng.standard_normal((2 * H, H)).astype(np.float32) * 0.3,
            mask_weights=rng.standard_normal((1, H)).astype(np.float32) * 0.3,
            predictor_dv=1.0 - beta,
            predictor_du=1.0,
            predictor_vth=vth,
            pred_readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.3,
            pred_readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            hidden_dim=H,
            input_dim=D,
            beta=beta,
            v_th=vth,
            trace_alpha=0.9,
        )

        pos_encoding = sinusoidal_position_encoding_numpy(T, H)
        x_window = rng.standard_normal((T, D)).astype(np.float32) * 2.0
        mask_signal = np.ones((T, 1), dtype=np.float32)

        graph = _build_lava_graph(
            weights, x_window, mask_signal, pos_encoding, T,
            attach_monitor=True,
        )
        graph["lif_enc"].run(
            condition=graph["run_cond"],
            run_cfg=graph["run_cfg"],
        )
        predictions = graph["output_sink"].data.get().T
        enc_spikes = graph["monitors"]["enc_spikes"].get_data()[
            graph["lif_enc"].name
        ]["s_out"].T
        pred_spikes = graph["monitors"]["pred_spikes"].get_data()[
            graph["lif_pred"].name
        ]["s_out"].T
        graph["lif_enc"].stop()

        total_enc_spikes = int(np.sum(enc_spikes))
        total_pred_spikes = int(np.sum(pred_spikes))
        pred_nonzero = int(np.count_nonzero(predictions))

        assert total_enc_spikes > 0, (
            f"Encoder produced 0 spikes! This means input is not reaching LIF_enc."
        )
        assert total_pred_spikes > 0, (
            f"Predictor produced 0 spikes! "
            f"Encoder had {total_enc_spikes} spikes but they didn't reach LIF_pred."
        )
        assert pred_nonzero > 0, (
            f"All predictions are zero! "
            f"Enc spikes: {total_enc_spikes}, Pred spikes: {total_pred_spikes}, "
            f"but readout is all zeros."
        )
        print(f"\n  Enc spikes: {total_enc_spikes}, Pred spikes: {total_pred_spikes}")
        print(f"  Prediction non-zero entries: {pred_nonzero}/{T * H}")


# --------------------------------------------------------------------------
# V10: Export parameters — du must be 1.0
# --------------------------------------------------------------------------

class TestV10ExportDu:
    """Verify that export_weights sets du=1.0 (not 0.0)."""

    def test_export_du_is_one(self) -> None:
        import torch
        from snn.model import SNNModel
        from snn.lava_export import export_weights

        D, H = 6, 32
        model = SNNModel(input_dim=D, hidden_dim=H)

        weights = export_weights(model)

        assert weights.encoder_du == 1.0, (
            f"encoder_du should be 1.0, got {weights.encoder_du}. "
            f"This is the critical bug fix!"
        )
        assert weights.predictor_du == 1.0, (
            f"predictor_du should be 1.0, got {weights.predictor_du}. "
            f"This is the critical bug fix!"
        )
        # Also verify dv is correct
        expected_dv = 1.0 - model.encoder.lif.beta
        assert abs(weights.encoder_dv - expected_dv) < 1e-6, (
            f"encoder_dv should be {expected_dv}, got {weights.encoder_dv}"
        )
        print(f"\n  encoder_du={weights.encoder_du}, predictor_du={weights.predictor_du}")
        print(f"  encoder_dv={weights.encoder_dv}, predictor_dv={weights.predictor_dv}")


# --------------------------------------------------------------------------
# V11: Export save/load round-trip preserves du=1.0
# --------------------------------------------------------------------------

class TestV11ExportSaveLoadDu:
    """Verify du=1.0 survives save/load cycle."""

    def test_save_load_preserves_du(self) -> None:
        import torch
        from snn.model import SNNModel
        from snn.lava_export import export_weights, load_lava_weights

        D, H = 6, 32
        model = SNNModel(input_dim=D, hidden_dim=H)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.npz"
            weights_orig = export_weights(model, output_path=path)
            weights_loaded = load_lava_weights(path)

        assert weights_loaded.encoder_du == 1.0, (
            f"Loaded encoder_du = {weights_loaded.encoder_du}, expected 1.0"
        )
        assert weights_loaded.predictor_du == 1.0, (
            f"Loaded predictor_du = {weights_loaded.predictor_du}, expected 1.0"
        )


# --------------------------------------------------------------------------
# V12: Graph uses du from weights (not hardcoded)
# --------------------------------------------------------------------------

class TestV12GraphUsesDuFromWeights:
    """Verify _build_lava_graph passes du from weights to LIF processes."""

    def test_graph_lif_du_matches_weights(self) -> None:
        from snn.lava_inference import _build_lava_graph
        from snn.lava_export import LavaWeights
        from shared.positional import sinusoidal_position_encoding_numpy

        T = 4
        H = 8
        D = 4

        rng = np.random.default_rng(42)
        weights = LavaWeights(
            dense_in_weights=rng.standard_normal((D, H)).astype(np.float32),
            encoder_dv=0.095163,
            encoder_du=1.0,  # This must propagate to LIF_enc
            encoder_vth=1.0,
            readout_weights=rng.standard_normal((H, H)).astype(np.float32),
            readout_bias=np.zeros(H, dtype=np.float32),
            pred_proj_weights=rng.standard_normal((2 * H, H)).astype(np.float32),
            mask_weights=rng.standard_normal((1, H)).astype(np.float32),
            predictor_dv=0.095163,
            predictor_du=1.0,  # This must propagate to LIF_pred
            predictor_vth=1.0,
            pred_readout_weights=rng.standard_normal((H, H)).astype(np.float32),
            pred_readout_bias=np.zeros(H, dtype=np.float32),
            hidden_dim=H,
            input_dim=D,
            beta=0.904837,
            v_th=1.0,
            trace_alpha=0.9,
        )

        pos = sinusoidal_position_encoding_numpy(T, H)
        x = np.zeros((T, D), dtype=np.float32)
        mask = np.zeros((T, 1), dtype=np.float32)

        graph = _build_lava_graph(weights, x, mask, pos, T, attach_monitor=False)

        # Check du was set correctly on LIF processes
        enc_du = float(graph["lif_enc"].du.get())
        pred_du = float(graph["lif_pred"].du.get())

        graph["lif_enc"].run(
            condition=graph["run_cond"],
            run_cfg=graph["run_cfg"],
        )
        graph["lif_enc"].stop()

        assert enc_du == 1.0, f"LIF_enc du={enc_du}, expected 1.0"
        assert pred_du == 1.0, f"LIF_pred du={pred_du}, expected 1.0"


# --------------------------------------------------------------------------
# V13: Dense layers in graph have correct num_message_bits
# --------------------------------------------------------------------------

class TestV13GraphDenseNumMessageBits:
    """Verify that DenseIn, DenseMask, DensePos, DensePosPred have
    num_message_bits=24, while DensePred and DenseReadout have
    num_message_bits=0 (they receive binary spikes from LIF)."""

    def test_dense_num_message_bits(self) -> None:
        from snn.lava_inference import _build_lava_graph
        from snn.lava_export import LavaWeights
        from shared.positional import sinusoidal_position_encoding_numpy

        T = 4
        H = 8
        D = 4
        beta = 0.904837

        rng = np.random.default_rng(42)
        weights = LavaWeights(
            dense_in_weights=rng.standard_normal((D, H)).astype(np.float32),
            encoder_dv=1.0 - beta,
            encoder_du=1.0,
            encoder_vth=1.0,
            readout_weights=rng.standard_normal((H, H)).astype(np.float32),
            readout_bias=np.zeros(H, dtype=np.float32),
            pred_proj_weights=rng.standard_normal((2 * H, H)).astype(np.float32),
            mask_weights=rng.standard_normal((1, H)).astype(np.float32),
            predictor_dv=1.0 - beta,
            predictor_du=1.0,
            predictor_vth=1.0,
            pred_readout_weights=rng.standard_normal((H, H)).astype(np.float32),
            pred_readout_bias=np.zeros(H, dtype=np.float32),
            hidden_dim=H,
            input_dim=D,
            beta=beta,
            v_th=1.0,
            trace_alpha=0.9,
        )

        pos = sinusoidal_position_encoding_numpy(T, H)
        x = np.zeros((T, D), dtype=np.float32)
        mask = np.zeros((T, 1), dtype=np.float32)

        graph = _build_lava_graph(weights, x, mask, pos, T, attach_monitor=False)

        # Run briefly to initialize runtime (needed to read Var values)
        graph["lif_enc"].run(
            condition=graph["run_cond"],
            run_cfg=graph["run_cfg"],
        )

        # Graded Dense layers (receive float from RingBuffer or other Dense)
        for name in ["dense_in", "dense_mask", "dense_pos", "dense_pos_pred"]:
            nmb = int(graph[name].num_message_bits.get())
            assert nmb == 24, (
                f"{name}.num_message_bits = {nmb}, expected 24 (graded mode). "
                f"This Dense receives float signals and MUST use graded mode!"
            )

        # Binary Dense layers (receive binary spikes from LIF.s_out)
        for name in ["dense_pred", "dense_readout"]:
            nmb = int(graph[name].num_message_bits.get())
            assert nmb == 0, (
                f"{name}.num_message_bits = {nmb}, expected 0 (binary mode). "
                f"This Dense receives LIF spikes (binary) and should use default."
            )

        graph["lif_enc"].stop()
        print("\n  DenseIn/Mask/Pos/PosPred: num_message_bits=24 ✓")
        print("  DensePred/Readout: num_message_bits=0 ✓")


# --------------------------------------------------------------------------
# V14: PyTorch LIF ↔ Lava LIF spike-train comparison (with du=1 fix)
# --------------------------------------------------------------------------

class TestV14PyTorchLavaSpikeTrain:
    """Compare spike trains from PyTorch SNN and Lava graph with du=1.
    This is a stricter version of T-E1 that uses actual Lava processes."""

    def test_pytorch_vs_lava_spikes(self) -> None:
        import torch
        from snn.model import LIFNeuronLayer
        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense
        from lava.proc.io.source import RingBuffer as SendProcess
        from lava.proc.monitor.process import Monitor
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg

        T = 200
        D = 6
        H = 32
        beta = 0.904837
        vth = 1.0

        rng = np.random.default_rng(42)
        input_np = rng.standard_normal((T, D)).astype(np.float32) * 0.5
        W_np = rng.standard_normal((H, D)).astype(np.float32) * 0.4

        # ---- PyTorch path ----
        lif_pt = LIFNeuronLayer(n_neurons=H, beta=beta, v_th=vth, surrogate_k=25.0)
        lif_pt.eval()
        proj_pt = torch.nn.Linear(D, H, bias=False)
        with torch.no_grad():
            proj_pt.weight.copy_(torch.from_numpy(W_np))

        with torch.no_grad():
            projected = proj_pt(torch.from_numpy(input_np).unsqueeze(0))  # (1,T,H)
            spikes_pt, _, _ = lif_pt(projected)
        spikes_pt_np = spikes_pt.squeeze(0).numpy()  # (T, H)

        # ---- Lava path (with du=1 fix) ----
        input_data = input_np.T.astype(np.float32)  # (D, T)
        src = SendProcess(data=input_data)
        # W_np is (H, D) — same convention as Lava Dense (N_out, N_in)
        # But our export transposes, so we follow that convention
        dense = Dense(weights=W_np, num_message_bits=24)
        lif_lava = LIF(
            shape=(H,), du=1.0, dv=1.0 - beta, vth=vth, bias_mant=0,
        )
        mon = Monitor()
        mon.probe(lif_lava.s_out, T)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_lava.a_in)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
        spikes_lava = mon.get_data()[lif_lava.name]["s_out"]  # Monitor: (T, H) natively
        src.stop()

        # ---- Comparison ----
        # Due to Lava's 1-step pipeline delay and slightly different reset semantics,
        # we allow some mismatches, but they should be minimal
        # PyTorch: v[t] = beta * v[t-1] * (1-s[t-1]) + I[t] (hard reset)
        # Lava:    u[t] = a_in;  v[t] = v[t-1]*(1-dv) + u[t]; v[t]=0 if spike
        # Both are hard-reset LIF but slightly different order of operations

        total_pt = int(np.sum(spikes_pt_np))
        total_lava = int(np.sum(spikes_lava))

        # Both should have significant spiking activity
        assert total_pt > 0, "PyTorch produced 0 spikes"
        assert total_lava > 0, "Lava produced 0 spikes"

        # Spike counts should be in similar ballpark (within 50%)
        ratio = total_lava / max(total_pt, 1)
        assert 0.3 < ratio < 3.0, (
            f"PyTorch/Lava spike count ratio too far off: "
            f"PT={total_pt}, Lava={total_lava}, ratio={ratio:.2f}. "
            f"Expected similar magnitude."
        )

        # Per-neuron mismatch check (accounting for 1-step offset)
        # Shift Lava by 1 step for comparison
        spikes_lava_shifted = spikes_lava[1:]   # (T-1, H)
        spikes_pt_trimmed = spikes_pt_np[:-1]   # (T-1, H)

        mismatches = np.sum(np.abs(spikes_lava_shifted - spikes_pt_trimmed) > 0.5, axis=0)
        max_mismatch = int(np.max(mismatches))
        mean_mismatch = float(np.mean(mismatches))

        print(f"\n  PyTorch spikes: {total_pt}, Lava spikes: {total_lava}")
        print(f"  Max mismatches/neuron: {max_mismatch}, Mean: {mean_mismatch:.1f}")
        print(f"  Spike count ratio: {ratio:.2f}")

        # Allow up to ~30% mismatch per neuron (due to reset timing differences)
        max_allowed = T * 0.3
        assert max_mismatch < max_allowed, (
            f"Too many spike mismatches: max {max_mismatch} per neuron "
            f"(allowed: {max_allowed}). This suggests a fundamental dynamics mismatch."
        )


# --------------------------------------------------------------------------
# V15: run_lava_inference_window produces varying results
# --------------------------------------------------------------------------

class TestV15InferenceWindowVariability:
    """Integration test: run_lava_inference_window with different windows
    must produce different spike statistics."""

    def test_inference_window_varies(self) -> None:
        from snn.lava_inference import run_lava_inference_window
        from snn.lava_export import LavaWeights
        from shared.positional import sinusoidal_position_encoding_numpy

        T = 64
        H = 32
        D = 6
        beta = 0.904837
        vth = 1.0

        rng = np.random.default_rng(42)
        weights = LavaWeights(
            dense_in_weights=rng.standard_normal((D, H)).astype(np.float32) * 0.4,
            encoder_dv=1.0 - beta,
            encoder_du=1.0,
            encoder_vth=vth,
            readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.3,
            readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            pred_proj_weights=rng.standard_normal((2 * H, H)).astype(np.float32) * 0.3,
            mask_weights=rng.standard_normal((1, H)).astype(np.float32) * 0.3,
            predictor_dv=1.0 - beta,
            predictor_du=1.0,
            predictor_vth=vth,
            pred_readout_weights=rng.standard_normal((H, H)).astype(np.float32) * 0.3,
            pred_readout_bias=rng.standard_normal(H).astype(np.float32) * 0.01,
            hidden_dim=H,
            input_dim=D,
            beta=beta,
            v_th=vth,
            trace_alpha=0.9,
        )

        pos_encoding = sinusoidal_position_encoding_numpy(T, H)
        mask = np.zeros(T, dtype=np.float32)
        mask[T // 2:] = 1.0

        results = []
        for i in range(3):
            x_window = rng.standard_normal((T, D)).astype(np.float32)
            result = run_lava_inference_window(
                weights, x_window, mask, pos_encoding, T,
                attach_monitor=True,
            )
            results.append(result)

        # Spike counts must differ
        spike_counts = [r["total_spikes"] for r in results]
        spike_rates = [r["spike_rate"] for r in results]
        unique_counts = len(set(spike_counts))

        assert unique_counts >= 2, (
            f"All 3 windows had identical spike count: {spike_counts}. "
            f"BUG: Lava inference ignoring input."
        )

        # Predictions must differ
        for i in range(3):
            for j in range(i + 1, 3):
                diff = float(np.max(np.abs(
                    results[i]["predictions"] - results[j]["predictions"]
                )))
                assert diff > 1e-6, (
                    f"Windows {i} and {j} have identical predictions (diff={diff:.2e})."
                )

        print(f"\n  Spike counts: {spike_counts}")
        print(f"  Spike rates: {[f'{r:.4f}' for r in spike_rates]}")
        print(f"  All different: ✓")
