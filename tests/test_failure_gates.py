"""Acceptance tests: failure-mode gate tests T-F1 through T-F3.

T-F1: Collapse gate -- Var[z_hat] < 1e-3 for 3 consecutive checks
      triggers INVALID (gate.passed = False).
T-F2: Saturation gate -- mean spike rate > 0.9 triggers INVALID.
T-F3: Silence gate -- total spike count = 0 triggers INVALID.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shared.gates import (
    CollapseGate,
    SaturationGate,
    SilenceGate,
    FailureGateChecker,
)


# -----------------------------------------------------------------------
# T-F1: Collapse gate
# -----------------------------------------------------------------------

class TestF1CollapseGate:
    """T-F1: If Var[z_hat] < threshold for n_consecutive checks,
    the gate fires (passed=False)."""

    def test_fires_after_consecutive_low_variance(self) -> None:
        gate = CollapseGate(threshold=1e-3, n_consecutive=3)

        for i in range(3):
            z_hat = torch.ones(128, 128) * 0.5
            z_hat = z_hat + torch.randn_like(z_hat) * 1e-4
            result = gate.check(z_hat)

        assert not result.passed, (
            "Collapse gate should have fired (passed=False) after 3 "
            "consecutive low-variance checks"
        )

    def test_does_not_fire_with_intermittent_high_variance(self) -> None:
        gate = CollapseGate(threshold=1e-3, n_consecutive=3)

        low = torch.ones(128, 128) * 0.5
        high = torch.randn(128, 128)

        gate.check(low)
        gate.check(low)
        gate.check(high)  # high -> resets counter
        gate.check(low)
        result = gate.check(low)  # only 2 consecutive

        assert result.passed, (
            "Collapse gate should NOT fire when high-variance check "
            "interrupts the consecutive run"
        )

    def test_does_not_fire_with_healthy_variance(self) -> None:
        gate = CollapseGate(threshold=1e-3, n_consecutive=3)

        for _ in range(10):
            z_hat = torch.randn(128, 128)
            result = gate.check(z_hat)

        assert result.passed


# -----------------------------------------------------------------------
# T-F2: Saturation gate
# -----------------------------------------------------------------------

class TestF2SaturationGate:
    """T-F2: If mean spike rate > threshold, the gate fires."""

    def test_fires_on_high_spike_rate(self) -> None:
        gate = SaturationGate(threshold=0.9)

        spike_counts = np.ones(128, dtype=np.float32) * 128
        result = gate.check(spike_counts, n_neurons=128, n_steps=128)

        assert not result.passed, (
            "Saturation gate should fire when spike rate = 1.0 > 0.9"
        )

    def test_does_not_fire_on_normal_rate(self) -> None:
        gate = SaturationGate(threshold=0.9)

        rng = np.random.default_rng(42)
        spike_counts = rng.binomial(128, 0.05, size=128).astype(np.float32)
        result = gate.check(spike_counts, n_neurons=128, n_steps=128)

        assert result.passed, (
            "Saturation gate should NOT fire at ~5% spike rate"
        )

    def test_fires_at_boundary(self) -> None:
        gate = SaturationGate(threshold=0.9)

        spike_counts = np.ones(128, dtype=np.float32) * (128 * 0.91)
        result = gate.check(spike_counts, n_neurons=128, n_steps=128)

        assert not result.passed


# -----------------------------------------------------------------------
# T-F3: Silence gate
# -----------------------------------------------------------------------

class TestF3SilenceGate:
    """T-F3: If total spike count = 0, the gate fires."""

    def test_fires_on_zero_spikes(self) -> None:
        gate = SilenceGate()

        result = gate.check(total_spikes=0)

        assert not result.passed, (
            "Silence gate should fire when total spike count = 0"
        )

    def test_does_not_fire_with_any_spikes(self) -> None:
        gate = SilenceGate()

        result = gate.check(total_spikes=1)

        assert result.passed, (
            "Silence gate should NOT fire when there is at least 1 spike"
        )


# -----------------------------------------------------------------------
# Composite gate checker
# -----------------------------------------------------------------------

class TestFailureGateChecker:
    """Test the combined FailureGateChecker with all three gates."""

    def test_detects_collapse(self) -> None:
        checker = FailureGateChecker(
            collapse_threshold=1e-3,
            collapse_consecutive=3,
            saturation_threshold=0.9,
        )

        low_var = torch.ones(128, 128) * 0.5
        normal_spikes = np.ones(128, dtype=np.float32)

        for _ in range(3):
            checker.check_collapse(low_var)
            checker.check_saturation(normal_spikes, n_neurons=128, n_steps=128)
            checker.check_silence(total_spikes=128)

        assert not checker.all_passed()

    def test_detects_saturation(self) -> None:
        checker = FailureGateChecker(
            collapse_threshold=1e-3,
            collapse_consecutive=3,
            saturation_threshold=0.9,
        )

        healthy_embeddings = torch.randn(128, 128)
        saturated_spikes = np.ones(128, dtype=np.float32) * 128

        checker.check_collapse(healthy_embeddings)
        checker.check_saturation(saturated_spikes, n_neurons=128, n_steps=128)
        checker.check_silence(total_spikes=128 * 128)

        assert not checker.all_passed()

    def test_detects_silence(self) -> None:
        checker = FailureGateChecker(
            collapse_threshold=1e-3,
            collapse_consecutive=3,
            saturation_threshold=0.9,
        )

        healthy_embeddings = torch.randn(128, 128)
        healthy_spikes = np.ones(128, dtype=np.float32) * 5

        checker.check_collapse(healthy_embeddings)
        checker.check_saturation(healthy_spikes, n_neurons=128, n_steps=128)
        checker.check_silence(total_spikes=0)

        assert not checker.all_passed()

    def test_passes_healthy(self) -> None:
        checker = FailureGateChecker(
            collapse_threshold=1e-3,
            collapse_consecutive=3,
            saturation_threshold=0.9,
        )

        healthy_embeddings = torch.randn(128, 128)
        healthy_spikes = np.ones(128, dtype=np.float32) * 5

        checker.check_collapse(healthy_embeddings)
        checker.check_saturation(healthy_spikes, n_neurons=128, n_steps=128)
        checker.check_silence(total_spikes=640)

        assert checker.all_passed()
