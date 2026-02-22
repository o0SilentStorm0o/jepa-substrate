"""Failure-mode gates for experiment validity.

Implements gates F1--F3 from the specification:
  F1: Representation collapse (Var[z_hat] < 1e-3 for 3 consecutive epochs)
  F2: SNN spike saturation (mean spike rate > 0.9)
  F3: SNN silence (total spike count = 0 for any test window)

Each gate returns a status object indicating whether the run should be
flagged INVALID and halted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a failure-mode gate check."""
    gate_id: str
    passed: bool
    value: float
    threshold: float
    message: str


class CollapseGate:
    """F1: Representation collapse detector.

    If Var[z_hat] < threshold for ``n_consecutive`` consecutive epochs,
    the run is flagged INVALID and halted.
    """

    def __init__(
        self,
        threshold: float = 1e-3,
        n_consecutive: int = 3,
    ) -> None:
        self.threshold = threshold
        self.n_consecutive = n_consecutive
        self._below_count = 0

    def check(self, predictions: torch.Tensor) -> GateResult:
        """Check for representation collapse.

        Parameters
        ----------
        predictions : torch.Tensor
            Predictor outputs, shape ``(N, H)`` or ``(B, T, H)``.

        Returns
        -------
        GateResult
            Gate check result.
        """
        variance = float(predictions.var().item())
        if variance < self.threshold:
            self._below_count += 1
        else:
            self._below_count = 0

        passed = self._below_count < self.n_consecutive
        msg = (
            f"F1 collapse gate: Var[z_hat] = {variance:.6e}, "
            f"threshold = {self.threshold:.6e}, "
            f"consecutive below = {self._below_count}/{self.n_consecutive}"
        )
        if not passed:
            logger.warning("GATE FAILED: %s", msg)

        return GateResult(
            gate_id="F1",
            passed=passed,
            value=variance,
            threshold=self.threshold,
            message=msg,
        )

    def reset(self) -> None:
        """Reset the consecutive counter."""
        self._below_count = 0


class SaturationGate:
    """F2: SNN spike saturation detector.

    If mean spike rate > threshold over an epoch, flag and halt.
    """

    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold

    def check(self, spike_counts: np.ndarray, n_neurons: int, n_steps: int) -> GateResult:
        """Check for spike saturation.

        Parameters
        ----------
        spike_counts : np.ndarray
            Total spike count per neuron over the epoch.
        n_neurons : int
            Number of neurons.
        n_steps : int
            Number of simulation steps.

        Returns
        -------
        GateResult
            Gate check result.
        """
        mean_rate = float(np.sum(spike_counts)) / (n_neurons * n_steps)
        passed = mean_rate <= self.threshold
        msg = (
            f"F2 saturation gate: mean spike rate = {mean_rate:.4f}, "
            f"threshold = {self.threshold}"
        )
        if not passed:
            logger.warning("GATE FAILED: %s", msg)

        return GateResult(
            gate_id="F2",
            passed=passed,
            value=mean_rate,
            threshold=self.threshold,
            message=msg,
        )


class SilenceGate:
    """F3: SNN silence detector.

    If total spike count = 0 for any test window, flag and halt.
    """

    def __init__(self) -> None:
        self.threshold = 0

    def check(self, total_spikes: int) -> GateResult:
        """Check for zero activity.

        Parameters
        ----------
        total_spikes : int
            Total spike count for a single window.

        Returns
        -------
        GateResult
            Gate check result.
        """
        passed = total_spikes > 0
        msg = (
            f"F3 silence gate: total spikes = {total_spikes}"
        )
        if not passed:
            logger.warning("GATE FAILED: %s", msg)

        return GateResult(
            gate_id="F3",
            passed=passed,
            value=float(total_spikes),
            threshold=0.0,
            message=msg,
        )


class FailureGateChecker:
    """Aggregated failure-mode gate checker.

    Combines F1, F2, F3 into a single interface.
    """

    def __init__(
        self,
        collapse_threshold: float = 1e-3,
        collapse_consecutive: int = 3,
        saturation_threshold: float = 0.9,
    ) -> None:
        self.collapse_gate = CollapseGate(collapse_threshold, collapse_consecutive)
        self.saturation_gate = SaturationGate(saturation_threshold)
        self.silence_gate = SilenceGate()
        self._results: List[GateResult] = []

    def check_collapse(self, predictions: torch.Tensor) -> GateResult:
        result = self.collapse_gate.check(predictions)
        self._results.append(result)
        return result

    def check_saturation(
        self, spike_counts: np.ndarray, n_neurons: int, n_steps: int
    ) -> GateResult:
        result = self.saturation_gate.check(spike_counts, n_neurons, n_steps)
        self._results.append(result)
        return result

    def check_silence(self, total_spikes: int) -> GateResult:
        result = self.silence_gate.check(total_spikes)
        self._results.append(result)
        return result

    def all_passed(self) -> bool:
        return all(r.passed for r in self._results)

    def get_results(self) -> List[GateResult]:
        return list(self._results)

    def reset(self) -> None:
        self.collapse_gate.reset()
        self._results.clear()
