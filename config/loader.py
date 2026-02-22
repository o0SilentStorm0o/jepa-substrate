"""Configuration loader for LavaJEPA experiment.

Loads the single source-of-truth YAML configuration and provides
typed access to all experiment parameters.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "experiment.yaml"


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    url: str
    sha256: str
    sampling_rate_hz: int
    window_length_sec: float
    T: int
    D: int
    normalization: str
    split_ratios: Dict[str, float]
    n_subjects: int
    n_train_subjects: int
    n_val_subjects: int
    n_test_subjects: int


@dataclass(frozen=True)
class ModelConfig:
    H: int
    encoder_type: str
    predictor_layers: int
    predictor_hidden: int
    tau_ema: float
    v_th: float
    tau_mem: int
    beta: float
    surrogate_k: float
    trace_alpha: float
    c_mask: float


@dataclass(frozen=True)
class TrainingConfig:
    N_train: int
    lr_max: float
    warmup_fraction: float
    optimizer: str
    betas: List[float]
    weight_decay: float
    loss: str
    val_every: int
    masking_policy: str


@dataclass(frozen=True)
class FutureBlockConfig:
    context_fraction: float


@dataclass(frozen=True)
class RandomDropConfig:
    target_probability: float


@dataclass(frozen=True)
class MultiTargetConfig:
    n_blocks: int
    block_length_fraction: float
    min_gap_fraction: float


@dataclass(frozen=True)
class MaskingConfig:
    future_block: FutureBlockConfig
    random_drop: RandomDropConfig
    multi_target: MultiTargetConfig


@dataclass(frozen=True)
class MeasurementConfig:
    batch_size: int
    warmup_calls: int
    timed_repetitions: int
    thread_count: int
    lava_sim_cfg: str
    lava_select_tag: str


@dataclass(frozen=True)
class SeedConfig:
    data_split: int
    init_seeds: List[int]


@dataclass(frozen=True)
class ExperimentConfig:
    S: int
    P: int
    A: int
    policies: List[str]
    ablations: List[str]
    total_runs: int


@dataclass(frozen=True)
class EnergyProxyConfig:
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class FailureGatesConfig:
    collapse_variance_threshold: float
    collapse_consecutive_epochs: int
    saturation_spike_rate: float
    silence_spike_count: int


@dataclass(frozen=True)
class TolerancesConfig:
    spike_mismatch_per_1000: int
    trace_max_abs_diff: float
    readout_max_abs_diff: float
    loss_relative_diff: float


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    masking: MaskingConfig
    measurement: MeasurementConfig
    seeds: SeedConfig
    experiment: ExperimentConfig
    energy_proxy: EnergyProxyConfig
    failure_gates: FailureGatesConfig
    tolerances: TolerancesConfig
    _raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def sha256(self) -> str:
        """SHA-256 hash of the raw YAML configuration."""
        raw_bytes = yaml.dump(self._raw, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()


def load_config(path: Path | str | None = None) -> Config:
    """Load experiment configuration from YAML file.

    Parameters
    ----------
    path : Path or str, optional
        Path to the YAML configuration file. Defaults to
        ``config/experiment.yaml`` in the project root.

    Returns
    -------
    Config
        Fully validated, immutable configuration object.
    """
    if path is None:
        path = _DEFAULT_CONFIG
    path = Path(path)

    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    data_cfg = DataConfig(**raw["data"])
    model_cfg = ModelConfig(**raw["model"])
    training_cfg = TrainingConfig(**raw["training"])

    masking_raw = raw["masking"]
    masking_cfg = MaskingConfig(
        future_block=FutureBlockConfig(**masking_raw["future_block"]),
        random_drop=RandomDropConfig(**masking_raw["random_drop"]),
        multi_target=MultiTargetConfig(**masking_raw["multi_target"]),
    )

    measurement_cfg = MeasurementConfig(**raw["measurement"])
    seed_cfg = SeedConfig(**raw["seeds"])
    experiment_cfg = ExperimentConfig(**raw["experiment"])
    energy_cfg = EnergyProxyConfig(**raw["energy_proxy"])
    failure_cfg = FailureGatesConfig(**raw["failure_gates"])
    tolerances_cfg = TolerancesConfig(**raw["tolerances"])

    return Config(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        masking=masking_cfg,
        measurement=measurement_cfg,
        seeds=seed_cfg,
        experiment=experiment_cfg,
        energy_proxy=energy_cfg,
        failure_gates=failure_cfg,
        tolerances=tolerances_cfg,
        _raw=raw,
    )


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
