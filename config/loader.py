"""Configuration loader for LavaJEPA experiment.

Loads the single source-of-truth YAML configuration and provides
typed access to all experiment parameters.

Supports both single-dataset (legacy ``data:`` key) and multi-dataset
(``datasets:`` mapping + ``active_datasets:`` list) configurations.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "experiment.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset configuration (replaces the old DataConfig for each dataset)."""
    key: str                    # e.g. "uci_har", "speech_commands_v2"
    name: str
    description: str
    url: str
    domain: str
    T: int
    D: int
    normalization: str
    split_by: str               # "subject_id", "speaker_id", "patient_id"
    preprocessing: str          # "none", "log_mel_spectrogram", "downsample_and_select_window"
    N_train_windows: int
    N_val_windows: int
    N_test_windows: int
    # Optional fields (dataset-specific)
    sha256: Optional[str] = None
    reference: Optional[str] = None
    sampling_rate_hz: Optional[int] = None
    original_sampling_rate_hz: Optional[int] = None
    target_sampling_rate_hz: Optional[int] = None
    window_length_sec: Optional[float] = None
    record_length_sec: Optional[int] = None
    n_subjects: Optional[int] = None
    n_train_subjects: Optional[int] = None
    n_val_subjects: Optional[int] = None
    n_test_subjects: Optional[int] = None
    n_classes: Optional[int] = None
    total_records: Optional[int] = None
    # Mel spectrogram parameters (SpeechCommands)
    mel_n_fft: Optional[int] = None
    mel_hop_length: Optional[int] = None
    # Downsampling parameters (PTB-XL)
    downsample_factor: Optional[int] = None
    window_index: Optional[int] = None


@dataclass(frozen=True)
class DataConfig:
    """Backward-compatible data config pointing to the *current* dataset.

    Allows existing code using ``cfg.data.T`` / ``cfg.data.D`` to keep working.
    """
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
    K: int
    S: int
    P: int
    A: int
    policies: List[str]
    ablations: List[str]
    runs_per_dataset: int
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
    data: DataConfig              # backward compat: points to first active dataset
    model: ModelConfig
    training: TrainingConfig
    masking: MaskingConfig
    measurement: MeasurementConfig
    seeds: SeedConfig
    experiment: ExperimentConfig
    energy_proxy: EnergyProxyConfig
    failure_gates: FailureGatesConfig
    tolerances: TolerancesConfig
    # Multi-dataset support
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    active_datasets: List[str] = field(default_factory=list)
    _raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def sha256(self) -> str:
        """SHA-256 hash of the raw YAML configuration."""
        raw_bytes = yaml.dump(self._raw, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def dataset_cfg(self, dataset_key: str) -> DatasetConfig:
        """Get configuration for a specific dataset by key."""
        return self.datasets[dataset_key]


def _parse_dataset_config(key: str, d: Dict[str, Any]) -> DatasetConfig:
    """Parse a single dataset entry from the YAML datasets mapping."""
    return DatasetConfig(
        key=key,
        name=d.get("name", key),
        description=d.get("description", ""),
        url=d.get("url", ""),
        domain=d.get("domain", "unknown"),
        T=d["T"],
        D=d["D"],
        normalization=d.get("normalization", "zscore"),
        split_by=d.get("split_by", "subject_id"),
        preprocessing=d.get("preprocessing", "none"),
        N_train_windows=d.get("N_train_windows", 0),
        N_val_windows=d.get("N_val_windows", 0),
        N_test_windows=d.get("N_test_windows", 0),
        sha256=d.get("sha256"),
        reference=d.get("reference"),
        sampling_rate_hz=d.get("sampling_rate_hz"),
        original_sampling_rate_hz=d.get("original_sampling_rate_hz"),
        target_sampling_rate_hz=d.get("target_sampling_rate_hz"),
        window_length_sec=d.get("window_length_sec"),
        record_length_sec=d.get("record_length_sec"),
        n_subjects=d.get("n_subjects"),
        n_train_subjects=d.get("n_train_subjects"),
        n_val_subjects=d.get("n_val_subjects"),
        n_test_subjects=d.get("n_test_subjects"),
        n_classes=d.get("n_classes"),
        total_records=d.get("total_records"),
        mel_n_fft=d.get("mel_n_fft"),
        mel_hop_length=d.get("mel_hop_length"),
        downsample_factor=d.get("downsample_factor"),
        window_index=d.get("window_index"),
    )


def _make_compat_data_config(ds: DatasetConfig) -> DataConfig:
    """Build a backward-compatible DataConfig from a DatasetConfig."""
    return DataConfig(
        dataset=ds.name,
        url=ds.url,
        sha256=ds.sha256 or "",
        sampling_rate_hz=ds.sampling_rate_hz or ds.target_sampling_rate_hz or 0,
        window_length_sec=ds.window_length_sec or 0.0,
        T=ds.T,
        D=ds.D,
        normalization=ds.normalization,
        split_ratios={},
        n_subjects=ds.n_subjects or 0,
        n_train_subjects=ds.n_train_subjects or 0,
        n_val_subjects=ds.n_val_subjects or 0,
        n_test_subjects=ds.n_test_subjects or 0,
    )


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

    # ---- Multi-dataset or legacy single-dataset ----
    datasets_dict: Dict[str, DatasetConfig] = {}
    active_list: List[str] = []

    if "datasets" in raw:
        # New multi-dataset format
        for ds_key, ds_raw in raw["datasets"].items():
            datasets_dict[ds_key] = _parse_dataset_config(ds_key, ds_raw)
        active_list = raw.get("active_datasets", list(datasets_dict.keys()))
        # Backward compat: first active dataset becomes cfg.data
        first_ds = datasets_dict[active_list[0]]
        data_cfg = _make_compat_data_config(first_ds)
    else:
        # Legacy single-dataset format
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

    # ExperimentConfig: support new fields K, runs_per_dataset
    exp_raw = raw["experiment"]
    experiment_cfg = ExperimentConfig(
        K=exp_raw.get("K", 1),
        S=exp_raw["S"],
        P=exp_raw["P"],
        A=exp_raw["A"],
        policies=exp_raw["policies"],
        ablations=exp_raw["ablations"],
        runs_per_dataset=exp_raw.get("runs_per_dataset", exp_raw["total_runs"]),
        total_runs=exp_raw["total_runs"],
    )

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
        datasets=datasets_dict,
        active_datasets=active_list,
        _raw=raw,
    )


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
