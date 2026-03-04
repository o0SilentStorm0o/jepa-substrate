# LavaJEPA: Comparing JEPA-Style Predictive Objectives on Spiking and Classical Neural Networks

> **Paper:** _Spiking Neural Networks Are Not Universally Better for JEPA-Style Predictive Objectives: Evidence for Regime-Dependent Advantages_
>
> 📄 **Link to paper will be added upon publication on arXiv.**

## Overview

This repository contains the complete source code, experiment scripts, configuration, and results for **LavaJEPA** — a controlled comparison of JEPA-style (Joint Embedding Predictive Architecture) self-supervised objectives implemented on two computational substrates:

| Substrate | Encoder | Training | Inference |
|-----------|---------|----------|-----------|
| **ANN** | 1-layer GRU (H=128) | PyTorch (CPU) | PyTorch (CPU, no grad) |
| **SNN** | Recurrent LIF neurons (H=128) | PyTorch + surrogate gradients | PyTorch (CPU, no grad)† |

> †The original design planned Intel Lava CPU simulation for SNN inference. Lava process-graph equivalence was verified offline (acceptance tests T-E1–T-E3, 100% spike match), but all reported measurements use PyTorch for both substrates.

Both substrates share identical task contracts, latent dimensions, masking policies, loss functions, and evaluation protocols. The only variable is the computational substrate.

## Key Findings

Across **120 experiments** (10 seeds × 6 conditions × 2 substrates), we find that the SNN advantage is **regime-dependent**:

| Masking Policy | ANN Loss | SNN Loss | Cohen's _d_ | Winner |
|----------------|----------|----------|-------------|--------|
| Future-block | 1.801 ± 0.056 | 7.189 ± 1.303 | −4.17 | ANN ✓ |
| Random-drop | 1.841 ± 0.060 | 1.698 ± 0.129 | +1.03 | **SNN ✓** |
| Multi-target | 1.865 ± 0.067 | 5.654 ± 0.276 | −12.30 | ANN ✓ |

**Latency** (PyTorch CPU, single-thread, median of 1,000 timed passes):

| Masking Policy | ANN (ms) | SNN (ms) | Ratio |
|----------------|----------|----------|-------|
| Future-block | 2.82 ± 0.03 | 7.16 ± 0.04 | 2.5× |
| Random-drop | 2.81 ± 0.03 | 7.17 ± 0.07 | 2.6× |
| Multi-target | 2.82 ± 0.03 | 7.18 ± 0.09 | 2.5× |

**Ablation highlights** (future-block policy):
- **OnlineTeacher:** SNN loss drops from 7.19 → 0.38 (18.9× improvement), making SNN dramatically better than ANN (_d_ = +27.27)
- **NoPos:** Removing positional encoding reduces SNN loss from 7.19 → 2.65 (2.7× improvement)
- **NoMaskTok:** Negligible effect

All comparisons are statistically significant (paired _t_-test, Holm–Bonferroni corrected, FWER ≤ 0.05). Confirmed by Wilcoxon signed-rank tests.

## Dataset

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (Anguita et al., 2013):
- 6-channel IMU (accelerometer + gyroscope) at 50 Hz
- Window length: T = 128 samples (2.56 s)
- Subject-level split: 21 train / 4 val / 5 test subjects (7,185 / 1,527 / 1,587 windows)

## Repository Structure

```
jepa-substrate/
├── config/
│   ├── experiment.yaml       # Sole source of truth for all hyperparameters
│   └── loader.py             # YAML config loader
├── shared/                   # Shared modules (both substrates)
│   ├── data.py               # WindowDataset, data loading
│   ├── loss.py               # JEPA loss (MSE in latent space)
│   ├── masking.py            # 3 masking policies (future_block, random_drop, multi_target)
│   ├── positional.py         # Sinusoidal positional encoding
│   ├── timing.py             # Latency measurement wrapper
│   ├── gates.py              # Failure-mode gates (collapse, saturation, silence)
│   ├── harness.py            # Evaluation harness
│   └── env_log.py            # Environment logging
├── ann/                      # ANN (GRU) implementation
│   ├── model.py              # GRU encoder + MLP predictor + EMA teacher
│   ├── train.py              # ANN training loop
│   └── evaluate.py           # ANN evaluation
├── snn/                      # SNN (LIF) implementation
│   ├── model.py              # LIF encoder + spiking predictor + readout
│   ├── train.py              # SNN training (surrogate gradient BPTT)
│   ├── evaluate.py           # SNN test-set evaluation (PyTorch)
│   ├── lava_export.py        # Weight export PyTorch → Lava
│   └── lava_inference.py     # Lava CPU inference (offline verification & latency benchmark)
├── analysis/
│   ├── stats.py              # Full statistical pipeline (S1–S6 from paper)
│   └── plots.py              # Plotting utilities
├── scripts/
│   ├── run_experiment.py     # Single experiment runner
│   ├── run_primary.py        # Run all 60 primary experiments
│   ├── run_ablation.py       # Run all 60 ablation experiments
│   ├── run_analysis.py       # Statistical analysis + plots
│   ├── generate_plots.py     # Publication figure generation
│   ├── benchmark_time_budget.py  # Micro-benchmarks for time budget
│   └── benchmark_lava.py     # Lava-specific benchmarks
├── tests/                    # 145 acceptance tests
│   ├── test_invariants.py    # T-I1..T-I4 (same mask, loss, regime, latency)
│   ├── test_export.py        # T-E1..T-E3 (spike/trace/readout round-trip)
│   ├── test_failure_gates.py # T-F1..T-F3 (collapse, saturation, silence)
│   └── test_lava_verification.py  # Lava process-graph equivalence tests
├── data/
│   └── download.py           # Dataset download + SHA-256 verification
├── results/
│   ├── uci_har/
│   │   └── uci_har_s{1..10}_{policy}_{substrate}_{condition}/
│   │       ├── results.csv       # Per-window metrics (loss, latency, spikes, ...)
│   │       ├── run_spec.json     # Run specification
│   │       ├── environment.json  # Hardware/software environment
│   │       └── training_history.json
│   └── analysis/
│       ├── statistical_results.csv   # All statistical tests
│       ├── table1.tex                # LaTeX results table
│       ├── fig2_loss_comparison.pdf  # Loss violin plots
│       ├── fig3_latency_cdf.pdf      # Latency CDF
│       ├── fig4_delta_loss.pdf       # Paired difference distributions
│       └── fig6_ablation.pdf         # Ablation bar chart
├── requirements.txt
└── README.md
```

## Reproducing the Experiments

### Prerequisites

- Python 3.10 (required by lava-nc)
- macOS or Linux (Lava CPU simulation)
- ~30 GB disk space for full results (observables + checkpoints)

### Setup

```bash
git clone https://github.com/o0SilentStorm0o/jepa-substrate.git
cd jepa-substrate
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download Dataset

```bash
python -m data.download
```

### Run Tests (145 acceptance tests)

```bash
pytest tests/ -v
```

### Run Experiments

```bash
# Primary experiments: 10 seeds × 3 policies × 2 substrates = 60 runs
# Estimated time: ~3 hours on Apple M2 Max (single-thread, PyTorch)
python scripts/run_primary.py

# Ablation experiments: 10 seeds × 3 conditions × 2 substrates = 60 runs
# Estimated time: ~3 hours
python scripts/run_ablation.py

# Statistical analysis + publication figures
python scripts/run_analysis.py
python scripts/generate_plots.py
```

### Configuration

All hyperparameters are in `config/experiment.yaml`. Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `H` | 128 | Latent dimension |
| `T` | 128 | Window length (samples) |
| `N_train` | 5,000 | Training steps |
| `tau_ema` | 0.996 | EMA teacher momentum |
| `v_th` | 1.0 | LIF firing threshold |
| `tau_mem` | 10 | Membrane time constant |
| `surrogate_k` | 25 | Surrogate gradient steepness |
| Seeds | 1–10 | 10 independent seeds |

## Statistical Protocol

Pre-registered analysis plan (implemented in `analysis/stats.py`):

1. **Paired _t_-test** on Δℒ = ℒ_SNN − ℒ_ANN for each condition
2. **Wilcoxon signed-rank** as non-parametric robustness check
3. **Holm–Bonferroni** correction across all 6 loss tests (FWER ≤ 0.05)
4. **Cohen's _d_** with 95% bootstrap CI (10,000 resamples)
5. Latency tests corrected separately (3 tests)

## Results

### Pre-computed Results

The `results/` directory contains per-window metrics for all 120 runs.
Each `results.csv` has columns:

```
window_index, loss, forward_ms, teacher_ms, overhead_ms, total_ms,
embedding_norm_mean, embedding_norm_std, teacher_norm_mean, teacher_norm_std,
prediction_variance, total_spikes, spike_rate, synaptic_events, energy_proxy
```

Aggregated analysis outputs are in `results/analysis/`.

### Figures

| Figure | Description | File |
|--------|-------------|------|
| Fig. 2 | Loss comparison (violin) | `results/analysis/fig2_loss_comparison.pdf` |
| Fig. 3 | Latency CDF | `results/analysis/fig3_latency_cdf.pdf` |
| Fig. 4 | Paired Δℒ distributions | `results/analysis/fig4_delta_loss.pdf` |
| Fig. 6 | Ablation bar chart | `results/analysis/fig6_ablation.pdf` |

## Hardware

All experiments were conducted on:
- Apple MacBook Pro (M2 Max), 12 cores (8P + 4E), 32 GB unified memory
- macOS Sequoia 15.3.1, single-thread CPU execution
- PyTorch 2.6.0 CPU (both substrates)
- Intel Lava 0.13.0 (offline equivalence verification only)
- Total experiment time: 5.81 h (22 Feb 2026, 12:08–18:47)

## Citation

```bibtex
@article{strnadel2026lavajepa,
  title   = {Spiking Neural Networks Are Not Universally Better for
             JEPA-Style Predictive Objectives: Evidence for
             Regime-Dependent Advantages},
  author  = {Strnadel, David},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {To appear}
}
```

## License

MIT
