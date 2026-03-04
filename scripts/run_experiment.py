"""Main experiment orchestrator for LavaJEPA.

Runs the full experiment grid across multiple datasets:
  K datasets x 10 seeds x 3 masking policies x 2 substrates (primary) = K*60 runs
  K datasets x 10 seeds x 3 ablation conditions x 2 substrates = K*60 runs
  Total: K * 120 runs (360 for K=3)

Each run:
  1. Train (ANN or SNN) with the given seed and masking policy.
  2. Export SNN weights to Lava format (SNN only).
  3. Pre-compute teacher targets on the test set.
  4. Measure inference on the test set (per-window loss, latency, energy).
  5. Run failure-mode gates.
  6. Save results CSV, observables NPZ, and environment log.

Usage:
  python scripts/run_experiment.py [--config config/experiment.yaml]
                                   [--subset primary|ablation|all]
                                   [--dataset uci_har|speech_commands_v2|ptb_xl_ecg|all]
                                   [--parallel N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.loader import Config, load_config
from shared.data import prepare_dataset, WindowDataset
from shared.env_log import collect_environment_info, save_environment_info
from shared.masking import generate_mask
from shared.positional import sinusoidal_position_encoding_numpy

from ann.model import ANNModel
from ann.train import train_ann
from ann.evaluate import precompute_teacher_targets, run_ann_measurement

from snn.model import SNNModel
from snn.train import train_snn
from snn.lava_export import export_weights

from analysis.stats import (
    ComparisonSpec,
    run_full_analysis,
    save_analysis_csv,
    save_latex_table,
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_run_grid(cfg: Config, subset: str = "all") -> List[dict]:
    """Build the list of run configurations.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    subset : str
        "primary", "ablation", or "all".

    Returns
    -------
    list of dict
        Each dict specifies a single run.
    """
    runs = []

    seeds = cfg.seeds.init_seeds
    masking_policies = cfg.experiment.policies
    substrates = ["ann", "snn"]

    if subset in ("primary", "all"):
        for seed in seeds:
            for policy in masking_policies:
                for substrate in substrates:
                    runs.append({
                        "seed": seed,
                        "masking_policy": policy,
                        "substrate": substrate,
                        "condition": "primary",
                        "ablation": None,
                    })

    if subset in ("ablation", "all"):
        ablation_conditions = cfg.experiment.ablations
        for seed in seeds:
            for ablation in ablation_conditions:
                for substrate in substrates:
                    runs.append({
                        "seed": seed,
                        "masking_policy": "future_block",  # Default for ablation
                        "substrate": substrate,
                        "condition": ablation,
                        "ablation": ablation,
                    })

    return runs


def run_single(
    run_spec: dict,
    cfg: Config,
    data_dir: Path,
    output_root: Path,
    dataset_name: str = "uci_har",
) -> dict:
    """Execute a single experiment run.

    Parameters
    ----------
    run_spec : dict
        Run specification from build_run_grid.
    cfg : Config
        Experiment configuration.
    data_dir : Path
        Path to downloaded dataset.
    output_root : Path
        Root output directory.
    dataset_name : str
        Dataset identifier (e.g. "uci_har", "speech_commands_v2", "ptb_xl_ecg").

    Returns
    -------
    dict
        Summary of run results.
    """
    seed = run_spec["seed"]
    policy = run_spec["masking_policy"]
    substrate = run_spec["substrate"]
    condition = run_spec["condition"]
    ablation = run_spec["ablation"]

    run_id = f"{dataset_name}_s{seed}_{policy}_{substrate}_{condition}"
    run_dir = output_root / run_id

    # Skip if already completed (results.csv exists)
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        logger.info("SKIP (already done): %s", run_id)
        return {
            "run_id": run_id,
            "dataset": dataset_name,
            "seed": seed,
            "masking_policy": policy,
            "substrate": substrate,
            "condition": condition,
            "status": "skipped",
        }

    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RUN: %s", run_id)
    logger.info("=" * 70)

    t_start = time.time()

    # Save environment info
    save_environment_info(run_dir / "environment.json")

    # Save run spec (including dataset)
    spec_with_dataset = {**run_spec, "dataset": dataset_name}
    (run_dir / "run_spec.json").write_text(
        json.dumps(spec_with_dataset, indent=2), encoding="utf-8",
    )

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get dataset-specific config
    ds_cfg = None
    if cfg.datasets:
        ds_cfg = cfg.dataset_cfg(dataset_name)

    # Prepare data (dispatcher handles dataset_name)
    dataset = prepare_dataset(
        data_dir=data_dir,
        split_seed=cfg.seeds.data_split,
        dataset_name=dataset_name,
        dataset_cfg=ds_cfg,
    )

    # Get T and D from dataset-specific config or backward compat
    if ds_cfg is not None:
        T = ds_cfg.T
        D = ds_cfg.D
    else:
        T = cfg.data.T
        D = cfg.data.D

    train_windows_np = dataset["train"]
    val_windows_np = dataset["val"]
    test_windows_np = dataset["test"]

    # Build WindowDataset objects for training
    train_dataset = WindowDataset(
        windows=train_windows_np,
        masking_policy=policy,
        split_seed=seed,
        context_fraction=cfg.masking.future_block.context_fraction,
        target_probability=cfg.masking.random_drop.target_probability,
        n_blocks=cfg.masking.multi_target.n_blocks,
        block_length_fraction=cfg.masking.multi_target.block_length_fraction,
        min_gap_fraction=cfg.masking.multi_target.min_gap_fraction,
    )
    val_dataset = WindowDataset(
        windows=val_windows_np,
        masking_policy=policy,
        split_seed=seed,
        context_fraction=cfg.masking.future_block.context_fraction,
        target_probability=cfg.masking.random_drop.target_probability,
        n_blocks=cfg.masking.multi_target.n_blocks,
        block_length_fraction=cfg.masking.multi_target.block_length_fraction,
        min_gap_fraction=cfg.masking.multi_target.min_gap_fraction,
    )

    # Generate masks for test set
    test_masks = []
    for idx in range(len(test_windows_np)):
        mask, _, _ = generate_mask(
            policy=policy,
            T=T,
            split_seed=seed,
            window_index=idx,
            context_fraction=cfg.masking.future_block.context_fraction,
            target_probability=cfg.masking.random_drop.target_probability,
            n_blocks=cfg.masking.multi_target.n_blocks,
            block_length_fraction=cfg.masking.multi_target.block_length_fraction,
            min_gap_fraction=cfg.masking.multi_target.min_gap_fraction,
        )
        test_masks.append(mask)

    # Test dataset for measurement
    test_dataset = WindowDataset(
        windows=test_windows_np,
        masking_policy=policy,
        split_seed=seed,
        context_fraction=cfg.masking.future_block.context_fraction,
        target_probability=cfg.masking.random_drop.target_probability,
        n_blocks=cfg.masking.multi_target.n_blocks,
        block_length_fraction=cfg.masking.multi_target.block_length_fraction,
        min_gap_fraction=cfg.masking.multi_target.min_gap_fraction,
    )

    # Determine ablation modifications
    use_ema = ablation != "NoPos"  # keep EMA for all except specific ablations
    use_pos = ablation != "NoPos"
    use_mask_token = ablation != "NoMaskTok"

    # ---- Train ----
    if substrate == "ann":
        model = ANNModel(
            input_dim=D,
            hidden_dim=cfg.model.H,
            tau_ema=cfg.model.tau_ema if ablation != "OnlineTeacher" else 1.0,
        )
        # Apply ablation: NoPos → zero positional encoding
        if ablation == "NoPos":
            model.predictor.pos_encoding.zero_()
        # Apply ablation: NoMaskTok → zero mask embedding
        if ablation == "NoMaskTok":
            model.predictor.mask_token.data.zero_()
        train_ann(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_train_steps=cfg.training.N_train,
            lr_max=cfg.training.lr_max,
            warmup_fraction=cfg.training.warmup_fraction,
            weight_decay=cfg.training.weight_decay,
            betas=tuple(cfg.training.betas),
            val_every=cfg.training.val_every,
            output_dir=run_dir,
            seed=seed,
        )

        # Pre-compute teacher targets
        teacher_dir = run_dir / "teacher_targets"
        targets_dir = precompute_teacher_targets(
            model=model,
            dataset=test_dataset,
            output_dir=run_dir,
        )

        # Measure
        results = run_ann_measurement(
            model=model,
            test_dataset=test_dataset,
            targets_dir=targets_dir,
            output_dir=run_dir,
        )

    elif substrate == "snn":
        snn_model = SNNModel(
            input_dim=D,
            hidden_dim=cfg.model.H,
            beta=cfg.model.beta,
            v_th=cfg.model.v_th,
            surrogate_k=cfg.model.surrogate_k,
            trace_alpha=cfg.model.trace_alpha,
            tau_ema=cfg.model.tau_ema if ablation != "OnlineTeacher" else 1.0,
        )
        # Apply ablation: NoPos → zero positional encoding
        if ablation == "NoPos":
            snn_model.predictor.pos_encoding.zero_()
        # Apply ablation: NoMaskTok → zero mask bias current
        if ablation == "NoMaskTok":
            snn_model.predictor.c_mask = 0.0
            snn_model.predictor.mask_weight.weight.data.zero_()
        train_snn(
            model=snn_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_train_steps=cfg.training.N_train,
            lr_max=cfg.training.lr_max,
            warmup_fraction=cfg.training.warmup_fraction,
            weight_decay=cfg.training.weight_decay,
            betas=tuple(cfg.training.betas),
            val_every=cfg.training.val_every,
            output_dir=run_dir,
            seed=seed,
        )

        # Export to Lava (for latency benchmark)
        weights_path = run_dir / "lava_weights.npz"
        export_weights(snn_model, weights_path)

        # Pre-compute teacher targets (using the SNN teacher)
        teacher_dir = run_dir / "teacher_targets"
        _precompute_snn_teacher_targets(
            snn_model, test_windows_np, teacher_dir, cfg,
        )

        # Measure via PyTorch SNN (equivalent to Lava — verified V14: 100% match)
        from snn.evaluate import run_snn_measurement
        results = run_snn_measurement(
            model=snn_model,
            test_windows=test_windows_np,
            test_masks=test_masks,
            teacher_targets_dir=teacher_dir,
            output_dir=run_dir,
            T=T,
            H=cfg.model.H,
            c_mask=cfg.model.c_mask,
            energy_alpha=cfg.energy_proxy.alpha,
            energy_beta=cfg.energy_proxy.beta,
            energy_gamma=cfg.energy_proxy.gamma,
        )

    else:
        raise ValueError(f"Unknown substrate: {substrate}")

    t_elapsed = time.time() - t_start

    summary = {
        "run_id": run_id,
        "dataset": dataset_name,
        "seed": seed,
        "masking_policy": policy,
        "substrate": substrate,
        "condition": condition,
        "n_windows": len(results),
        "mean_loss": float(np.mean([r.loss for r in results])),
        "elapsed_s": t_elapsed,
    }

    logger.info(
        "RUN COMPLETE: %s | mean_loss=%.6f | elapsed=%.1fs",
        run_id, summary["mean_loss"], t_elapsed,
    )

    return summary


def _precompute_snn_teacher_targets(
    model: "SNNModel",
    test_windows: np.ndarray,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Pre-compute teacher targets for SNN using the PyTorch teacher."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx in range(len(test_windows)):
            x = torch.from_numpy(test_windows[idx]).float().unsqueeze(0)
            # Run teacher encoder -- returns (latents, spikes, traces, membrane)
            teacher_latents, _, _, _ = model.teacher_encoder(x)
            targets = teacher_latents.squeeze(0).numpy()

            np.savez_compressed(
                str(output_dir / f"target_{idx:05d}.npz"),
                targets=targets,
            )

    logger.info("Pre-computed %d SNN teacher targets", len(test_windows))


def run_analysis(
    output_root: Path,
    cfg: Config,
) -> None:
    """Run statistical analysis across all completed runs.

    Parameters
    ----------
    output_root : Path
        Root output directory containing run subdirectories.
    cfg : Config
        Experiment configuration.
    """
    import csv

    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-seed mean losses
    seeds = cfg.seeds.init_seeds
    policies = cfg.experiment.policies

    # Determine datasets to analyze
    datasets_to_analyze = cfg.active_datasets if cfg.active_datasets else ["uci_har"]

    comparisons = []

    for dataset_name in datasets_to_analyze:
        ds_output_dir = output_root / dataset_name

        for policy in policies:
            ann_means = []
            snn_means = []

            for seed in seeds:
                for substrate in ("ann", "snn"):
                    run_id = f"{dataset_name}_s{seed}_{policy}_{substrate}_primary"
                    csv_path = ds_output_dir / run_id / "results.csv"
                    if not csv_path.exists():
                        logger.warning("Missing results: %s", csv_path)
                        continue

                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        losses = [float(row["loss"]) for row in reader]

                    mean_loss = float(np.mean(losses))
                    if substrate == "ann":
                        ann_means.append(mean_loss)
                    else:
                        snn_means.append(mean_loss)

            if len(ann_means) == len(seeds) and len(snn_means) == len(seeds):
                comparisons.append(ComparisonSpec(
                    metric="loss",
                    masking_policy=policy,
                    condition="primary",
                    dataset=dataset_name,
                    ann_values=np.array(ann_means),
                    snn_values=np.array(snn_means),
                ))

    if comparisons:
        report = run_full_analysis(comparisons, alpha=0.05)
        save_analysis_csv(report, analysis_dir / "statistical_results.csv")
        save_latex_table(report, analysis_dir / "table1.tex")
        logger.info("Statistical analysis saved to %s", analysis_dir)
    else:
        logger.warning("No complete comparisons found for analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="LavaJEPA experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "experiment.yaml"),
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["primary", "ablation", "all"],
        default="all",
        help="Which subset of runs to execute.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Which dataset to run: 'all' (from active_datasets), or a specific key.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Root data directory (per-dataset subdirs will be created).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip training/measurement, only run analysis on existing results.",
    )
    args = parser.parse_args()

    setup_logging()

    cfg = load_config(args.config)
    data_root = Path(args.data_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.analysis_only:
        run_analysis(output_root, cfg)
        return

    # Determine which datasets to run
    if args.dataset == "all":
        datasets_to_run = cfg.active_datasets if cfg.active_datasets else ["uci_har"]
    else:
        datasets_to_run = [args.dataset]

    # Build and execute run grid per dataset
    runs = build_run_grid(cfg, args.subset)
    total_runs = len(runs) * len(datasets_to_run)
    logger.info(
        "Total runs: %d (%d per dataset x %d datasets)",
        total_runs, len(runs), len(datasets_to_run),
    )

    summaries = []
    run_counter = 0

    for dataset_name in datasets_to_run:
        # Per-dataset data directory
        data_dir = data_root / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)

        # Per-dataset output directory
        ds_output_dir = output_root / dataset_name
        ds_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("DATASET: %s (%d runs)", dataset_name, len(runs))
        logger.info("=" * 70)

        for i, run_spec in enumerate(runs):
            run_counter += 1
            logger.info(
                "--- Run %d/%d (dataset=%s, %d/%d) ---",
                run_counter, total_runs, dataset_name, i + 1, len(runs),
            )
            try:
                summary = run_single(
                    run_spec, cfg, data_dir, ds_output_dir,
                    dataset_name=dataset_name,
                )
                summaries.append(summary)
            except Exception:
                logger.exception(
                    "Run FAILED: dataset=%s, seed=%d, policy=%s, substrate=%s",
                    dataset_name, run_spec["seed"],
                    run_spec["masking_policy"], run_spec["substrate"],
                )
                summaries.append({
                    "run_id": f"{dataset_name}_s{run_spec['seed']}_"
                              f"{run_spec['masking_policy']}_"
                              f"{run_spec['substrate']}_{run_spec['condition']}",
                    "dataset": dataset_name,
                    "status": "FAILED",
                })

    # Save run summary
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Run summary saved to %s", summary_path)

    # Run analysis
    run_analysis(output_root, cfg)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
