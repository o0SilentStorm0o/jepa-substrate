"""Run all 30 primary SNN experiments (10 seeds x 3 masking policies)."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["OMP_NUM_THREADS"] = "1"

from config.loader import load_config
from scripts.run_experiment import run_single, build_run_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "config" / "experiment.yaml"))
    data_dir = PROJECT_ROOT / "data" / "uci_har"
    output_root = PROJECT_ROOT / "results"
    output_root.mkdir(parents=True, exist_ok=True)

    runs = build_run_grid(cfg, "primary")
    snn_runs = [r for r in runs if r["substrate"] == "snn"]
    logger.info("Total SNN primary runs: %d", len(snn_runs))

    summaries = []
    for i, run_spec in enumerate(snn_runs):
        logger.info(
            "--- Run %d/%d: seed=%d, policy=%s ---",
            i + 1, len(snn_runs), run_spec["seed"], run_spec["masking_policy"],
        )
        try:
            summary = run_single(run_spec, cfg, data_dir, output_root)
            summaries.append(summary)
        except Exception:
            logger.exception("Run FAILED")
            summaries.append({"run_id": "FAILED", "seed": run_spec["seed"]})

    out_path = output_root / "snn_primary_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    n_ok = sum(1 for s in summaries if "mean_loss" in s)
    logger.info("ALL SNN PRIMARY RUNS COMPLETE: %d/%d successful", n_ok, len(snn_runs))


if __name__ == "__main__":
    main()
