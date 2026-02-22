"""Environment logging for reproducibility.

Implements R2 from the specification: every run records config hash,
git commit, Python version, PyTorch version, Lava version, OS, CPU model, RAM.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return "unknown"


def _get_git_dirty() -> bool:
    """Check if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.returncode != 0
    except FileNotFoundError:
        return False


def _get_cpu_model() -> str:
    """Get CPU model string."""
    machine = platform.machine()
    processor = platform.processor()
    if processor:
        return processor
    return machine


def _get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback for macOS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
        except FileNotFoundError:
            pass
    return -1.0


def _get_torch_version() -> str:
    """Get PyTorch version."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not installed"


def _get_lava_version() -> str:
    """Get Intel Lava version."""
    try:
        import lava
        return getattr(lava, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _get_numpy_version() -> str:
    """Get NumPy version."""
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return "not installed"


def collect_environment_info(config_sha256: str = "") -> Dict[str, Any]:
    """Collect full environment information for a run.

    Parameters
    ----------
    config_sha256 : str
        SHA-256 hash of the experiment configuration.

    Returns
    -------
    dict
        Environment information dictionary.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "git_commit": _get_git_commit(),
        "git_dirty": _get_git_dirty(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "pytorch_version": _get_torch_version(),
        "lava_version": _get_lava_version(),
        "numpy_version": _get_numpy_version(),
        "os": platform.platform(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": _get_cpu_model(),
        "ram_gb": round(_get_ram_gb(), 2),
        "hostname": platform.node(),
    }


def save_environment_info(
    output_path: Path | str,
    config_sha256: str = "",
) -> Dict[str, Any]:
    """Collect and save environment info to a JSON file.

    Parameters
    ----------
    output_path : Path or str
        Output file path (typically ``env.json``).
    config_sha256 : str
        SHA-256 hash of the experiment configuration.

    Returns
    -------
    dict
        The saved environment information.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = collect_environment_info(config_sha256)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)

    return info
