"""UCI HAR dataset download and verification.

Implements D1 from the specification:
  The dataset is obtained via the official UCI ML Repository URL and
  verified by SHA-256 hash of the archive.

The dataset contains 3-axis accelerometer and 3-axis gyroscope signals
(D=6) sampled at 50 Hz from 30 subjects performing 6 activities.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Official UCI HAR Dataset URL
UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)

# Expected directory structure after extraction
_RAW_DIR = "UCI HAR Dataset"
_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_uci_har(
    data_dir: Path | str,
    url: str = UCI_HAR_URL,
    force: bool = False,
) -> Path:
    """Download and extract the UCI HAR dataset.

    Parameters
    ----------
    data_dir : Path or str
        Directory where the dataset will be stored.
    url : str
        Download URL.
    force : bool
        If True, re-download even if already present.

    Returns
    -------
    Path
        Path to the extracted dataset root directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "UCI_HAR_Dataset.zip"
    extract_dir = data_dir / _RAW_DIR

    if extract_dir.exists() and not force:
        logger.info("Dataset already extracted at %s", extract_dir)
        return extract_dir

    if not zip_path.exists() or force:
        logger.info("Downloading UCI HAR dataset from %s", url)
        urllib.request.urlretrieve(url, str(zip_path))
        logger.info("Download complete: %s", zip_path)

    # Log SHA-256 hash for data versioning (R4)
    file_hash = _sha256_file(zip_path)
    logger.info("SHA-256 of archive: %s", file_hash)

    # Save hash to file for reproducibility
    hash_file = data_dir / "archive_sha256.txt"
    with open(hash_file, "w", encoding="utf-8") as fh:
        fh.write(file_hash + "\n")

    logger.info("Extracting archive to %s", data_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(data_dir))

    if not extract_dir.exists():
        raise FileNotFoundError(
            f"Expected directory {extract_dir} after extraction"
        )

    logger.info("Extraction complete: %s", extract_dir)
    return extract_dir


def _load_inertial_signals(
    dataset_dir: Path,
    split: str,
) -> np.ndarray:
    """Load raw inertial signal data for a given split.

    Parameters
    ----------
    dataset_dir : Path
        Root directory of the extracted UCI HAR dataset.
    split : str
        One of ``"train"`` or ``"test"``.

    Returns
    -------
    np.ndarray
        Signal data of shape ``(N_windows, 128, 6)``, where 6 channels are
        body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y,
        body_gyro_z.
    """
    signal_dir = dataset_dir / split / "Inertial Signals"
    if not signal_dir.exists():
        raise FileNotFoundError(f"Inertial signals directory not found: {signal_dir}")

    channels = []
    for signal_name in _SIGNALS:
        filename = f"{signal_name}_{split}.txt"
        filepath = signal_dir / filename
        data = np.loadtxt(str(filepath), dtype=np.float32)
        channels.append(data)

    # Stack channels: each is (N_windows, 128) -> (N_windows, 128, 6)
    signals = np.stack(channels, axis=-1)
    return signals


def _load_subject_ids(dataset_dir: Path, split: str) -> np.ndarray:
    """Load subject IDs for each window in a given split.

    Parameters
    ----------
    dataset_dir : Path
        Root directory of the extracted UCI HAR dataset.
    split : str
        One of ``"train"`` or ``"test"``.

    Returns
    -------
    np.ndarray
        Subject IDs of shape ``(N_windows,)``.
    """
    filepath = dataset_dir / split / f"subject_{split}.txt"
    return np.loadtxt(str(filepath), dtype=np.int32)


def _load_activity_labels(dataset_dir: Path, split: str) -> np.ndarray:
    """Load activity labels for each window.

    Parameters
    ----------
    dataset_dir : Path
        Root directory of the extracted UCI HAR dataset.
    split : str
        One of ``"train"`` or ``"test"``.

    Returns
    -------
    np.ndarray
        Activity labels of shape ``(N_windows,)``.
    """
    filepath = dataset_dir / split / f"y_{split}.txt"
    return np.loadtxt(str(filepath), dtype=np.int32)


def load_raw_data(
    dataset_dir: Path | str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all raw data from the UCI HAR dataset.

    Returns
    -------
    dict
        Dictionary with keys ``"train"`` and ``"test"``, each containing:
        - ``"signals"``: shape ``(N, 128, 6)``
        - ``"subjects"``: shape ``(N,)``
        - ``"labels"``: shape ``(N,)``
    """
    dataset_dir = Path(dataset_dir)
    result = {}
    for split in ["train", "test"]:
        result[split] = {
            "signals": _load_inertial_signals(dataset_dir, split),
            "subjects": _load_subject_ids(dataset_dir, split),
            "labels": _load_activity_labels(dataset_dir, split),
        }
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"
    extract_dir = download_uci_har(data_dir)
    raw_data = load_raw_data(extract_dir)
    for split_name, split_data in raw_data.items():
        n = split_data["signals"].shape[0]
        subjects = np.unique(split_data["subjects"])
        logger.info(
            "%s: %d windows, %d subjects (%s)",
            split_name, n, len(subjects), subjects,
        )
