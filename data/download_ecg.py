"""PTB-XL ECG dataset download and preprocessing.

Downloads the PTB-XL large-scale ECG dataset from PhysioNet, downsamples
from 500 Hz to 100 Hz, and extracts a fixed 128-sample window (index 3
of 7 non-overlapping windows from the 1000-sample record at 100 Hz),
producing windows of shape (T=128, D=12).

The official 10-fold split is used: folds 1-8 = train, fold 9 = val,
fold 10 = test. This is by-patient, ensuring no patient leakage.

References
----------
Wagner, P. et al. (2020). PTB-XL, a large publicly available
electrocardiography dataset. Scientific Data, 7(1), 154.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# PhysioNet PTB-XL URL (version 1.0.3)
PTB_XL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

# 5 diagnostic superclasses
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_TO_IDX = {c: i for i, c in enumerate(sorted(SUPERCLASSES))}
# Sorted: CD=0, HYP=1, MI=2, NORM=3, STTC=4


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_ptb_xl(
    data_dir: Path | str,
    url: str = PTB_XL_URL,
    force: bool = False,
) -> Path:
    """Download and extract PTB-XL dataset.

    Parameters
    ----------
    data_dir : Path or str
        Root directory for the dataset.
    url : str
        Download URL.
    force : bool
        Re-download even if present.

    Returns
    -------
    Path
        Path to the extracted dataset root containing ptbxl_database.csv.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ptb-xl-1.0.3.zip"

    # The zip extracts to a subdirectory
    extract_dir = data_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    if extract_dir.exists() and not force:
        logger.info("Dataset already extracted at %s", extract_dir)
        return extract_dir

    if not zip_path.exists() or force:
        logger.info("Downloading PTB-XL from %s", url)
        import urllib.request
        urllib.request.urlretrieve(url, str(zip_path))
        logger.info("Download complete: %s", zip_path)

    # Log SHA-256
    file_hash = _sha256_file(zip_path)
    logger.info("SHA-256 of archive: %s", file_hash)
    hash_file = data_dir / "archive_sha256.txt"
    with open(hash_file, "w", encoding="utf-8") as fh:
        fh.write(file_hash + "\n")

    # Extract
    logger.info("Extracting archive to %s", data_dir)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(data_dir))

    if not extract_dir.exists():
        # Try to find the extracted directory
        candidates = [d for d in data_dir.iterdir() if d.is_dir() and "ptb" in d.name.lower()]
        if candidates:
            extract_dir = candidates[0]
        else:
            raise FileNotFoundError(
                f"Expected directory after extraction in {data_dir}"
            )

    logger.info("Extraction complete: %s", extract_dir)
    return extract_dir


def _load_wfdb_record(
    record_path: str,
    dataset_dir: Path,
) -> np.ndarray:
    """Load a single WFDB record using wfdb.

    Parameters
    ----------
    record_path : str
        Relative path to the record (without extension), e.g. "records500/00000/00001_hr".
    dataset_dir : Path
        Root of extracted PTB-XL.

    Returns
    -------
    np.ndarray
        ECG signal, shape (n_samples, 12).
    """
    import wfdb

    full_path = str(dataset_dir / record_path)
    record = wfdb.rdrecord(full_path)
    return record.p_signal.astype(np.float32)


def _downsample(
    signal: np.ndarray,
    factor: int = 5,
) -> np.ndarray:
    """Downsample signal by integer factor using scipy.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (n_samples, n_channels).
    factor : int
        Downsampling factor (500->100 Hz = factor 5).

    Returns
    -------
    np.ndarray
        Downsampled signal, shape (n_samples // factor, n_channels).
    """
    from scipy.signal import decimate

    # decimate works on axis=0 by default
    # Apply per channel to avoid issues
    result = np.zeros(
        (signal.shape[0] // factor, signal.shape[1]),
        dtype=np.float32,
    )
    for ch in range(signal.shape[1]):
        result[:, ch] = decimate(signal[:, ch], factor, zero_phase=True)
    return result


def _select_window(
    signal: np.ndarray,
    window_index: int = 3,
    window_size: int = 128,
) -> np.ndarray:
    """Select a fixed window from a downsampled record.

    With 1000 samples at 100 Hz, there are 7 non-overlapping windows
    of size 128 (7*128 = 896 < 1000). Window 3 (0-indexed) spans
    samples 384-511.

    Parameters
    ----------
    signal : np.ndarray
        Downsampled signal, shape (1000, 12).
    window_index : int
        Which window to select (0-6).
    window_size : int
        Window size.

    Returns
    -------
    np.ndarray
        Window of shape (128, 12).
    """
    start = window_index * window_size
    end = start + window_size
    if end > signal.shape[0]:
        raise ValueError(
            f"Window {window_index} (samples {start}-{end}) exceeds "
            f"signal length {signal.shape[0]}"
        )
    return signal[start:end]


def _parse_scp_codes(scp_codes_str: str) -> Dict[str, float]:
    """Parse SCP codes from the database CSV string representation.

    The CSV stores SCP codes as a dict-like string, e.g.
    ``"{'NORM': 100.0}"``.
    """
    import ast
    try:
        return ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return {}


def load_ptb_xl_data(
    dataset_dir: Path,
    downsample_factor: int = 5,
    window_index: int = 3,
    window_size: int = 128,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load and preprocess PTB-XL dataset.

    Uses the official fold-based split: folds 1-8 train, fold 9 val,
    fold 10 test. All records are included (self-supervised pretraining
    does not require diagnostic labels). Superclass labels are still
    computed for optional downstream probing.

    Parameters
    ----------
    dataset_dir : Path
        Root of extracted PTB-XL.
    downsample_factor : int
        Factor for downsampling (500 Hz -> 100 Hz).
    window_index : int
        Which non-overlapping window to select.
    window_size : int
        Size of each window.

    Returns
    -------
    dict
        ``{"train": {"signals": (N,128,12), "labels": (N,), "patients": (N,)},
           "val": ..., "test": ...}``
    """
    import pandas as pd

    # Load database CSV
    db_path = dataset_dir / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"Database CSV not found: {db_path}")

    df = pd.read_csv(db_path, index_col="ecg_id")

    # Load SCP statements for superclass mapping (optional, for labeling)
    scp_path = dataset_dir / "scp_statements.csv"
    scp_df = pd.read_csv(scp_path, index_col=0)

    # Build SCP code -> superclass mapping
    scp_to_superclass = {}
    for code, row in scp_df.iterrows():
        if pd.notna(row.get("diagnostic_class")):
            scp_to_superclass[code] = row["diagnostic_class"]

    # Assign superclass to each record (best-effort, may be None)
    def _get_superclass(scp_codes_str):
        codes = _parse_scp_codes(scp_codes_str)
        best_class = None
        best_likelihood = 0.0
        for code, likelihood in codes.items():
            sc = scp_to_superclass.get(code)
            if sc and sc in SUPERCLASSES and likelihood > best_likelihood:
                best_class = sc
                best_likelihood = likelihood
        return best_class

    df["superclass"] = df["scp_codes"].apply(_get_superclass)

    # Include ALL records (self-supervised learning does not need labels)
    # Records without a superclass get label -1 (unlabeled)
    logger.info(
        "Records with diagnostic superclass: %d / %d",
        df["superclass"].notna().sum(), len(df),
    )

    # Split by official folds
    split_map = {
        "train": df[df["strat_fold"].isin(range(1, 9))],
        "val": df[df["strat_fold"] == 9],
        "test": df[df["strat_fold"] == 10],
    }

    result = {}
    for split_name, split_df in split_map.items():
        signals_list = []
        labels_list = []
        patients_list = []

        for ecg_id, row in split_df.iterrows():
            # Use 500 Hz records for higher quality downsampling
            record_path = row["filename_hr"]  # 500 Hz path

            try:
                signal_500hz = _load_wfdb_record(record_path, dataset_dir)
            except Exception as e:
                logger.warning(
                    "Skipping record %d: %s", ecg_id, e,
                )
                continue

            # Downsample 500 -> 100 Hz
            signal_100hz = _downsample(signal_500hz, factor=downsample_factor)

            # Select window
            try:
                window = _select_window(
                    signal_100hz,
                    window_index=window_index,
                    window_size=window_size,
                )
            except ValueError as e:
                logger.warning("Skipping record %d: %s", ecg_id, e)
                continue

            label_idx = SUPERCLASS_TO_IDX.get(row["superclass"], -1)
            patient_id = row["patient_id"]

            signals_list.append(window)
            labels_list.append(label_idx)
            patients_list.append(patient_id)

        if signals_list:
            signals = np.stack(signals_list, axis=0)
            labels = np.array(labels_list, dtype=np.int32)
            patients = np.array(patients_list, dtype=np.int32)
        else:
            signals = np.empty((0, window_size, 12), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int32)
            patients = np.empty((0,), dtype=np.int32)

        result[split_name] = {
            "signals": signals,
            "labels": labels,
            "patients": patients,
        }
        logger.info(
            "%s: %d windows, shape=%s",
            split_name, len(signals), signals.shape,
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "ptb_xl"
    dataset_dir = download_ptb_xl(data_dir / "raw")
    raw_data = load_ptb_xl_data(dataset_dir)
    for split_name, split_data in raw_data.items():
        logger.info(
            "%s: %d windows, shape=%s",
            split_name, len(split_data["signals"]), split_data["signals"].shape,
        )
