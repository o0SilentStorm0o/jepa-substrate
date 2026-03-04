"""Google Speech Commands V2 download and preprocessing.

Downloads the 35-class Speech Commands V2 dataset, computes log-mel
spectrograms (n_fft=512, hop_length=125, n_mels=80), and produces
windows of shape (T=128, D=80).

The official by-speaker validation/test lists are used to partition
the data, ensuring no speaker leakage across splits.

References
----------
Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary
Speech Recognition. arXiv:1804.03209.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Official Speech Commands V2 URL
SC_V2_URL = (
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
)

# All 35 classes (30 keywords + 5 auxiliary)
CLASSES_35 = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "visual",
    "wow", "yes", "zero",
]

# Sorted for deterministic label mapping
CLASSES_35_SORTED = sorted(CLASSES_35)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_35_SORTED)}


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_speech_commands(
    data_dir: Path | str,
    url: str = SC_V2_URL,
    force: bool = False,
) -> Path:
    """Download and extract Speech Commands V2.

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
        Path to the extracted dataset root.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "speech_commands_v0.02.tar.gz"
    extract_dir = data_dir / "speech_commands_v2"

    if extract_dir.exists() and not force:
        logger.info("Dataset already extracted at %s", extract_dir)
        return extract_dir

    if not tar_path.exists() or force:
        logger.info("Downloading Speech Commands V2 from %s", url)
        urllib.request.urlretrieve(url, str(tar_path))
        logger.info("Download complete: %s", tar_path)

    # Log SHA-256
    file_hash = _sha256_file(tar_path)
    logger.info("SHA-256 of archive: %s", file_hash)
    hash_file = data_dir / "archive_sha256.txt"
    with open(hash_file, "w", encoding="utf-8") as fh:
        fh.write(file_hash + "\n")

    # Extract
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting archive to %s", extract_dir)
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(str(extract_dir))

    logger.info("Extraction complete: %s", extract_dir)
    return extract_dir


def _load_list_file(path: Path) -> List[str]:
    """Load a newline-separated list file (validation_list.txt, etc.)."""
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _extract_speaker_id(filename: str) -> str:
    """Extract speaker hash from filename like 'nohash_0.wav' → speaker ID.

    Files are structured as <class>/<speaker_id>_nohash_<n>.wav
    """
    basename = Path(filename).stem  # e.g. "abc123_nohash_0"
    return basename.rsplit("_nohash_", 1)[0]


def _load_wav(wav_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a WAV file and return as float32 array, padded/trimmed to 1s.

    Parameters
    ----------
    wav_path : Path
        Path to WAV file.
    target_sr : int
        Expected sample rate.

    Returns
    -------
    np.ndarray
        Audio waveform of shape (target_sr,) = (16000,).
    """
    import scipy.io.wavfile as wavfile

    sr, audio = wavfile.read(str(wav_path))
    if sr != target_sr:
        raise ValueError(
            f"Expected {target_sr} Hz, got {sr} Hz for {wav_path}"
        )

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    # Pad or trim to exactly 1 second
    target_len = target_sr
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    elif len(audio) > target_len:
        audio = audio[:target_len]

    return audio


def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 125,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute log-mel spectrogram for a 1-second waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Audio signal of shape (sr,).
    sr : int
        Sample rate.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length in samples.
    n_mels : int
        Number of mel bands.

    Returns
    -------
    np.ndarray
        Log-mel spectrogram of shape (T, n_mels) where T = ceil(sr/hop_length).
        With sr=16000, hop=125: T=128.
    """
    from scipy.signal import stft as _stft

    # Pad waveform so that STFT produces exactly T=128 frames
    # Required: n_frames = (n_samples - n_fft) / hop_length + 1 = 128
    # => n_samples = (128 - 1) * hop_length + n_fft
    required_len = (128 - 1) * hop_length + n_fft
    if len(waveform) < required_len:
        waveform = np.pad(waveform, (0, required_len - len(waveform)))
    elif len(waveform) > required_len:
        waveform = waveform[:required_len]

    # Compute STFT
    # scipy.signal.stft returns (freqs, times, Zxx)
    _, _, Zxx = _stft(
        waveform,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        boundary=None,
        padded=False,
    )
    # Power spectrogram: |Zxx|^2, shape (n_fft//2+1, n_frames)
    power = np.abs(Zxx) ** 2

    # Build mel filterbank
    mel_fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)

    # Apply mel filterbank: (n_mels, n_fft//2+1) @ (n_fft//2+1, n_frames)
    mel_spec = mel_fb @ power  # (n_mels, n_frames)

    # Log scale (with floor for numerical stability)
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    # Transpose to (n_frames, n_mels) = (T, D)
    log_mel = log_mel.T.astype(np.float32)

    return log_mel


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
) -> np.ndarray:
    """Create a mel filterbank matrix.

    Parameters
    ----------
    sr : int
        Sample rate.
    n_fft : int
        FFT size.
    n_mels : int
        Number of mel bands.

    Returns
    -------
    np.ndarray
        Filterbank of shape (n_mels, n_fft // 2 + 1).
    """
    n_freqs = n_fft // 2 + 1
    fmax = sr / 2.0

    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    # Convert to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def load_speech_commands_data(
    dataset_dir: Path,
    n_fft: int = 512,
    hop_length: int = 125,
    n_mels: int = 80,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load and preprocess all Speech Commands V2 data.

    Uses the official validation_list.txt and testing_list.txt for
    by-speaker partitioning. All remaining samples go to training.

    Parameters
    ----------
    dataset_dir : Path
        Root of extracted Speech Commands V2.
    n_fft, hop_length, n_mels : int
        Log-mel spectrogram parameters.

    Returns
    -------
    dict
        ``{"train": {"signals": (N,T,D), "labels": (N,), "speakers": (N,)},
           "val": ..., "test": ...}``
    """
    # Load official split lists
    val_list = set(_load_list_file(dataset_dir / "validation_list.txt"))
    test_list = set(_load_list_file(dataset_dir / "testing_list.txt"))

    splits: Dict[str, Dict[str, list]] = {
        "train": {"signals": [], "labels": [], "speakers": []},
        "val": {"signals": [], "labels": [], "speakers": []},
        "test": {"signals": [], "labels": [], "speakers": []},
    }

    for class_name in CLASSES_35_SORTED:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            logger.warning("Missing class directory: %s", class_dir)
            continue

        label_idx = CLASS_TO_IDX[class_name]

        for wav_file in sorted(class_dir.glob("*.wav")):
            relative = f"{class_name}/{wav_file.name}"
            speaker_id = _extract_speaker_id(wav_file.name)

            # Determine split
            if relative in val_list:
                split_name = "val"
            elif relative in test_list:
                split_name = "test"
            else:
                split_name = "train"

            # Load and convert to log-mel
            try:
                waveform = _load_wav(wav_file)
                log_mel = compute_log_mel_spectrogram(
                    waveform,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                )  # (T, D)
            except Exception as e:
                logger.warning("Skipping %s: %s", wav_file, e)
                continue

            splits[split_name]["signals"].append(log_mel)
            splits[split_name]["labels"].append(label_idx)
            splits[split_name]["speakers"].append(speaker_id)

    result = {}
    for split_name, data in splits.items():
        n = len(data["signals"])
        if n == 0:
            logger.warning("No samples in %s split", split_name)
            signals = np.empty((0, 128, n_mels), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int32)
        else:
            signals = np.stack(data["signals"], axis=0)  # (N, T, D)
            labels = np.array(data["labels"], dtype=np.int32)

        # Ensure T dimension is exactly 128 (trim if slightly longer)
        if signals.shape[1] > 128:
            signals = signals[:, :128, :]
        elif signals.shape[1] < 128:
            pad_len = 128 - signals.shape[1]
            signals = np.pad(
                signals, ((0, 0), (0, pad_len), (0, 0)),
                mode="constant",
            )

        result[split_name] = {
            "signals": signals,
            "labels": labels,
            "speakers": np.array(data["speakers"]),
        }
        logger.info(
            "%s: %d windows, shape=%s",
            split_name, n, signals.shape,
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "speech_commands_v2"
    dataset_dir = download_speech_commands(data_dir / "raw")
    raw_data = load_speech_commands_data(dataset_dir)
    for split_name, split_data in raw_data.items():
        logger.info(
            "%s: %d windows, shape=%s",
            split_name, len(split_data["signals"]), split_data["signals"].shape,
        )
