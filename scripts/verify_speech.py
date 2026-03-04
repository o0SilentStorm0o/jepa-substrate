"""Quick download & verify script for Speech Commands V2."""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.download_speech import download_speech_commands, load_speech_commands_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = Path("data/speech_commands_v2/raw")
print("Starting Speech Commands V2 download...")
dataset_dir = download_speech_commands(data_dir)
print(f"Downloaded to: {dataset_dir}")

print("\nLoading and computing log-mel spectrograms (this may take a while)...")
raw = load_speech_commands_data(dataset_dir, n_fft=512, hop_length=125, n_mels=80)

for split_name in ("train", "val", "test"):
    d = raw[split_name]
    print(f"  {split_name}: {d['signals'].shape}")

print(f"\nExpected: train=84843, val=9981, test=11005")
print(f"Actual:   train={len(raw['train']['signals'])}, val={len(raw['val']['signals'])}, test={len(raw['test']['signals'])}")

# Verify T=128, D=80
for split_name in ("train", "val", "test"):
    s = raw[split_name]["signals"]
    assert s.shape[1] == 128, f"{split_name}: T={s.shape[1]}, expected 128"
    assert s.shape[2] == 80, f"{split_name}: D={s.shape[2]}, expected 80"
print("Shape assertions passed: all (N, 128, 80)")
