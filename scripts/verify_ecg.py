"""Quick download & verify script for PTB-XL ECG."""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.download_ecg import download_ptb_xl, load_ptb_xl_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = Path("data/ptb_xl/raw")
print("Starting PTB-XL download...")
dataset_dir = download_ptb_xl(data_dir)
print(f"Downloaded to: {dataset_dir}")

print("\nLoading PTB-XL records (downsampling + windowing)...")
raw = load_ptb_xl_data(dataset_dir, downsample_factor=5, window_index=3, window_size=128)

for split_name in ("train", "val", "test"):
    d = raw[split_name]
    print(f"  {split_name}: {d['signals'].shape}")

print(f"\nExpected: train=17441, val=2203, test=2193")
print(f"Actual:   train={len(raw['train']['signals'])}, val={len(raw['val']['signals'])}, test={len(raw['test']['signals'])}")

# Verify T=128, D=12
for split_name in ("train", "val", "test"):
    s = raw[split_name]["signals"]
    assert s.shape[1] == 128, f"{split_name}: T={s.shape[1]}, expected 128"
    assert s.shape[2] == 12, f"{split_name}: D={s.shape[2]}, expected 12"
print("Shape assertions passed: all (N, 128, 12)")
