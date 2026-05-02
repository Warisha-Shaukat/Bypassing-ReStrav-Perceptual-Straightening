import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import sys
import time  # Added for stability pause

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import dinov2_features as d2

# Batch size 16 is a good balance for an RTX 5060
batch_size = 16
device = "cuda:0" if torch.cuda.is_available() else "cpu"

real_root = Path(r"C:\Users\wishi\Desktop\Dip_proj\ReStraV-main\DATA\REAL")
fake_root = Path(r"C:\Users\wishi\Desktop\Dip_proj\ReStraV-main\DATA\FAKE")
output_h5 = Path("training_features.h5")

real_videos = sorted(real_root.rglob("*.mp4"))
fake_videos = sorted(fake_root.rglob("*.mp4"))
all_videos = [(str(p), 1) for p in real_videos] + [(str(p), 0) for p in fake_videos]

print(f"Using Device: {device}")
print(f"Found {len(real_videos)} real and {len(fake_videos)} fake videos.")

# Initialize the HDF5 file
with h5py.File(output_h5, "w") as h5f:
    dt = h5py.special_dtype(vlen=str)
    path_ds = h5f.create_dataset("path", (len(all_videos),), dtype=dt)
    label_ds = h5f.create_dataset("label", (len(all_videos),), dtype="i")
    feat_ds = h5f.create_dataset("features", (len(all_videos), 21), dtype="f")

    # Main extraction loop
    for idx in tqdm(range(0, len(all_videos), batch_size), desc="Extracting features"):
        batch_items = all_videos[idx:idx+batch_size]
        batch_paths = [p for p, _ in batch_items]
        batch_labels = [l for _, l in batch_items]

        try:
            # Extract embeddings using DINOv2
            Z = d2.extract_dinov2_embeddings(batch_paths, device=device)
            # Calculate temporal features
            feats = d2.features_from_Z(Z).cpu().numpy()

            # Save to dataset
            for j, (path, label, f) in enumerate(zip(batch_paths, batch_labels, feats)):
                pos = idx + j
                path_ds[pos] = path
                label_ds[pos] = label
                feat_ds[pos, :] = f
            
            # STABILITY PAUSE: Gives the hardware a 100ms break to prevent
            # "Clock Watchdog Timeout" on high-end laptops.
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError processing batch starting at index {idx}: {e}")
            continue

print(f"\nSuccess! Saved features for {len(all_videos)} videos to {output_h5}")