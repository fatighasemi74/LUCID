"""
experiments/part1/train_bootstrap.py
=====================================
Train K=10 ECG diffusion priors on bootstrap resamples of PTB-XL.

What this does
--------------
For k = 0, 1, ..., K-1:
    1. Sample patients WITH REPLACEMENT from D_ECG (PTB-XL train split)
       — same number of patients as the original, but some appear multiple
       times and some not at all. This is the standard non-parametric bootstrap.
    2. Train a full UNet1D diffusion prior on that resample (same hyperparameters
       as single training, same number of epochs).
    3. Save checkpoint to experiments/part1/checkpoints/prior_k{k:02d}.pt

Output
------
10 checkpoint files:
    experiments/part1/checkpoints/prior_k00.pt
    experiments/part1/checkpoints/prior_k01.pt
    ...
    experiments/part1/checkpoints/prior_k09.pt

Plus one shared norm_stats.json (computed from the FULL training set once,
reused for all K runs — normalization must be identical across all priors).

Runtime estimate
----------------
~7 hours per run × 10 runs = ~70 hours total (3 nights on RTX 3050 Ti).
Runs sequentially. Each run saves its own best checkpoint so if you interrupt
between runs nothing is lost. Resume by setting START_K below.

Monitoring
----------
Watch the per-epoch output. Healthy training looks like:
    epoch 001  train 0.19x  val 0.17x  ← best
    epoch 005  train 0.16x  val 0.16x  ← best
    ...
    epoch 050  train 0.13x  val 0.13x  ← best
If val loss stops improving after epoch 50 and stays flat, that's normal.
If train loss goes UP after epoch 100, learning rate may be too high — but
the cosine schedule handles this automatically.

To stop safely: Ctrl+C between epochs. The last completed checkpoint is always
saved as prior_k{k:02d}_last.pt so you can resume the current k if needed.

To resume from a specific k:
    Edit START_K below and rerun.

GitHub: commit after ALL K runs complete (Commit 2 in the plan).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.ptbxl import (
    load_metadata, split_metadata, load_record, extract_lead_ii
)
from data.preprocess import (
    NormStats, extract_nonoverlapping_windows, ECGWindowDataset
)
from models.diffusion_core import GaussianDiffusion
from models.unet1d import UNet1D, count_parameters

# ======================================================================
# Configuration — edit START_K to resume after interruption
# ======================================================================

K          = 10       # number of bootstrap members (Lakshminarayanan et al. 2017)
START_K    = 0        # set to k+1 to resume after completing run k

PTBXL_DIR  = Path(
    r"D:\p2p_audit\ptbxl"
    r"\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
PTBXL_CSV  = PTBXL_DIR / "ptbxl_database.csv"
OUT_DIR    = ROOT / "experiments" / "part1" / "checkpoints"
NORM_STATS = ROOT / "configs" / "norm_stats.json"

# ── model (same as single training) ───────────────────────────────────
BASE_CH    = 64
TIME_DIM   = 128
N_RES      = 2
DROPOUT    = 0.1

# ── diffusion ─────────────────────────────────────────────────────────
DIFF_T     = 1000
SCHEDULE   = "cosine"
PRED_TYPE  = "x0"

# ── training ──────────────────────────────────────────────────────────
EPOCHS     = 200
BATCH_SIZE = 8
LR         = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP  = 1.0
VAL_EVERY  = 5
SEED       = 42


# ======================================================================
# Reproducibility
# ======================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# Bootstrap patient resample
# ======================================================================

def bootstrap_resample_patients(
    metadata : pd.DataFrame,
    seed     : int,
) -> pd.DataFrame:
    """
    Sample patients WITH REPLACEMENT from the metadata.

    Resamples at the PATIENT level (not window level) so that a patient's
    full 10s recording either appears or doesn't — no partial patients.
    Same number of patients as the original split.

    Parameters
    ----------
    metadata : pd.DataFrame
        Full training split metadata (one row per recording).
    seed : int
        Random seed for this bootstrap run (= k for reproducibility).

    Returns
    -------
    pd.DataFrame
        Resampled metadata. Some patients appear multiple times,
        some not at all. Total rows = len(metadata).
    """
    rng = np.random.default_rng(seed)

    # Get unique patient IDs
    if "patient_id" in metadata.columns:
        patients = metadata["patient_id"].unique()
    else:
        # Fall back to record-level resample if no patient_id column
        patients = metadata.index.values

    # Sample patients with replacement
    resampled_patients = rng.choice(patients, size=len(patients), replace=True)

    # Gather all records for the resampled patients
    if "patient_id" in metadata.columns:
        rows = []
        for pid in resampled_patients:
            rows.append(metadata[metadata["patient_id"] == pid])
        return pd.concat(rows, ignore_index=False)
    else:
        return metadata.loc[resampled_patients]


# ======================================================================
# Data loading
# ======================================================================

def load_windows_local(
    metadata   : pd.DataFrame,
    split_name : str = "train",
) -> tuple[list[np.ndarray], list[dict]]:
    """Load Lead II windows from local PTB-XL files."""
    records  = metadata.reset_index()
    windows, meta = [], []
    n_skip = 0
    t0 = time.time()

    print(f"  [data] loading {len(records)} {split_name} records …")
    for i, (_, row) in enumerate(records.iterrows()):
        rel  = str(row["filename_hr"]).replace("/", "\\")
        path = str(PTBXL_DIR / rel)
        try:
            import wfdb
            rec  = wfdb.rdrecord(path)
            sig  = rec.p_signal.astype(np.float32)
            lead = extract_lead_ii(sig)
        except Exception:
            n_skip += 1
            continue

        wins = extract_nonoverlapping_windows(lead)
        for w in wins:
            windows.append(w)
            meta.append({
                "ecg_id"    : int(row.get("ecg_id", i)),
                "strat_fold": int(row.get("strat_fold", 0)),
            })

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(records)} records  "
                  f"{len(windows)} windows  {time.time()-t0:.0f}s")

    print(f"  [data] {split_name}: {len(windows)} windows  "
          f"({n_skip} skipped)  {time.time()-t0:.1f}s")
    return windows, meta


# ======================================================================
# Single training run
# ======================================================================

def run_epoch(
    model  : UNet1D,
    diff   : GaussianDiffusion,
    loader : DataLoader,
    device : torch.device,
    opt    : torch.optim.Optimizer | None = None,
) -> float:
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x0, _ in loader:
            x0    = x0.to(device)
            t     = torch.randint(0, diff.T, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            x_t   = diff.q_sample(x0, t, noise)
            pred  = model(x_t, t)
            loss  = nn.functional.mse_loss(pred, diff.target(x0, noise))
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            total += loss.item()
            n     += 1
    return total / max(1, n)


def train_one_prior(
    k          : int,
    meta_train : pd.DataFrame,
    meta_val   : pd.DataFrame,
    stats      : NormStats,
    device     : torch.device,
    n_epochs   : int = EPOCHS,
) -> Path:
    """
    Train one bootstrap prior θ_k and save the best checkpoint.

    Parameters
    ----------
    k          : bootstrap index (0-indexed)
    meta_train : full training metadata (will be resampled inside)
    meta_val   : validation metadata (NOT resampled — fixed for all k)
    stats      : global normalization stats (shared across all k)
    device     : torch device

    Returns
    -------
    Path to the saved best checkpoint.
    """
    print()
    print("=" * 60)
    print(f"Bootstrap run k={k:02d} / {K-1}")
    print("=" * 60)

    # ── bootstrap resample training patients ──────────────────────────
    seed_everything(SEED + k)   # different seed per run
    meta_k = bootstrap_resample_patients(meta_train, seed=k)
    print(f"  Bootstrap resample: {len(meta_k)} records "
          f"(from {len(meta_train)} originals, with replacement)")

    # ── load data ─────────────────────────────────────────────────────
    train_wins, train_meta = load_windows_local(meta_k, split_name=f"train_k{k:02d}")
    val_wins,   val_meta   = load_windows_local(meta_val, split_name="val")

    train_ds = ECGWindowDataset(train_wins, stats, train_meta)
    val_ds   = ECGWindowDataset(val_wins,   stats, val_meta)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"  train={len(train_ds)} windows  val={len(val_ds)} windows")

    # ── model + diffusion ─────────────────────────────────────────────
    model = UNet1D(
        base_ch=BASE_CH, time_dim=TIME_DIM,
        n_res=N_RES, dropout=DROPOUT,
    ).to(device)
    diff = GaussianDiffusion(
        T=DIFF_T, schedule=SCHEDULE, pred_type=PRED_TYPE
    ).to(device)

    if k == 0:
        print(f"  UNet1D params: {count_parameters(model):,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=LR / 20
    )

    best_val  = float("inf")
    best_path = OUT_DIR / f"prior_k{k:02d}.pt"
    last_path = OUT_DIR / f"prior_k{k:02d}_last.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────
    t_run = time.time()
    for ep in range(1, n_epochs + 1):
        t0      = time.time()
        tr_loss = run_epoch(model, diff, train_loader, device, opt)
        scheduler.step()
        elapsed = time.time() - t0

        if ep % VAL_EVERY == 0 or ep == 1:
            val_loss = run_epoch(model, diff, val_loader, device, opt=None)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                torch.save({
                    "model"      : model.state_dict(),
                    "epoch"      : ep,
                    "val_loss"   : val_loss,
                    "k"          : k,
                    "global_mean": stats.mean,
                    "global_std" : stats.std,
                    "base_ch"    : BASE_CH,
                    "time_dim"   : TIME_DIM,
                    "n_res"      : N_RES,
                    "T"          : DIFF_T,
                    "schedule"   : SCHEDULE,
                    "pred_type"  : PRED_TYPE,
                }, str(best_path))
            marker = " ← best" if improved else ""
            print(f"  k={k:02d} epoch {ep:03d}/{n_epochs}  "
                  f"train {tr_loss:.5f}  val {val_loss:.5f}  "
                  f"{elapsed:.1f}s{marker}")
        else:
            print(f"  k={k:02d} epoch {ep:03d}/{n_epochs}  "
                  f"train {tr_loss:.5f}  {elapsed:.1f}s")

        # Always save last (for safe resuming)
        torch.save({
            "model"      : model.state_dict(),
            "optimizer"  : opt.state_dict(),
            "epoch"      : ep,
            "val_loss"   : best_val,
            "k"          : k,
            "global_mean": stats.mean,
            "global_std" : stats.std,
            "base_ch"    : BASE_CH,
            "time_dim"   : TIME_DIM,
            "n_res"      : N_RES,
            "T"          : DIFF_T,
            "schedule"   : SCHEDULE,
            "pred_type"  : PRED_TYPE,
        }, str(last_path))

    elapsed_total = time.time() - t_run
    print(f"  k={k:02d} done.  best_val={best_val:.6f}  "
          f"total={elapsed_total/3600:.2f}h  "
          f"saved → {best_path.name}")

    return best_path


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train K=10 bootstrap ECG diffusion priors on PTB-XL"
    )
    ap.add_argument("--k", type=int, default=K,
                    help="number of bootstrap members (default 10)")
    ap.add_argument("--start-k", type=int, default=START_K,
                    help="start from this k (0-indexed). Use to resume.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke-test", action="store_true",
                    help="run 2 members × 3 epochs on 200 records for testing")
    args = ap.parse_args()

    smoke = args.smoke_test
    K_run = 2 if smoke else args.k
    EPOCHS_run = 3 if smoke else args.epochs

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Pulse2Posterior — Bootstrap Prior Training")
    print(f"K={K_run}  start_k={args.start_k}  epochs={EPOCHS_run}")
    print(f"Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})"
           if device.type == "cuda" else ""))
    if smoke:
        print("[SMOKE TEST MODE]")
    print("=" * 60)

    # ── load metadata once ────────────────────────────────────────────
    meta_all   = pd.read_csv(str(PTBXL_CSV), index_col="ecg_id")
    meta_train = split_metadata(meta_all, "train")
    meta_val   = split_metadata(meta_all, "val")
    print(f"PTB-XL: {len(meta_train)} train records, {len(meta_val)} val records")

    if smoke:
        meta_train = meta_train.iloc[:200]
        meta_val   = meta_val.iloc[:50]

    # ── compute or load shared normalization stats ────────────────────
    # Stats computed from the FULL training set (not resampled) so that
    # all K priors use identical normalization. Compute once, reuse.
    if NORM_STATS.exists():
        print(f"[norm] loading existing stats from {NORM_STATS}")
        stats = NormStats.load(NORM_STATS)
    else:
        print("[norm] computing stats from full training set …")
        full_wins, _ = load_windows_local(meta_train, "full_train")
        stats = NormStats.from_windows(full_wins)
        NORM_STATS.parent.mkdir(parents=True, exist_ok=True)
        stats.save(NORM_STATS)
        del full_wins   # free memory before training loop

    # ── bootstrap training loop ───────────────────────────────────────
    completed = []
    t_total = time.time()

    for k in range(args.start_k, K_run):
        best_path = train_one_prior(
            k, meta_train, meta_val, stats, device, n_epochs=EPOCHS_run
        )
        completed.append((k, best_path))

        elapsed = (time.time() - t_total) / 3600
        remaining = (K_run - k - 1) * (elapsed / (k - args.start_k + 1))
        print(f"\n[progress] {k+1}/{K_run} runs done  "
              f"elapsed={elapsed:.1f}h  "
              f"est. remaining={remaining:.1f}h")

    # ── summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Bootstrap training complete.")
    print("=" * 60)
    print(f"Checkpoints saved to {OUT_DIR}/")
    for k, path in completed:
        print(f"  k={k:02d}: {path.name}")
    print()
    print("Next steps:")
    print("  1. git add experiments/part1/checkpoints/prior_k*.pt")
    print("     git commit -m 'data: K=10 bootstrap prior checkpoints'")
    print("     git push origin feat/prior-vi-part1")
    print("  2. python experiments/part1/generate.py --snr-db 5 10 20 40")


if __name__ == "__main__":
    main()
