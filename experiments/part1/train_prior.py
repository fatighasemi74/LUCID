"""
experiments/part1/train_prior.py
=================================
Train the unconditional ECG diffusion prior on PTB-XL Lead II at 500 Hz.

What this script does (in order)
---------------------------------
1.  Load PTB-XL metadata from local CSV
2.  Stream Lead II windows from local records500/ files
3.  Compute global normalization stats from training split → save to configs/
4.  Train UNet1D with GaussianDiffusion (x0 prediction, cosine schedule)
5.  Save best checkpoint (lowest val loss) to experiments/part1/checkpoints/

Usage
-----
# Full training run
python experiments/part1/train_prior.py

# Fast smoke-test (100 records, 3 epochs)
python experiments/part1/train_prior.py --max-records 100 --epochs 3

# Resume from checkpoint
python experiments/part1/train_prior.py --resume experiments/part1/checkpoints/prior_last.pt

Paths (edit PTBXL_DIR if your dataset is elsewhere)
----------------------------------------------------
PTBXL_DIR : D:/p2p_audit/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── repo root on path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.ptbxl import load_metadata, split_metadata, load_record, extract_lead_ii
from data.preprocess import (
    NormStats, extract_nonoverlapping_windows,
    ECGWindowDataset, WINDOW_SAMPLES,
)
from models.diffusion_core import GaussianDiffusion, ddim_sample
from models.unet1d import UNet1D, count_parameters

# ======================================================================
# Configuration
# ======================================================================

PTBXL_DIR = Path(
    r"D:\p2p_audit\ptbxl"
    r"\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
PTBXL_CSV    = PTBXL_DIR / "ptbxl_database.csv"
RECORDS_DIR  = PTBXL_DIR / "records500"

OUT_DIR      = ROOT / "experiments" / "part1" / "checkpoints"
NORM_STATS   = ROOT / "configs" / "norm_stats.json"

# ── model ──────────────────────────────────────────────────────────────
BASE_CH      = 64
TIME_DIM     = 128
N_RES        = 2
DROPOUT      = 0.1

# ── diffusion ──────────────────────────────────────────────────────────
DIFF_T       = 1000
SCHEDULE     = "cosine"
PRED_TYPE    = "x0"

# ── training ───────────────────────────────────────────────────────────
EPOCHS       = 200
BATCH_SIZE   = 8       # safe for 4 GB VRAM; reduce to 4 if OOM
LR           = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
VAL_EVERY    = 5       # validate every N epochs
SEED         = 42

# ── DDIM validation samples ────────────────────────────────────────────
N_VAL_SAMPLES = 4      # quick generation check during training
DDIM_STEPS    = 20     # fast during training; use 50 at final evaluation


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
# Data loading (local files)
# ======================================================================

def load_records_local(
    metadata    : "pd.DataFrame",
    max_records : int | None = None,
    split_name  : str = "train",
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Load Lead II windows from local PTB-XL records500/ files.

    Returns
    -------
    windows  : list of (WINDOW_SAMPLES,) float32 arrays (physical mV)
    meta     : list of dicts with ecg_id, strat_fold, filename
    """
    records = metadata.reset_index()
    if max_records is not None:
        records = records.iloc[:max_records]

    windows, meta = [], []
    n_skip = 0
    t0 = time.time()

    print(f"[data] loading {len(records)} {split_name} records from local disk …")

    for i, (_, row) in enumerate(records.iterrows()):
        # Build local file path from filename_hr field
        # filename_hr looks like "records500/00000/00001_hr"
        rel_path = str(row["filename_hr"]).replace("/", "\\")
        # wfdb appends extension; files are .hea + .dat
        # We just need the stem path to pass to wfdb
        local_record = str(PTBXL_DIR / rel_path)

        try:
            import wfdb
            rec  = wfdb.rdrecord(local_record)
            sig  = rec.p_signal.astype(np.float32)
            lead = extract_lead_ii(sig)
        except Exception as e:
            n_skip += 1
            continue

        wins = extract_nonoverlapping_windows(lead)
        for w in wins:
            windows.append(w)
            meta.append({
                "ecg_id"    : int(row.get("ecg_id", i)),
                "strat_fold": int(row.get("strat_fold", 0)),
                "filename"  : str(row.get("filename_hr", "")),
            })

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(records)} records  "
                  f"{len(windows)} windows  {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"[data] {split_name}: {len(windows)} windows from "
          f"{len(records)-n_skip} records  ({n_skip} skipped)  "
          f"{elapsed:.1f}s")
    return windows, meta


def build_loaders(
    max_records: int | None = None,
) -> tuple[DataLoader, DataLoader, NormStats]:
    """
    Build train and val DataLoaders plus NormStats.

    NormStats are computed from the training split and saved to
    configs/norm_stats.json for reuse in all subsequent runs.
    """
    import pandas as pd
    meta_all = pd.read_csv(str(PTBXL_CSV), index_col="ecg_id")

    meta_train = split_metadata(meta_all, "train")
    meta_val   = split_metadata(meta_all, "val")

    # Load training data
    train_wins, train_meta = load_records_local(
        meta_train, max_records=max_records, split_name="train"
    )

    # Compute and save normalization stats from training split
    if NORM_STATS.exists():
        print(f"[norm] loading existing stats from {NORM_STATS}")
        stats = NormStats.load(NORM_STATS)
    else:
        print("[norm] computing stats from training windows …")
        stats = NormStats.from_windows(train_wins)
        NORM_STATS.parent.mkdir(parents=True, exist_ok=True)
        stats.save(NORM_STATS)

    # Load validation data (use same stats)
    val_max = max_records // 4 if max_records else None
    val_wins, val_meta = load_records_local(
        meta_val, max_records=val_max, split_name="val"
    )

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

    print(f"[data] train={len(train_ds)} windows  val={len(val_ds)} windows")
    return train_loader, val_loader, stats


# ======================================================================
# Training loop
# ======================================================================

def run_epoch(
    model  : UNet1D,
    diff   : GaussianDiffusion,
    loader : DataLoader,
    device : torch.device,
    opt    : torch.optim.Optimizer | None = None,
) -> float:
    """One training or validation epoch. Returns mean loss."""
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x0, _ in loader:
            x0    = x0.to(device)                         # (B, 1, L)
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


# ======================================================================
# Checkpoint helpers
# ======================================================================

def save_checkpoint(
    path    : Path,
    model   : UNet1D,
    opt     : torch.optim.Optimizer,
    epoch   : int,
    val_loss: float,
    stats   : NormStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model"      : model.state_dict(),
        "optimizer"  : opt.state_dict(),
        "epoch"      : epoch,
        "val_loss"   : val_loss,
        "global_mean": stats.mean,
        "global_std" : stats.std,
        # Architecture config — needed to rebuild the model
        "base_ch"    : BASE_CH,
        "time_dim"   : TIME_DIM,
        "n_res"      : N_RES,
        "T"          : DIFF_T,
        "schedule"   : SCHEDULE,
        "pred_type"  : PRED_TYPE,
    }, str(path))


def load_checkpoint(
    path  : Path,
    model : UNet1D,
    opt   : torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint. Returns (start_epoch, best_val_loss)."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    print(f"[resume] epoch {ckpt['epoch']}  val_loss {ckpt['val_loss']:.6f}")
    return ckpt["epoch"] + 1, ckpt["val_loss"]


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Train ECG diffusion prior on PTB-XL")
    ap.add_argument("--max-records", type=int, default=None,
                    help="cap on training records (None=all ~17k). "
                         "Use 100 for a fast smoke-test.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint to resume from")
    args = ap.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Pulse2Posterior — ECG Diffusion Prior Training")
    print("=" * 60)
    print(f"Device  : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM    : {vram:.1f} GB")
    print(f"PTB-XL  : {PTBXL_DIR}")
    print(f"Epochs  : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    if args.max_records:
        print(f"[SMOKE TEST] max_records={args.max_records}")
    print()

    # ── data ──────────────────────────────────────────────────────────
    train_loader, val_loader, stats = build_loaders(args.max_records)

    # ── model ─────────────────────────────────────────────────────────
    model = UNet1D(
        base_ch=BASE_CH, time_dim=TIME_DIM,
        n_res=N_RES, dropout=DROPOUT,
    ).to(device)
    diff = GaussianDiffusion(T=DIFF_T, schedule=SCHEDULE, pred_type=PRED_TYPE).to(device)

    print(f"[model] UNet1D  params: {count_parameters(model):,}")

    # ── optimiser ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr / 20
    )

    # ── resume ────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    best_path   = OUT_DIR / "prior_best.pt"
    last_path   = OUT_DIR / "prior_last.pt"

    if args.resume:
        start_epoch, best_val = load_checkpoint(
            Path(args.resume), model, opt, device
        )

    # ── training loop ─────────────────────────────────────────────────
    history = {"train": [], "val": []}

    for ep in range(start_epoch, args.epochs + 1):
        t0       = time.time()
        tr_loss  = run_epoch(model, diff, train_loader, device, opt)
        scheduler.step()
        elapsed  = time.time() - t0

        # Validation
        if ep % VAL_EVERY == 0 or ep == 1:
            val_loss = run_epoch(model, diff, val_loader, device, opt=None)
            history["val"].append((ep, val_loss))
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(best_path, model, opt, ep, val_loss, stats)
            marker = " ← best" if improved else ""
            print(f"epoch {ep:03d}/{args.epochs}  "
                  f"train {tr_loss:.5f}  val {val_loss:.5f}  "
                  f"{elapsed:.1f}s{marker}")
        else:
            val_loss = float("nan")
            print(f"epoch {ep:03d}/{args.epochs}  "
                  f"train {tr_loss:.5f}  {elapsed:.1f}s")

        history["train"].append((ep, tr_loss))

        # Always save last checkpoint (for resuming)
        save_checkpoint(last_path, model, opt, ep, val_loss, stats)

    # ── done ──────────────────────────────────────────────────────────
    print()
    print(f"Training complete.")
    print(f"Best checkpoint : {best_path}  (val_loss={best_val:.6f})")
    print(f"Norm stats      : {NORM_STATS}")
    print()
    print("Next step: run experiments/part1/generate.py")


if __name__ == "__main__":
    main()
