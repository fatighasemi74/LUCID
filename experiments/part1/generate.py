"""
experiments/part1/generate.py
==============================
Generate the Part 1 semi-synthetic dataset.

Takes real ECG windows from the PTB-XL TEST split and applies the known
forward operator H_φ to produce synthetic PPG observations at multiple
SNR levels. Saves one file per SNR level.

What this produces
------------------
For each SNR in {5, 10, 20, 40} dB, one .npz file containing:
    ecg      : (N, 4000)  real Lead II ECG windows (normalized)
    ppg      : (N, 4000)  synthetic PPG = H_φ(ecg) + noise at this SNR
    phi_true : dict       the TRUE operator parameters used (ground truth
                          for parameter recovery evaluation in run.py)
    snr_db   : float      the SNR level for this file
    mean     : float      global normalization mean (for denormalization)
    std      : float      global normalization std

Why this matters
----------------
Because we BUILT the PPG, we know:
    * the true ECG x (ground truth for reconstruction quality)
    * the true operator H_φ (ground truth for parameter recovery)
    * the true noise level σ² (ground truth for noise estimation)
This is the "answer key" that makes Part 1 answerable.

Usage
-----
python experiments/part1/generate.py

# Custom SNR levels
python experiments/part1/generate.py --snr-db 10 20

# Limit records (for testing)
python experiments/part1/generate.py --max-records 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.ptbxl import load_metadata, split_metadata, load_record, extract_lead_ii
from data.preprocess import (
    NormStats, extract_nonoverlapping_windows, WINDOW_SAMPLES
)

# ======================================================================
# Paths
# ======================================================================

PTBXL_DIR = Path(
    r"D:\p2p_audit\ptbxl"
    r"\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
PTBXL_CSV  = PTBXL_DIR / "ptbxl_database.csv"
NORM_STATS = ROOT / "configs" / "norm_stats.json"
OUT_DIR    = ROOT / "experiments" / "part1" / "synthetic_data"

# ======================================================================
# TRUE operator parameters φ_true (fixed, known for Part 1)
# ======================================================================
# These are the ground-truth values that VI must recover in run.py.
# They are chosen to be physiologically plausible but simple:
#   a     : amplitude scaling (PPG amplitude relative to ECG)
#   b     : DC offset
#   tau   : pulse transit delay in seconds (~0.2s is realistic)
#   sigma : Gaussian kernel std in samples (controls lowpass bandwidth)
#           10 samples at 500 Hz = 20ms smoothing

PHI_TRUE = {
    "a"    : 0.8,     # amplitude scale
    "b"    : 0.05,    # DC offset
    "tau"  : 0.2,     # delay in seconds (100 samples at 500 Hz)
    "sigma": 10.0,    # Gaussian kernel std in samples
}

# SNR levels to generate (dB)
DEFAULT_SNR_DB = [5, 10, 20, 40]

FS = 500   # Hz


# ======================================================================
# Forward operator H_φ  (known, fixed in Part 1)
# ======================================================================

def gaussian_kernel(sigma: float, kernel_size: int = 61) -> torch.Tensor:
    """1-D Gaussian kernel, normalized. Shape (kernel_size,)."""
    half = kernel_size // 2
    x    = torch.arange(-half, half + 1, dtype=torch.float32)
    k    = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def apply_forward_operator(
    ecg   : torch.Tensor,
    phi   : dict,
    fs    : int = FS,
) -> torch.Tensor:
    """
    Apply H_φ(ECG) = a · lowpass(ECG)_{· − τ} + b

    Parameters
    ----------
    ecg : (N, 1, L) normalized ECG windows
    phi : dict with keys a, b, tau, sigma
    fs  : sampling rate in Hz

    Returns
    -------
    ppg_clean : (N, 1, L)  noiseless synthetic PPG
    """
    # Step 1: Gaussian lowpass convolution
    k      = gaussian_kernel(phi["sigma"]).to(ecg.device)
    pad    = len(k) // 2
    h      = F.conv1d(ecg, k.view(1, 1, -1), padding=pad)   # (N,1,L)

    # Step 2: Time delay (integer sample shift)
    shift  = int(round(phi["tau"] * fs))
    if shift > 0:
        h  = torch.roll(h, shifts=shift, dims=-1)
        h[..., :shift] = 0.0    # zero-pad leading samples (causal)

    # Step 3: Amplitude scaling and offset
    h = phi["a"] * h + phi["b"]

    return h


def snr_to_sigma(signal: torch.Tensor, snr_db: float) -> float:
    """
    Compute noise std σ that achieves the target SNR (dB) for this signal.

    SNR_dB = 10 · log10(P_signal / P_noise)
    → P_noise = P_signal / 10^(SNR_dB/10)
    → σ = sqrt(P_noise)

    Parameters
    ----------
    signal : (N, 1, L) — the clean PPG (after H_φ)
    snr_db : float

    Returns
    -------
    sigma : float  noise standard deviation
    """
    p_signal = float(signal.pow(2).mean())
    p_noise  = p_signal / (10 ** (snr_db / 10))
    return float(np.sqrt(p_noise))


# ======================================================================
# Main generation loop
# ======================================================================

def generate(
    max_records: int | None = None,
    snr_levels : list[float] = DEFAULT_SNR_DB,
) -> None:
    """
    Load PTB-XL test split, apply H_φ, add noise at each SNR, save files.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── normalization stats (must exist — run train_prior.py first) ──
    if not NORM_STATS.exists():
        raise FileNotFoundError(
            f"Norm stats not found at {NORM_STATS}\n"
            "Run train_prior.py first to compute and save them."
        )
    stats = NormStats.load(NORM_STATS)

    # ── load test split ──────────────────────────────────────────────
    import pandas as pd
    meta_all  = pd.read_csv(str(PTBXL_CSV), index_col="ecg_id")
    meta_test = split_metadata(meta_all, "test")

    if max_records is not None:
        meta_test = meta_test.iloc[:max_records]

    records = meta_test.reset_index()
    print(f"[generate] loading {len(records)} test records …")

    ecg_windows = []
    for i, (_, row) in enumerate(records.iterrows()):
        rel  = str(row["filename_hr"]).replace("/", "\\")
        path = str(PTBXL_DIR / rel)
        try:
            import wfdb
            rec  = wfdb.rdrecord(path)
            sig  = rec.p_signal.astype(np.float32)
            lead = extract_lead_ii(sig)
        except Exception as e:
            continue

        wins = extract_nonoverlapping_windows(lead)
        for w in wins:
            ecg_windows.append(stats.normalize(w).astype(np.float32))

    if not ecg_windows:
        raise RuntimeError("No windows loaded. Check PTB-XL path.")

    print(f"[generate] {len(ecg_windows)} ECG windows loaded")

    # Stack and move to tensor
    ecg_np  = np.stack(ecg_windows)                       # (N, L)
    ecg_t   = torch.tensor(ecg_np).unsqueeze(1)           # (N, 1, L)

    # ── apply forward operator ───────────────────────────────────────
    print(f"[generate] applying H_phi with phi_true={PHI_TRUE}")
    with torch.no_grad():
        ppg_clean = apply_forward_operator(ecg_t, PHI_TRUE)  # (N, 1, L)

    # ── generate one file per SNR level ─────────────────────────────
    for snr_db in snr_levels:
        sigma = snr_to_sigma(ppg_clean, snr_db)
        noise = torch.randn_like(ppg_clean) * sigma
        ppg_noisy = (ppg_clean + noise).squeeze(1).numpy()  # (N, L)

        out_path = OUT_DIR / f"snr_{int(snr_db):02d}dB.npz"
        np.savez(
            str(out_path),
            ecg      = ecg_np,                  # (N, L) normalized ECG
            ppg      = ppg_noisy,               # (N, L) synthetic PPG
            snr_db   = np.float32(snr_db),
            noise_sigma = np.float32(sigma),
            # phi_true stored as individual arrays (npz doesn't support dicts)
            phi_a    = np.float32(PHI_TRUE["a"]),
            phi_b    = np.float32(PHI_TRUE["b"]),
            phi_tau  = np.float32(PHI_TRUE["tau"]),
            phi_sigma= np.float32(PHI_TRUE["sigma"]),
            # normalization stats for denormalization at evaluation time
            global_mean = np.float32(stats.mean),
            global_std  = np.float32(stats.std),
        )
        print(f"  saved {out_path.name}  "
              f"(N={len(ecg_np)}, SNR={snr_db}dB, σ_noise={sigma:.5f})")

    print()
    print(f"[generate] done. Files saved to {OUT_DIR}/")
    print()
    print("phi_true (ground truth for parameter recovery):")
    for k, v in PHI_TRUE.items():
        print(f"  {k:8s} = {v}")
    print()
    print("Next step: run experiments/part1/run.py")


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Part 1 semi-synthetic PPG dataset"
    )
    ap.add_argument(
        "--snr-db", nargs="+", type=float, default=DEFAULT_SNR_DB,
        help="SNR levels in dB (default: 5 10 20 40)"
    )
    ap.add_argument(
        "--max-records", type=int, default=None,
        help="cap on test records for quick testing"
    )
    args = ap.parse_args()

    generate(
        max_records=args.max_records,
        snr_levels=args.snr_db,
    )


if __name__ == "__main__":
    main()
