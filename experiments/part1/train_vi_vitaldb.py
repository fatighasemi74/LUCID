"""
experiments/part1/train_vi_vitaldb.py
======================================
Fit q_lambda*(Phi) on VitalDB paired ECG+PPG data.

New structure:
    Phi = [log_sigma2]  dim=1
    Z   = [tau, a, b]   dim=3  — handled by MCMC

Approach: empirical estimation of log_sigma2 from residual variance
after least-squares fit of Z per window. VI gradient-based approach
fails because fixed Z=(0.2,0.8,0.05) is wrong for real VitalDB data.

Usage
-----
python experiments/part1/train_vi_vitaldb.py --max-cases 200

Saves
-----
experiments/part1/checkpoints/lambda_star_vitaldb.pt
experiments/part1/checkpoints/vitaldb_stats.npy
experiments/part1/checkpoints/vitaldb_test_ids.npy
"""

import argparse
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.vi import save_lambda, MeanFieldGaussian
from data.vitaldb import (find_paired_cases, split_case_ids,
                           compute_paired_stats, VitalDBPairedDataset,
                           FS)
from torch.utils.data import DataLoader

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def apply_h(ecg_t, tau_s, k, fs=500):
    """Apply lowpass + delay to ECG. Returns filtered signal."""
    h = F_nn.conv1d(ecg_t.unsqueeze(0).unsqueeze(0),
                    k.view(1, 1, -1), padding=k.shape[0]//2).squeeze()
    if tau_s > 0:
        h = torch.roll(h, tau_s)
        h[:tau_s] = 0.0
    return h


def fit_z_least_squares(ecg_t, ppg_t, fs=500):
    """
    Find (tau, a, b) minimizing ||ppg - (a * lowpass(shift(ecg, tau)) + b)||^2
    via grid search over tau + linear regression for (a, b).
    Returns (tau_s, a, b, residual_variance).
    """
    # Build lowpass kernel once
    half = 30
    t = torch.arange(-half, half+1, dtype=torch.float32)
    k = torch.exp(-t**2 / (2*10.0**2))
    k = k / k.sum()

    best_err = float('inf')
    best = (int(0.1*fs), 1.0, 0.0)

    # Grid over tau: 0 to 0.5 seconds in 10ms steps
    for tau_s in range(0, int(0.5*fs), int(0.01*fs)):
        h = apply_h(ecg_t, tau_s, k, fs)
        h_np = h.numpy()
        p_np = ppg_t.numpy()

        # Linear regression: ppg = a*h + b
        A = np.stack([h_np, np.ones_like(h_np)], axis=1)
        result = np.linalg.lstsq(A, p_np, rcond=None)
        a_hat, b_hat = float(result[0][0]), float(result[0][1])

        if a_hat <= 0:
            continue

        err = float(np.mean((p_np - a_hat*h_np - b_hat)**2))
        if err < best_err:
            best_err = err
            best = (tau_s, a_hat, b_hat)

    return best, best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=200)
    ap.add_argument("--n-estimate", type=int, default=500,
                    help="Windows to use for sigma2 estimation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Device: cpu (empirical estimation)")
    print(f"Max cases: {args.max_cases}  N estimate: {args.n_estimate}\n")

    # Find and split cases
    case_ids = find_paired_cases(max_cases=args.max_cases)
    train_ids, val_ids, test_ids = split_case_ids(case_ids, seed=args.seed)

    # Compute normalization stats
    ecg_stats, ppg_stats = compute_paired_stats(train_ids)

    # Save test IDs and stats
    np.save(str(CKPT_DIR / "vitaldb_test_ids.npy"),
            np.array(test_ids))
    np.save(str(CKPT_DIR / "vitaldb_stats.npy"),
            np.array([ecg_stats, ppg_stats]))
    print(f"Saved vitaldb_test_ids.npy ({len(test_ids)} test cases)")
    print(f"Saved vitaldb_stats.npy")

    # Build dataset
    print(f"\nBuilding training dataset...")
    train_ds = VitalDBPairedDataset(train_ids, ecg_stats, ppg_stats,
                                    verbose=True)
    print(f"Train windows: {len(train_ds)}")

    # Estimate log_sigma2 empirically
    print(f"\nEstimating log_sigma2 from {args.n_estimate} windows...")
    rng = np.random.RandomState(args.seed)
    n_est = min(args.n_estimate, len(train_ds))
    indices = rng.choice(len(train_ds), n_est, replace=False)

    log_sigma2_vals = []
    tau_vals, a_vals, b_vals = [], [], []

    for i, idx in enumerate(indices):
        ecg_t, ppg_t = train_ds[idx]
        ecg_t = ecg_t.squeeze()
        ppg_t = ppg_t.squeeze()

        (tau_s, a_hat, b_hat), res_var = fit_z_least_squares(ecg_t, ppg_t)

        if 1e-6 < res_var < 100:
            log_sigma2_vals.append(math.log(res_var))
            tau_vals.append(tau_s / FS)
            a_vals.append(a_hat)
            b_vals.append(b_hat)

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_est}  "
                  f"mean log_sigma2={np.mean(log_sigma2_vals):.3f}  "
                  f"mean tau={np.mean(tau_vals):.3f}s  "
                  f"mean a={np.mean(a_vals):.3f}")

    log_sigma2_arr = np.array(log_sigma2_vals)
    mu_emp  = float(np.mean(log_sigma2_arr))
    std_emp = float(np.std(log_sigma2_arr))

    print(f"\nEmpirical estimates from {len(log_sigma2_vals)} windows:")
    print(f"  log_sigma2: mean={mu_emp:.4f}  std={std_emp:.4f}")
    print(f"  noise sigma: {math.exp(mu_emp/2):.4f}")
    print(f"  tau:  mean={np.mean(tau_vals):.3f}  std={np.std(tau_vals):.3f} s")
    print(f"  a:    mean={np.mean(a_vals):.3f}  std={np.std(a_vals):.3f}")
    print(f"  b:    mean={np.mean(b_vals):.3f}  std={np.std(b_vals):.3f}")

    # Build q_lambda* from empirical distribution
    q_lambda_star = MeanFieldGaussian(phi_dim=1)
    q_lambda_star.mu_lambda.data.fill_(mu_emp)
    # log_sigma2_lambda stores log(variance of q), i.e. log(std_emp^2)
    q_lambda_star.log_sigma2_lambda.data.fill_(math.log(std_emp**2 + 1e-8))
    q_lambda_star.eval()
    for p in q_lambda_star.parameters():
        p.requires_grad_(False)

    # Save
    out_path = CKPT_DIR / "lambda_star_vitaldb.pt"
    save_lambda(q_lambda_star, out_path)

    # Also save Z statistics for MCMC initialization
    z_stats = {
        "tau_mean": float(np.mean(tau_vals)),
        "tau_std":  float(np.std(tau_vals)),
        "a_mean":   float(np.mean(a_vals)),
        "a_std":    float(np.std(a_vals)),
        "b_mean":   float(np.mean(b_vals)),
        "b_std":    float(np.std(b_vals)),
    }
    np.save(str(CKPT_DIR / "vitaldb_z_stats.npy"), z_stats)
    print(f"\nSaved vitaldb_z_stats.npy")

    print(f"\nFinal q_lambda* (empirically fitted):")
    print(f"  mu_lambda    = {mu_emp:.4f}")
    print(f"  sigma_lambda = {std_emp:.4f}")
    print(f"  log_sigma2 ~ N({mu_emp:.3f}, {std_emp:.3f}²)")
    print(f"  noise sigma ~ {math.exp(mu_emp/2):.4f}")
    print(f"\n→ All outputs saved to {CKPT_DIR}/")


if __name__ == "__main__":
    main()
