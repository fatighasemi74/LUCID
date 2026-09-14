"""
experiments/part1/train_vi.py
==============================
Phase 1 Step 2: learn q_{λ*}(φ) from paired VitalDB data via offline VI.

What this does
--------------
Trains a mean-field Gaussian q_λ(φ) by maximising the ELBO on paired
(ECG, PPG) windows from VitalDB. φ = {τ, a, b, log_σ²} — the four
likelihood parameters of the known forward operator H_φ.

After training, λ* is frozen and saved to:
    experiments/part1/checkpoints/lambda_star.pt

λ* is used in Phase 2 as:
    1. The MH proposal kernel for φ updates
    2. The initialization for φ at the start of each inner MCMC chain

Usage
-----
python experiments/part1/train_vi.py

# Quick smoke test (5 cases, 10 epochs)
python experiments/part1/train_vi.py --smoke-test

# Custom epochs
python experiments/part1/train_vi.py --epochs 100

What φ represents (Part 1 minimal set)
---------------------------------------
Index 0 — τ    : pulse transit delay (seconds). Prior: N(0.2, 0.1²)
Index 1 — a    : amplitude scaling. Prior: N(0.8, 0.2²)
Index 2 — b    : DC offset. Prior: N(0.05, 0.1²)
Index 3 — log_σ²: log noise variance. Prior: N(-3, 1²)

phi_to_params_fn maps a φ vector to likelihood parameter updates.
This function is the SINGLE place where the φ layout is defined —
used here and in run.py and sampler.py.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.vitaldb import (
    find_paired_cases, split_case_ids,
    VitalDBPairedDataset, compute_paired_stats,
)
from models.likelihood import GaussianLikelihood
from models.noise_cov import CovMode
from inference.vi import MeanFieldGaussian, train_vi, save_lambda

# ======================================================================
# Configuration
# ======================================================================

OUT_DIR    = ROOT / "experiments" / "part1" / "checkpoints"
LAMBDA_OUT = OUT_DIR / "lambda_star.pt"

# φ layout (Part 1): [τ, a, b, log_σ²]
PHI_DIM = 4

# Prior over φ — mean and log-variance (for KL term in ELBO)
# These reflect our Part 1 ground truth values with some uncertainty
PHI_PRIOR_MU = torch.tensor([
    0.2,    # τ   ~ N(0.2, 0.1²)
    0.8,    # a   ~ N(0.8, 0.2²)
    0.05,   # b   ~ N(0.05, 0.1²)
    -3.0,   # log_σ² ~ N(-3, 1²)
])
PHI_PRIOR_LOG_SIGMA2 = torch.tensor([
    2 * torch.tensor(0.1).log(),    # log(0.1²)
    2 * torch.tensor(0.2).log(),    # log(0.2²)
    2 * torch.tensor(0.1).log(),    # log(0.1²)
    2 * torch.tensor(1.0).log(),    # log(1²)
])

# VI training
N_EPOCHS      = 50
LR            = 1e-3
N_PHI_SAMPLES = 8
BETA_KL       = 1.0
BATCH_SIZE    = 16

# VitalDB
MAX_CASES_TRAIN = None   # None = all available
MAX_CASES_VAL   = None
SEED = 42


# ======================================================================
# φ → likelihood parameter mapping
# ======================================================================

def phi_to_params_fn(phi: torch.Tensor) -> dict:
    """
    Map a φ vector (shape (4,)) to likelihood parameter updates.

    This is the canonical φ layout for Part 1:
        phi[0] = τ        (delay in seconds)
        phi[1] = a        (amplitude, stored as log_a = log(|a|))
        phi[2] = b        (offset)
        phi[3] = log_σ²   (log noise variance, broadcast to all dims)

    Returns dict compatible with _set_likelihood_params() in mala.py and vi.py.

    NOTE: This function is imported by run.py and sampler.py.
    It is the single source of truth for the φ layout.
    """
    tau    = phi[0]
    log_a  = phi[1].abs().log()   # a must be positive; store as log_a
    b      = phi[2]
    log_s2 = phi[3]               # log σ² — scalar, broadcast to obs_dim

    return {
        "log_a"   : log_a,
        "b"       : b,
        "tau"     : tau,
        "log_diag": log_s2.expand(1),   # NoiseCov expects (n_params,)
    }


# ======================================================================
# Functional (differentiable) ELBO for VI
# ======================================================================
# The standard vi.py elbo() uses .data assignment inside the likelihood
# module, which breaks the autograd graph. Gradients cannot flow back
# to q_lambda. This functional version computes log L directly from
# phi as a differentiable tensor, so reparametrisation gradients work.

def _gaussian_kernel_fn(sigma: float, size: int, device: torch.device) -> torch.Tensor:
    half = size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    k = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return k / k.sum()


def log_likelihood_functional(
    y   : torch.Tensor,
    x   : torch.Tensor,
    phi : torch.Tensor,
    obs_dim     : int   = 4000,
    kernel_sigma: float = 10.0,
    kernel_size : int   = 61,
) -> torch.Tensor:
    """
    Differentiable log N(y ; H_phi(x), sigma^2 I).
    phi: [tau, a, b, log_sigma^2]
    Gradient flows through phi[1] (a), phi[2] (b), phi[3] (log_sigma^2).
    tau uses torch.roll which is non-differentiable — tau is estimated
    via the prior mean and updated by MH in Phase 2.
    """
    import torch.nn.functional as F_nn
    a      = phi[1].abs()
    b      = phi[2]
    log_s2 = phi[3]

    kernel = _gaussian_kernel_fn(kernel_sigma, kernel_size, x.device).view(1, 1, -1)
    h      = F_nn.conv1d(x, kernel, padding=kernel_size // 2)
    # Apply delay tau (phi[0]) — must match generate.py convention
    tau_samples = int(round(float(phi[0]) * 500.0))
    if tau_samples > 0:
        h = torch.roll(h, shifts=tau_samples, dims=-1)
        h[..., :tau_samples] = 0.0  # causal: zero leading samples
    h      = a * h + b

    resid    = (y - h).view(x.shape[0], -1)
    mah      = (resid.pow(2) / log_s2.exp()).sum(-1)
    log_norm = -0.5 * (obs_dim * math.log(2 * math.pi) + obs_dim * log_s2)
    return log_norm - 0.5 * mah   # (B,)


def elbo_functional(
    q_lambda     : MeanFieldGaussian,
    y            : torch.Tensor,
    x            : torch.Tensor,
    n_phi_samples: int = 8,
    beta_kl      : float = 1.0,
    prior_mu     : torch.Tensor | None = None,
    prior_log_s2 : torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ELBO with correct gradient flow through q_lambda.
    Uses reparametrisation: phi_s = mu_lambda + sigma_lambda * eps.
    """
    phi_samples = q_lambda.sample(n_phi_samples)   # (S, 4) — differentiable

    log_liks = []
    for s in range(n_phi_samples):
        ll_s = log_likelihood_functional(y, x.detach(), phi_samples[s])
        log_liks.append(ll_s)

    log_liks_stack = torch.stack(log_liks, dim=0)  # (S, B)
    log_lik_mean   = (
        torch.logsumexp(log_liks_stack, dim=0) - math.log(n_phi_samples)
    ).mean()

    kl = q_lambda.kl_to_prior(prior_mu, prior_log_s2)
    return log_lik_mean - beta_kl * kl


# ======================================================================
# Build VitalDB paired DataLoader
# ======================================================================

def build_vitaldb_loaders(
    max_cases  : int | None = None,
    batch_size : int = BATCH_SIZE,
    smoke_test : bool = False,
) -> tuple[DataLoader, DataLoader, tuple, tuple]:
    """
    Discover VitalDB cases, split, compute stats, build loaders.

    Returns
    -------
    train_loader, val_loader, ecg_stats, ppg_stats
    """
    n_cases = 20 if smoke_test else max_cases
    cases   = find_paired_cases(max_cases=n_cases)
    train_ids, val_ids, _ = split_case_ids(cases, seed=SEED)

    if smoke_test:
        train_ids = train_ids[:5]
        val_ids   = val_ids[:2]

    # Compute normalization stats from training cases
    ecg_stats, ppg_stats = compute_paired_stats(train_ids,
                                                max_cases=len(train_ids))

    print(f"[VI data] train={len(train_ids)} cases  val={len(val_ids)} cases")

    train_ds = VitalDBPairedDataset(
        train_ids, ecg_stats, ppg_stats, verbose=True
    )
    val_ds = VitalDBPairedDataset(
        val_ids, ecg_stats, ppg_stats, verbose=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, ecg_stats, ppg_stats


# ======================================================================
# Validation ELBO
# ======================================================================

def eval_elbo(
    q          : MeanFieldGaussian,
    likelihood : GaussianLikelihood,
    loader     : DataLoader,
    device     : torch.device,
    n_samples  : int = 4,
) -> float:
    """Compute mean ELBO on the validation set using functional likelihood."""
    q.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_b, y_b = batch[0].to(device), batch[1].to(device)
            val = elbo_functional(
                q, y_b, x_b,
                n_phi_samples=n_samples,
                beta_kl=BETA_KL,
                prior_mu=PHI_PRIOR_MU.to(device),
                prior_log_s2=PHI_PRIOR_LOG_SIGMA2.to(device),
            )
            total += float(val)
            n += 1
    return total / max(1, n)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 Step 2: VI training on VitalDB paired data"
    )
    ap.add_argument("--epochs",     type=int,  default=N_EPOCHS)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--batch-size", type=int,  default=BATCH_SIZE)
    ap.add_argument("--smoke-test", action="store_true",
                    help="5 cases, 10 epochs — quick pipeline check")
    args = ap.parse_args()

    smoke     = args.smoke_test
    n_epochs  = 10 if smoke else args.epochs
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Pulse2Posterior — VI Training (Phase 1 Step 2)")
    print("=" * 60)
    print(f"Device : {device}")
    print(f"φ dim  : {PHI_DIM}  layout: [τ, a, b, log_σ²]")
    print(f"Epochs : {n_epochs}  LR: {args.lr}  Batch: {args.batch_size}")
    if smoke:
        print("[SMOKE TEST]")
    print()

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, ecg_stats, ppg_stats = build_vitaldb_loaders(
        batch_size=args.batch_size, smoke_test=smoke
    )

    # ── Likelihood module ─────────────────────────────────────────────
    # obs_dim = window length (4000 at 500Hz)
    # kernel_sigma and kernel_size match generate.py PHI_TRUE
    obs_dim    = 4000
    likelihood = GaussianLikelihood(
        obs_dim=obs_dim,
        fs=500.0,
        cov_mode=CovMode.ISOTROPIC,
        kernel_sigma=10.0,
        kernel_size=61,
    ).to(device)

    # ── Variational distribution q_λ(φ) ──────────────────────────────
    # Initialize μ_λ at the Part 1 true values (good starting point)
    init_mu = PHI_PRIOR_MU.clone()
    q = MeanFieldGaussian(
        phi_dim=PHI_DIM,
        init_mu=init_mu,
    ).to(device)

    print(f"[VI] initial μ_λ = {q.mu_lambda.data.tolist()}")
    print(f"[VI] initial σ_λ = {q.sigma.data.tolist()}")
    print()

    # ── Train ─────────────────────────────────────────────────────────
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    best_val_elbo = float("-inf")
    best_mu       = q.mu_lambda.data.clone()
    best_ls2      = q.log_sigma2_lambda.data.clone()

    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        q.train()
        total, n = 0.0, 0

        for batch in train_loader:
            x_b = batch[0].to(device)   # (B, 1, L) ECG
            y_b = batch[1].to(device)   # (B, 1, L) PPG

            opt.zero_grad(set_to_none=True)
            loss = -elbo_functional(
                q, y_b, x_b,
                n_phi_samples=N_PHI_SAMPLES,
                beta_kl=BETA_KL,
                prior_mu=PHI_PRIOR_MU.to(device),
                prior_log_s2=PHI_PRIOR_LOG_SIGMA2.to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step()
            total += float(-loss.detach())
            n     += 1

        train_elbo = total / max(1, n)
        elapsed    = time.time() - t0

        if ep % 5 == 0 or ep == 1 or ep == n_epochs:
            val_elbo = eval_elbo(q, likelihood, val_loader, device)
            improved = val_elbo > best_val_elbo
            if improved:
                best_val_elbo = val_elbo
                best_mu  = q.mu_lambda.data.clone()
                best_ls2 = q.log_sigma2_lambda.data.clone()
            marker = " ← best" if improved else ""
            print(f"epoch {ep:03d}/{n_epochs}  "
                  f"train_ELBO={train_elbo:.4f}  "
                  f"val_ELBO={val_elbo:.4f}  "
                  f"{elapsed:.1f}s{marker}")
            print(f"  μ_λ = [τ={q.mu_lambda[0].item():.4f}  "
                  f"a={q.mu_lambda[1].item():.4f}  "
                  f"b={q.mu_lambda[2].item():.4f}  "
                  f"log_σ²={q.mu_lambda[3].item():.4f}]")
        else:
            print(f"epoch {ep:03d}/{n_epochs}  "
                  f"train_ELBO={train_elbo:.4f}  {elapsed:.1f}s")

    # ── Restore best and freeze ───────────────────────────────────────
    q.mu_lambda.data.copy_(best_mu)
    q.log_sigma2_lambda.data.copy_(best_ls2)
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)

    # ── Save λ* ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_lambda(q, LAMBDA_OUT)

    print()
    print("=" * 60)
    print("VI training complete. λ* frozen and saved.")
    print("=" * 60)
    print(f"  λ* path : {LAMBDA_OUT}")
    print(f"  μ_λ*    : {q.mu_lambda.data.tolist()}")
    print(f"  σ_λ*    : {q.sigma.data.tolist()}")
    print()
    print("Interpretation:")
    print(f"  τ   ~ N({q.mu_lambda[0].item():.3f}, "
          f"{q.sigma[0].item():.3f}²)  [delay in seconds]")
    print(f"  a   ~ N({q.mu_lambda[1].item():.3f}, "
          f"{q.sigma[1].item():.3f}²)  [amplitude]")
    print(f"  b   ~ N({q.mu_lambda[2].item():.3f}, "
          f"{q.sigma[2].item():.3f}²)  [offset]")
    print(f"  σ²  ~ N({q.mu_lambda[3].item():.3f}, "
          f"{q.sigma[3].item():.3f}²)  [log noise var]")
    print()
    print("Next step: python experiments/part1/run.py")


if __name__ == "__main__":
    main()
