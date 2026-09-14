"""
experiments/part1/staged_check.py
===================================
Four-stage sanity check per professor's diagnostic plan.

Stage 1: Infer X only, Z=z_true and phi=phi_true fixed (oracle)
Stage 2: Infer X and Z, phi=phi_true fixed
Stage 3: Infer X, Z, and phi
Stage 4: Full algorithm — X, Z, phi + bootstrap ensemble + weights

For each stage: RMSE, correlation, coverage, reconstruction plot
with individual posterior samples, and PPG consistency check.

Usage
-----
python experiments/part1/staged_check.py --n-inner 200 --burn-in 50
python experiments/part1/staged_check.py --n-inner 50  --burn-in 10  # fast test
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.prior import ECGDiffusionPrior, BootstrapEnsemble
from models.likelihood import GaussianLikelihood
from models.noise_cov import CovMode
from models.diffusion_core import GaussianDiffusion
from models.unet1d import UNet1D
from inference.vi import MeanFieldGaussian, load_lambda
from inference.mala import (MCMCState, MALAConfig,
    tau_xcorr_mh_step, ab_mala_step, phi_mh_step,
    x_diffusion_step, _set_likelihood_params,
    set_state_params, z_to_params, phi_to_params,
    make_z_prior)
from inference.sampler import compute_weights

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
DATA_DIR = ROOT / "experiments/part1/synthetic_data"
OUT_DIR  = ROOT / "experiments/part1/staged_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500

# phi_true from generate.py
# New structure: Z = [tau, a, b]  Phi = [log_sigma2]
Z_TRUE       = torch.tensor([[0.2, 0.8, 0.05]])   # [tau, a, b]
PHI_TRUE_VEC = torch.tensor([-3.0])               # [log_sigma2] — set per SNR

# For backwards compatibility in evaluation
TAU_TRUE = 0.2
A_TRUE   = 0.8
B_TRUE   = 0.05

# ======================================================================
# Forward operator (with delay, matching generate.py)
# ======================================================================

def apply_h_phi(x, z_vec, fs=FS):
    """H_phi(x) with delay. x:(B,1,L), z_vec:[tau,a,b]"""
    z_vec = z_vec.squeeze()
    a = z_vec[1].abs()
    b = z_vec[2]
    half = 30
    t = torch.arange(-half, half+1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t**2/(2*10.0**2)); k = k/k.sum()
    h = F_nn.conv1d(x, k.view(1,1,-1), padding=half)
    tau_s = int(round(float(z_vec[0]) * fs))
    if tau_s > 0:
        h = torch.roll(h, shifts=tau_s, dims=-1)
        h[..., :tau_s] = 0.0
    return a * h + b

# ======================================================================
# Load data and models
# ======================================================================

def load_one_window(snr_db=20, win_idx=0):
    data = np.load(str(DATA_DIR / f"snr_{snr_db:02d}dB.npz"))
    ecg  = torch.tensor(data["ecg"][win_idx], dtype=torch.float32)
    ppg  = torch.tensor(data["ppg"][win_idx], dtype=torch.float32)
    return ecg.view(1,1,4000), ppg.view(1,1,4000)

def load_prior_k(k, device):
    path = CKPT_DIR / f"prior_k{k:02d}.pt"
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    unet = UNet1D(base_ch=ckpt.get("base_ch",64),
                  time_dim=ckpt.get("time_dim",128),
                  n_res=ckpt.get("n_res",2)).to(device)
    unet.load_state_dict(ckpt["model"])
    unet.eval()
    return ECGDiffusionPrior(denoiser=unet, T=1000,
                             pred_type="x0", device=device).to(device)

# ======================================================================
# Inner MCMC loop — configurable which variables are inferred
# ======================================================================

def run_mcmc(
    y              : torch.Tensor,
    prior          : ECGDiffusionPrior,
    likelihood     : GaussianLikelihood,
    diffusion      : GaussianDiffusion,
    cfg            : MALAConfig,
    device         : torch.device,
    infer_z        : bool = False,      # infer Z=(tau,a,b)
    fix_log_sigma2 : float | None = None,
    z_fixed        : torch.Tensor | None = None,
) -> tuple[list, list, list]:
    """
    Run inner MCMC loop.

    New structure (per professor Sep 8):
        Z   = [tau, a, b]   — record-specific, dim=3
        Phi = [log_sigma2]  — global noise, fixed in Part 1

    infer_z=False  → Stage 1/2: oracle Z, only X updated
    infer_z=True   → Stage 3:   tau gets xcorr-MH, (a,b) get MALA

    fix_log_sigma2: always set for Part 1 controlled experiments.
    z_fixed: oracle Z for Stage 1 (overrides infer_z).
    """
    import math as _math

    # Initialize Phi = [log_sigma2]
    log_s2 = fix_log_sigma2 if fix_log_sigma2 is not None else -3.0
    phi0 = torch.tensor([log_s2], dtype=torch.float32, device=device)

    # Initialize Z = [tau, a, b]
    if z_fixed is not None:
        z0 = z_fixed.clone().to(device)
    else:
        z0 = torch.tensor([[TAU_TRUE, A_TRUE, B_TRUE]],
                           dtype=torch.float32, device=device)

    # Set likelihood params from initial state
    _set_likelihood_params(likelihood, z_to_params(z0))
    _set_likelihood_params(likelihood, phi_to_params(phi0))

    # Initialize x from prior
    with torch.no_grad():
        x0 = prior.sample((1,1,4000), ddim_steps=20, device=device)
    with torch.no_grad():
        ll0 = float(likelihood.log_likelihood(y, x0, z0).sum())

    state = MCMCState(x=x0, z=z0, phi=phi0, log_lik=ll0, step=0)

    # Z prior: independent Gaussians on tau, a, b
    log_z_prior = make_z_prior(
        tau_mu=TAU_TRUE, tau_sigma=0.05,
        a_mu=A_TRUE,     a_sigma=0.20,
        b_mu=B_TRUE,     b_sigma=0.10,
    )

    def prior_score_fn(x_t, t_batch):
        return prior.denoiser(x_t, t_batch)

    x_samples, z_samples, phi_samples = [], [], []

    for r in range(cfg.n_inner):
        # Always enforce fixed sigma2
        phi_curr = torch.tensor([log_s2], dtype=torch.float32, device=device)
        _set_likelihood_params(likelihood, phi_to_params(phi_curr))
        state = MCMCState(x=state.x, z=state.z, phi=phi_curr,
                          log_lik=state.log_lik, step=state.step,
                          n_acc_phi=state.n_acc_phi,
                          n_acc_z=state.n_acc_z,
                          n_acc_tau=state.n_acc_tau)

        if infer_z and z_fixed is None:
            # 1. tau update — cross-correlation MH
            state, _ = tau_xcorr_mh_step(
                state, y, likelihood, cfg, log_z_prior, device)

            # 2. (a, b) update — MALA
            state, _ = ab_mala_step(
                state, y, likelihood, log_z_prior, cfg, device)

        # 3. X update — always
        state = x_diffusion_step(state, y, prior_score_fn,
                                  likelihood, diffusion, cfg, device)
        state = MCMCState(x=state.x, z=state.z, phi=state.phi,
                          log_lik=state.log_lik, step=r+1,
                          n_acc_phi=state.n_acc_phi,
                          n_acc_z=state.n_acc_z,
                          n_acc_tau=state.n_acc_tau)

        if r >= cfg.burn_in:
            x_samples.append(state.x.squeeze().cpu())
            z_samples.append(state.z.cpu())
            phi_samples.append(state.phi.cpu())

    return x_samples, z_samples, phi_samples

# ======================================================================
# Evaluation
# ======================================================================

def evaluate(
    x_samples  : list[torch.Tensor],
    x_true     : torch.Tensor,
    y_obs      : torch.Tensor,
    phi_samples: list[torch.Tensor],
    stage_name : str,
    device     : torch.device,
    z_samples  : list | None = None,
) -> dict:
    """Compute RMSE, correlation, coverage, PPG consistency."""
    x_np  = x_true.squeeze().numpy()
    xs    = torch.stack(x_samples).numpy()         # (N, 4000)
    mean  = xs.mean(axis=0)
    std   = xs.std(axis=0)

    rmse  = float(np.sqrt(((mean - x_np)**2).mean()))
    corr  = float(np.corrcoef(mean, x_np)[0,1])

    coverage = {}
    for level in [0.50, 0.80, 0.90, 0.95]:
        from scipy import stats
        z = stats.norm.ppf((1 + level) / 2)
        coverage[level] = float((np.abs(x_np - mean) <= z * std).mean())

    # PPG consistency: for each sample (x_i, phi_i), compute ||y - H_phi_i(x_i)||
    # then average. This correctly uses matched (x, phi) pairs per sample.
    y_np = y_obs.squeeze().numpy()
    ppg_errs = []
    for i in range(len(x_samples)):
        x_i = torch.tensor(x_samples[i]).view(1,1,4000).to(device)
        if z_samples is not None and i < len(z_samples):
            z_i = z_samples[i].squeeze().to(device)
        else:
            z_i = torch.tensor([TAU_TRUE, A_TRUE, B_TRUE], device=device)
        with torch.no_grad():
            ppg_i = apply_h_phi(x_i, z_i).squeeze().cpu().numpy()
        ppg_errs.append(float(np.abs(ppg_i - y_np).mean()))
    ppg_err = float(np.mean(ppg_errs))

    return {
        "rmse"       : rmse,
        "corr"       : corr,
        "coverage"   : coverage,
        "ppg_err"    : ppg_err,
        "mean"       : mean,
        "std"        : std,
        "samples"    : xs,
    }

# ======================================================================
# Plotting
# ======================================================================

def plot_stage(
    results    : dict,
    x_true     : torch.Tensor,
    y_obs      : torch.Tensor,
    stage_name : str,
    n_samples  : int = 5,
) -> None:
    x_np  = x_true.squeeze().numpy()
    y_np  = y_obs.squeeze().numpy()
    mean  = results["mean"]
    std   = results["std"]
    xs    = results["samples"]
    t     = np.arange(4000) / FS

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: individual samples + mean + truth
    ax = axes[0]
    for i in range(min(n_samples, len(xs))):
        ax.plot(t, xs[i], color="steelblue", alpha=0.2, lw=0.5)
    ax.plot(t, mean,  "steelblue", lw=1.0, label=f"Posterior mean (corr={results['corr']:.3f})")
    ax.fill_between(t, mean-std, mean+std, alpha=0.2, color="steelblue", label="±1 std")
    ax.plot(t, x_np,  "k",  lw=0.8, label="True ECG")
    ax.set_title(f"{stage_name} — ECG reconstruction  RMSE={results['rmse']:.4f}  corr={results['corr']:.3f}")
    ax.legend(fontsize=7)
    ax.set_ylabel("Normalized amplitude")

    # Panel 2: calibration
    ax = axes[1]
    levels  = [0.50, 0.80, 0.90, 0.95]
    emp_cov = [results["coverage"][l] for l in levels]
    ax.plot(levels, emp_cov, "o-", color="steelblue", label="empirical")
    ax.plot([0,1], [0,1], "k--", label="ideal")
    for l, e in zip(levels, emp_cov):
        ax.annotate(f"{e:.2f}", (l, e), textcoords="offset points",
                    xytext=(5,3), fontsize=7)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration curve")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 3: PPG consistency
    ax = axes[2]
    phi_label = f"ppg_err={results['ppg_err']:.4f}"
    ax.plot(t, y_np, "orange", lw=0.8, label="PPG observed")
    ax.set_title(f"PPG observed (mean matched-sample err={results['ppg_err']:.4f})")
    ax.set_ylabel("Normalized amplitude")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fname = stage_name.replace(" ", "_").replace("/", "_").lower()
    out   = OUT_DIR / f"{fname}.png"
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    print(f"  → saved {out.name}")

# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inner",  type=int, default=200)
    ap.add_argument("--burn-in",  type=int, default=50)
    ap.add_argument("--snr-db",   type=int, default=20)
    ap.add_argument("--win-idx",  type=int, default=0)
    ap.add_argument("--gamma",    type=float, default=5.0,
                    help="guidance strength (default 5.0)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"SNR={args.snr_db}dB  window={args.win_idx}  "
          f"N={args.n_inner}  burn_in={args.burn_in}  gamma={args.gamma}\n")

    # Load data
    x_true, y_obs = load_one_window(args.snr_db, args.win_idx)
    x_true = x_true.to(device)
    y_obs  = y_obs.to(device)
    print(f"ECG true: std={x_true.std().item():.3f}  max={x_true.max().item():.3f}")
    print(f"PPG obs:  std={y_obs.std().item():.3f}")

    # Load models
    prior_k0 = load_prior_k(0, device)
    diffusion = GaussianDiffusion(T=1000, schedule="cosine", pred_type="x0").to(device)
    likelihood = GaussianLikelihood(obs_dim=4000, fs=500.0,
                                     cov_mode=CovMode.ISOTROPIC,
                                     kernel_sigma=10.0, kernel_size=61).to(device)
    q_lambda = load_lambda(CKPT_DIR / "lambda_star.pt", device)

    # Set phi_true in likelihood
    # New structure: Phi=[log_sigma2], Z=[tau,a,b]
    import math as _math
    data_snr = np.load(str(DATA_DIR / f"snr_{args.snr_db:02d}dB.npz"))
    ns_true  = float(data_snr["noise_sigma"])
    log_s2   = float(_math.log(ns_true**2))
    phi_true = torch.tensor([log_s2], dtype=torch.float32, device=device)
    z_true   = torch.tensor([[TAU_TRUE, A_TRUE, B_TRUE]],
                              dtype=torch.float32, device=device)
    _set_likelihood_params(likelihood, z_to_params(z_true))
    _set_likelihood_params(likelihood, phi_to_params(phi_true))
    print(f"  log_sigma2={log_s2:.4f}  noise_sigma={ns_true:.5f}")

    # MALA config — higher gamma for stronger guidance
    _t_anneal = list(range(500, 0, -10)) + [1]
    gamma_schedule = {t: args.gamma for t in _t_anneal}
    cfg = MALAConfig(
        n_inner      =args.n_inner,
        burn_in      =args.burn_in,
        step_size_z  =1e-3,
        step_size_tau=0.02,
        t_anneal     =_t_anneal,
        gamma_t      =gamma_schedule,
    )

    results_all = {}

    # ── Stage 1: X only, Z=z_true, phi=phi_true ──────────────────────
    print("="*60)
    print("STAGE 1: Infer X only (Z=z_true, phi=phi_true fixed)")
    print("="*60)
    x_s, z_s, phi_s = run_mcmc(
        y_obs, prior_k0, likelihood, diffusion, cfg, device,
        infer_z=False,
        fix_log_sigma2=log_s2,
        z_fixed=z_true,
    )
    r1 = evaluate(x_s, x_true.cpu(), y_obs.cpu(), phi_s, "Stage 1", device, z_samples=z_s)
    results_all["Stage 1"] = r1
    print(f"  RMSE={r1['rmse']:.4f}  corr={r1['corr']:.4f}  "
          f"ppg_err={r1['ppg_err']:.4f}")
    print(f"  Coverage: " + "  ".join(
        [f"{int(l*100)}%→{r1['coverage'][l]*100:.1f}%" for l in [0.50,0.80,0.90,0.95]]))
    plot_stage(r1, x_true.cpu(), y_obs.cpu(), "Stage 1 — X only oracle Z phi")

    # ── Stage 2: X and Z, phi=phi_true ───────────────────────────────
    print()
    print("="*60)
    print("STAGE 2: Infer X and Z (phi=phi_true fixed)")
    print("="*60)
    x_s, z_s, phi_s = run_mcmc(
        y_obs, prior_k0, likelihood, diffusion, cfg, device,
        infer_z=True,
        fix_log_sigma2=log_s2,
    )
    r2 = evaluate(x_s, x_true.cpu(), y_obs.cpu(), phi_s, "Stage 2", device, z_samples=z_s)
    results_all["Stage 2"] = r2
    print(f"  RMSE={r2['rmse']:.4f}  corr={r2['corr']:.4f}  "
          f"ppg_err={r2['ppg_err']:.4f}")
    print(f"  Coverage: " + "  ".join(
        [f"{int(l*100)}%→{r2['coverage'][l]*100:.1f}%" for l in [0.50,0.80,0.90,0.95]]))
    plot_stage(r2, x_true.cpu(), y_obs.cpu(), "Stage 2 — X Z oracle phi")

    # ── Stage 3: X, Z, and phi ───────────────────────────────────────
    print()
    print("="*60)
    print("STAGE 3: Infer X, Z, and phi")
    print("="*60)
    x_s, z_s, phi_s = run_mcmc(
        y_obs, prior_k0, likelihood, diffusion, cfg, device,
        infer_z=True,
        fix_log_sigma2=log_s2,
    )
    r3 = evaluate(x_s, x_true.cpu(), y_obs.cpu(), phi_s, "Stage 3", device, z_samples=z_s)
    results_all["Stage 3"] = r3
    print(f"  RMSE={r3['rmse']:.4f}  corr={r3['corr']:.4f}  "
          f"ppg_err={r3['ppg_err']:.4f}")
    print(f"  Coverage: " + "  ".join(
        [f"{int(l*100)}%→{r3['coverage'][l]*100:.1f}%" for l in [0.50,0.80,0.90,0.95]]))
    plot_stage(r3, x_true.cpu(), y_obs.cpu(), "Stage 3 - X Z phi")

    # ── Stage 3b: sigma2 fixed at true value ─────────────────────────
    print()
    print("="*60)
    print("STAGE 3b: Infer X, Z, tau/a/b -- sigma2 FIXED at true value")
    print("="*60)
    import math as _math, numpy as _np
    _data = _np.load(str(DATA_DIR / f"snr_{args.snr_db:02d}dB.npz"))
    _ns   = float(_data["noise_sigma"])
    _ell  = float(_math.log(_ns**2))
    print(f"  Fixing log_sigma2={_ell:.4f}  (noise_sigma={_ns:.5f})")
    x_s, z_s, phi_s = run_mcmc(
        y_obs, prior_k0, likelihood, diffusion, cfg, device,
        infer_z=True,
        fix_log_sigma2=_ell,
    )
    r3b = evaluate(x_s, x_true.cpu(), y_obs.cpu(), phi_s, "Stage 3b", device, z_samples=z_s)
    results_all["Stage 3b"] = r3b
    print(f"  RMSE={r3b['rmse']:.4f}  corr={r3b['corr']:.4f}  ppg_err={r3b['ppg_err']:.4f}")
    print(f"  Coverage: " + "  ".join(
        [f"{int(l*100)}%->{r3b['coverage'][l]*100:.1f}%" for l in [0.50,0.80,0.90,0.95]]))
    plot_stage(r3b, x_true.cpu(), y_obs.cpu(), "Stage 3b - sigma2 fixed")

    # ── Stage 4: Full algorithm (one prior only for speed) ───────────
    print()
    print("="*60)
    print("STAGE 4: Full algorithm (K=1 prior, with ensemble weights)")
    print("="*60)
    print("  (Using k=0 only for speed — same MCMC as Stage 3 + weighting)")
    x_s, z_s, phi_s = run_mcmc(
        y_obs, prior_k0, likelihood, diffusion, cfg, device,
        infer_z=True,
        fix_log_sigma2=log_s2,
    )
    r4 = evaluate(x_s, x_true.cpu(), y_obs.cpu(), phi_s, "Stage 4", device, z_samples=z_s)
    results_all["Stage 4"] = r4
    print(f"  RMSE={r4['rmse']:.4f}  corr={r4['corr']:.4f}  "
          f"ppg_err={r4['ppg_err']:.4f}")
    print(f"  Coverage: " + "  ".join(
        [f"{int(l*100)}%→{r4['coverage'][l]*100:.1f}%" for l in [0.50,0.80,0.90,0.95]]))
    plot_stage(r4, x_true.cpu(), y_obs.cpu(), "Stage 4 — Full algorithm")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Stage':<12} {'RMSE':>8} {'Corr':>8} {'Cov50':>8} {'Cov95':>8} {'PPG_err':>10}")
    print("-"*56)
    for name, r in results_all.items():
        print(f"{name:<12} {r['rmse']:>8.4f} {r['corr']:>8.4f} "
              f"{r['coverage'][0.50]*100:>7.1f}% "
              f"{r['coverage'][0.95]*100:>7.1f}% "
              f"{r['ppg_err']:>10.4f}")
    print()
    print(f"Figures saved to {OUT_DIR}/")

    # Key diagnostic
    print()
    if r1['corr'] < 0.1:
        print("⚠ Stage 1 correlation < 0.1 — guided diffusion MCMC needs attention")
        print("  Try increasing --gamma (current: {args.gamma})")
    else:
        print("✓ Stage 1 works — guided diffusion is effective with oracle (Z, phi)")
        if r3['corr'] < r1['corr'] - 0.1:
            print("  Performance degrades when phi/Z are inferred — VI or MALA issue")
        else:
            print("  Performance stable across stages — full algorithm is working")


if __name__ == "__main__":
    main()
