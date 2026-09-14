"""
experiments/part1/phi_ablation.py
===================================
Professor's tasks before K=10:

TASK A — phi ablation with sigma2 fixed at true value:
  (1) infer tau only
  (2) infer a only
  (3) infer b only
  (4) infer (tau, a, b)
  Report matched-sample PPG error for each.
  Identify which component drives the forward discrepancy.

TASK B — credible interval check:
  Current coverage at 50% nominal is 82-86% (too wide).
  Check: are intervals equal-tailed, HPD, or pointwise?
  Recompute coverage for scalar clinical quantities g(X):
    - RR interval (heart rate)
    - QRS amplitude (max absolute value)
    - PR segment mean

Usage
-----
python experiments/part1/phi_ablation.py --n-windows 5 --snr-db 20
"""

import sys, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet1d import UNet1D
from models.diffusion_core import GaussianDiffusion
from models.prior import ECGDiffusionPrior
from models.likelihood import GaussianLikelihood
from models.noise_cov import CovMode
from inference.vi import load_lambda
from inference.mala import (MCMCState, MALAConfig,
                             phi_mh_step, z_mala_step,
                             x_diffusion_step, _set_likelihood_params)
from inference.sampler import make_gaussian_z_prior

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
DATA_DIR = ROOT / "experiments/part1/synthetic_data"
OUT_DIR  = ROOT / "experiments/part1/phi_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS       = 500
PHI_TRUE = torch.tensor([0.2, 0.8, 0.05, -3.0])  # tau, a, b, log_sigma2
Z_TRUE   = torch.tensor([[0.2]])


def phi_to_params(phi):
    return {"log_a": phi[1].abs().log(), "b": phi[2],
            "tau": phi[0], "log_diag": phi[3].expand(1)}


def apply_h_phi(x, phi_vec, fs=FS):
    phi_vec = phi_vec.to(x.device)
    a = phi_vec[1].abs(); b = phi_vec[2]
    half = 30
    t = torch.arange(-half, half+1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t**2 / (2*10.0**2)); k = k / k.sum()
    h = F.conv1d(x, k.view(1,1,-1), padding=half)
    tau_s = int(round(float(phi_vec[0]) * fs))
    if tau_s > 0:
        h = torch.roll(h, shifts=tau_s, dims=-1)
        h[..., :tau_s] = 0.0
    return a * h + b


def load_prior(device):
    ckpt = torch.load(str(CKPT_DIR/"prior_k00.pt"),
                      map_location=device, weights_only=False)
    unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
    unet.load_state_dict(ckpt["model"]); unet.eval()
    return ECGDiffusionPrior(denoiser=unet, T=1000, pred_type="x0",
                             device=device).to(device)


def run_mcmc_ablation(
    y, prior, lik, diff, cfg, device, q_lambda,
    infer_mask,        # list[bool]: [infer_tau, infer_a, infer_b]
    fix_log_sigma2,    # float: always fixed at true value
    ns,                # noise_sigma for setting true sigma2
):
    """
    Run MCMC inferring only the phi components indicated by infer_mask.
    log_sigma2 is always fixed at fix_log_sigma2.
    infer_mask = [True, False, False] → infer tau only, keep a=0.8, b=0.05
    """
    # Start phi at true values
    phi_curr = PHI_TRUE.clone().to(device)
    phi_curr[3] = fix_log_sigma2
    z_curr = Z_TRUE.clone().to(device)

    log_z_prior = make_gaussian_z_prior(mu=0.2, sigma=0.1)

    def score_fn(x_t, t_b):
        return prior.denoiser(x_t, t_b)

    _set_likelihood_params(lik, phi_to_params(phi_curr))
    state = MCMCState(x=torch.randn(1,1,4000,device=device),
                      z=z_curr, phi=phi_curr, log_lik=-1e6, step=0)

    x_samples, phi_samples = [], []

    for r in range(cfg.n_inner):
        # Phi MH — only propose components indicated by infer_mask
        # For masked-out components, keep at true values
        old_phi = state.phi.clone()
        state, _ = phi_mh_step(state, y, lik, q_lambda,
                                phi_to_params, device)
        # Override: keep components at true values if not inferred
        new_phi = state.phi.clone()
        for j, (infer_this, true_val) in enumerate(
                zip(infer_mask, [0.2, 0.8, 0.05])):
            if not infer_this:
                new_phi[j] = true_val
        # Always fix log_sigma2
        new_phi[3] = fix_log_sigma2
        _set_likelihood_params(lik, phi_to_params(new_phi))
        state = MCMCState(x=state.x, z=state.z, phi=new_phi,
                          log_lik=state.log_lik, step=r+1,
                          n_acc_phi=state.n_acc_phi,
                          n_acc_z=state.n_acc_z)

        # Z update
        state, _ = z_mala_step(state, y, lik, log_z_prior, cfg, device)

        # X update
        state = x_diffusion_step(state, y, score_fn, lik, diff, cfg, device)
        state = MCMCState(x=state.x, z=state.z, phi=state.phi,
                          log_lik=state.log_lik, step=r+1,
                          n_acc_phi=state.n_acc_phi,
                          n_acc_z=state.n_acc_z)

        if r >= cfg.burn_in:
            x_samples.append(state.x.squeeze().cpu())
            phi_samples.append(state.phi.cpu())

    return x_samples, phi_samples


def matched_ppg_err(x_samples, phi_samples, y_obs_np, device):
    """Mean ||y - H_phi_i(x_i)|| over matched posterior samples."""
    errs = []
    for i in range(len(x_samples)):
        x_i   = x_samples[i].view(1,1,4000).to(device)
        phi_i = phi_samples[i].to(device)
        with torch.no_grad():
            ppg_i = apply_h_phi(x_i, phi_i).squeeze().cpu().numpy()
        errs.append(float(np.abs(ppg_i - y_obs_np).mean()))
    return float(np.mean(errs)), float(np.std(errs))


# ======================================================================
# Clinical scalar quantities g(X)
# ======================================================================

def rr_interval(x_np, fs=FS):
    """Mean RR interval in seconds — estimated from peak spacing."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(x_np, height=0.5*x_np.max(),
                           distance=int(0.4*fs))
    if len(peaks) < 2:
        return float('nan')
    return float(np.mean(np.diff(peaks)) / fs)


def qrs_amplitude(x_np):
    """Max absolute amplitude — proxy for QRS peak height."""
    return float(np.abs(x_np).max())


def pr_mean(x_np, fs=FS):
    """Mean signal value in first 40ms (proxy for PR baseline)."""
    return float(x_np[:int(0.04*fs)].mean())


def compute_scalars(samples_np):
    """Compute g(X) for each posterior sample. Returns (N,) arrays."""
    rr   = np.array([rr_interval(s)    for s in samples_np])
    qrs  = np.array([qrs_amplitude(s)  for s in samples_np])
    pr   = np.array([pr_mean(s)        for s in samples_np])
    return rr, qrs, pr


def scalar_coverage(scalar_samples, scalar_true, level):
    """
    Check if scalar_true is inside the equal-tailed posterior interval
    at the given nominal level.
    Returns 1 if inside, 0 if outside.
    """
    alpha = 1 - level
    lo = np.nanpercentile(scalar_samples, 100 * alpha/2)
    hi = np.nanpercentile(scalar_samples, 100 * (1 - alpha/2))
    return float(lo <= scalar_true <= hi)


def pointwise_coverage_check(samples_np, x_true_np, level):
    """
    Current method: pointwise equal-tailed intervals at each timestep.
    Returns empirical coverage (fraction of timesteps inside interval).
    This is what staged_check currently computes.
    """
    mean = samples_np.mean(axis=0)
    std  = samples_np.std(axis=0)
    z    = stats.norm.ppf((1 + level) / 2)
    return float((np.abs(x_true_np - mean) <= z * std).mean())


def equal_tailed_coverage(samples_np, x_true_np, level):
    """
    Equal-tailed empirical quantile intervals at each timestep.
    No Gaussian assumption — uses actual sample quantiles.
    """
    alpha = 1 - level
    lo = np.percentile(samples_np, 100*alpha/2,   axis=0)
    hi = np.percentile(samples_np, 100*(1-alpha/2), axis=0)
    return float(((x_true_np >= lo) & (x_true_np <= hi)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=5)
    ap.add_argument("--snr-db",    type=int, default=20)
    ap.add_argument("--gamma",     type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  SNR={args.snr_db}dB\n")

    data    = np.load(str(DATA_DIR/f"snr_{args.snr_db:02d}dB.npz"))
    ecg_all = data["ecg"]
    ppg_all = data["ppg"]
    ns      = float(data["noise_sigma"])
    fix_ell = float(math.log(ns**2))

    prior   = load_prior(device)
    diff    = GaussianDiffusion(T=1000, schedule="cosine",
                                 pred_type="x0").to(device)
    lik     = GaussianLikelihood(obs_dim=4000, fs=500.0,
                                  cov_mode=CovMode.ISOTROPIC,
                                  kernel_sigma=10.0,
                                  kernel_size=61).to(device)
    q_lam   = load_lambda(CKPT_DIR/"lambda_star.pt", device)
    gamma_t = {t: args.gamma for t in list(range(500,0,-10))+[1]}
    cfg     = MALAConfig(n_inner=30, burn_in=5, step_size_z=1e-3,
                         t_anneal=list(range(500,0,-10))+[1],
                         gamma_t=gamma_t)

    windows = [(ecg_all[i], ppg_all[i]) for i in range(args.n_windows)]

    # ================================================================
    # TASK A: phi ablation — which component drives PPG discrepancy?
    # ================================================================
    print("="*60)
    print("TASK A: phi ablation (sigma2 fixed at true value)")
    print(f"fix_log_sigma2={fix_ell:.4f}  noise_sigma={ns:.5f}")
    print("="*60)

    ablations = [
        ("tau only",   [True,  False, False]),
        ("a only",     [False, True,  False]),
        ("b only",     [False, False, True ]),
        ("tau+a+b",    [True,  True,  True ]),
        ("none (oracle)", [False, False, False]),
    ]

    abl_results = {}
    print(f"\n{'Ablation':<18} {'PPG_err_mean':>14} {'PPG_err_std':>12} "
          f"{'RMSE':>8} {'Corr':>8}")
    print("-"*65)

    for name, mask in ablations:
        ppg_errs_all, rmse_all, corr_all = [], [], []

        for x_true_np, y_obs_np in windows:
            y_t = torch.tensor(y_obs_np).view(1,1,4000).to(device)
            x_s, phi_s = run_mcmc_ablation(
                y_t, prior, lik, diff, cfg, device,
                q_lam, infer_mask=mask,
                fix_log_sigma2=fix_ell, ns=ns)

            ppg_m, ppg_s = matched_ppg_err(x_s, phi_s, y_obs_np, device)
            ppg_errs_all.append(ppg_m)

            xs = torch.stack(x_s).numpy()
            mean = xs.mean(axis=0)
            rmse_all.append(float(np.sqrt(((mean-x_true_np)**2).mean())))
            corr_all.append(float(np.corrcoef(mean, x_true_np)[0,1]))

        abl_results[name] = dict(
            ppg=np.mean(ppg_errs_all), ppg_std=np.std(ppg_errs_all),
            rmse=np.mean(rmse_all), corr=np.mean(corr_all))

        print(f"{name:<18} {np.mean(ppg_errs_all):>14.4f} "
              f"{np.std(ppg_errs_all):>12.4f} "
              f"{np.mean(rmse_all):>8.4f} {np.mean(corr_all):>8.4f}")

    # Plot ablation
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [n for n, _ in ablations]
    ppgs  = [abl_results[n]["ppg"] for n in names]
    stds  = [abl_results[n]["ppg_std"] for n in names]
    colors = ["steelblue"]*4 + ["green"]
    bars = ax.bar(names, ppgs, color=colors, alpha=0.7)
    ax.errorbar(range(len(names)), ppgs, yerr=stds,
                fmt='none', color='black', capsize=4)
    ax.axhline(0.067, color='red', ls='--', lw=1.5,
               label="Oracle (Stage 1) PPG err=0.067")
    ax.set_ylabel("Matched-sample PPG error")
    ax.set_title(f"phi ablation — which component drives PPG discrepancy?\n"
                 f"SNR={args.snr_db}dB, sigma2 fixed at true value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR/"phi_ablation.png"), dpi=120)
    plt.close(fig)
    print(f"\n→ saved phi_ablation.png")

    # ================================================================
    # TASK B: credible interval check
    # ================================================================
    print()
    print("="*60)
    print("TASK B: Credible interval type and scalar coverage")
    print("="*60)

    print("\nInterval type comparison (Stage 1 oracle, window 0):")
    print("Running Stage 1 on window 0 for interval analysis...")

    x_true_np, y_obs_np = windows[0]
    y_t = torch.tensor(y_obs_np).view(1,1,4000).to(device)
    x_s, phi_s = run_mcmc_ablation(
        y_t, prior, lik, diff, cfg, device,
        q_lam, infer_mask=[False,False,False],
        fix_log_sigma2=fix_ell, ns=ns)

    xs_np = torch.stack(x_s).numpy()

    print(f"\n  N posterior samples: {len(x_s)}")
    print(f"\n  {'Level':>8} {'Gaussian(z)':>14} {'Equal-tailed%':>15} "
          f"{'Difference':>12}")
    print(f"  {'-'*52}")

    for level in [0.50, 0.80, 0.90, 0.95]:
        gauss_cov  = pointwise_coverage_check(xs_np, x_true_np, level)
        etail_cov  = equal_tailed_coverage(xs_np, x_true_np, level)
        cov_diff   = gauss_cov - etail_cov
        print(f"  {int(level*100):>7}%  {gauss_cov*100:>13.1f}%  "
              f"{etail_cov*100:>14.1f}%  {cov_diff*100:>+11.1f}%")

    print(f"\n  Interpretation:")
    print(f"  The Gaussian method uses z*std which assumes the posterior")
    print(f"  at each timestep is Gaussian. The equal-tailed method uses")
    print(f"  actual sample quantiles. If they differ, the posterior is")
    print(f"  non-Gaussian (e.g. skewed from DDIM initialization).")

    # Scalar clinical quantities
    print(f"\n  Scalar clinical quantities — coverage check:")
    print(f"  (Using {args.n_windows} windows × Stage 1 oracle samples)")

    scalar_names  = ["RR interval", "QRS amplitude", "PR mean"]
    scalar_fns    = [rr_interval, qrs_amplitude, pr_mean]
    nominal_levels = [0.50, 0.80, 0.90, 0.95]

    all_scalar_cov = {name: {l: [] for l in nominal_levels}
                      for name in scalar_names}

    for win_idx, (x_true_np_w, y_obs_np_w) in enumerate(windows):
        y_tw = torch.tensor(y_obs_np_w).view(1,1,4000).to(device)
        x_sw, phi_sw = run_mcmc_ablation(
            y_tw, prior, lik, diff, cfg, device,
            q_lam, infer_mask=[False,False,False],
            fix_log_sigma2=fix_ell, ns=ns)
        xs_w = torch.stack(x_sw).numpy()

        # True scalar values
        true_scalars = [fn(x_true_np_w) for fn in scalar_fns]

        # Posterior scalar distributions
        for sname, sfn, s_true in zip(scalar_names, scalar_fns,
                                       true_scalars):
            s_samples = np.array([sfn(s) for s in xs_w])
            s_samples = s_samples[~np.isnan(s_samples)]
            if len(s_samples) < 3 or np.isnan(s_true):
                continue
            for level in nominal_levels:
                cov = scalar_coverage(s_samples, s_true, level)
                all_scalar_cov[sname][level].append(cov)

    print(f"\n  {'Quantity':<16} " +
          "  ".join([f"Nom{int(l*100)}%→Emp" for l in nominal_levels]))
    print(f"  {'-'*70}")
    for sname in scalar_names:
        row = f"  {sname:<16}"
        for level in nominal_levels:
            vals = all_scalar_cov[sname][level]
            if len(vals) == 0:
                row += f"  {'N/A':>10}"
            else:
                emp = np.mean(vals) * 100
                row += f"  {int(level*100)}%→{emp:4.0f}%"
        print(row)

    print(f"\n  Note: scalar coverage is computed per-window (0 or 1)")
    print(f"  then averaged across windows. With only {args.n_windows} windows,")
    print(f"  estimates are noisy — need 20+ windows for reliable scalar coverage.")

    # Calibration plot for both methods
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: pointwise interval comparison
    levels_pct = [50, 80, 90, 95]
    gauss_covs = [pointwise_coverage_check(xs_np, windows[0][0], l/100)*100
                  for l in levels_pct]
    etail_covs = [equal_tailed_coverage(xs_np, windows[0][0], l/100)*100
                  for l in levels_pct]

    ax = axes[0]
    ax.plot([0,100],[0,100],'k--',lw=1,label="Ideal")
    ax.plot(levels_pct, gauss_covs, 'o-', color="steelblue",
            label="Gaussian z*std (current)")
    ax.plot(levels_pct, etail_covs, 's-', color="darkorange",
            label="Equal-tailed quantile")
    ax.set_xlabel("Nominal coverage %")
    ax.set_ylabel("Empirical coverage %")
    ax.set_title("Pointwise interval type comparison\n(window 0, Stage 1 oracle)")
    ax.legend(fontsize=8)
    ax.set_xlim(40,100); ax.set_ylim(40,100)

    # Right: scalar coverage
    ax = axes[1]
    for i, sname in enumerate(scalar_names):
        emp_covs = [np.mean(all_scalar_cov[sname][l])*100
                    if all_scalar_cov[sname][l] else float('nan')
                    for l in nominal_levels]
        ax.plot(levels_pct, emp_covs, 'o-',
                label=sname, alpha=0.8)
    ax.plot([0,100],[0,100],'k--',lw=1,label="Ideal")
    ax.set_xlabel("Nominal coverage %")
    ax.set_ylabel("Empirical coverage %")
    ax.set_title(f"Scalar clinical quantity coverage\n({args.n_windows} windows)")
    ax.legend(fontsize=7)
    ax.set_xlim(40,100); ax.set_ylim(-5,105)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR/"calibration_check.png"), dpi=120)
    plt.close(fig)
    print(f"\n→ saved calibration_check.png")
    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
