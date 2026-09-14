"""
experiments/part1/check_log_sigma2.py
=======================================
Professor's diagnostic for log_sigma2:

TASK 1: Fix X=x_true, Z=z_true, (tau,a,b)=true values.
        Plot conditional log posterior as function of ell=log_sigma2.
        Verify maximum is near ell_true=-3.0.

TASK 2: Add separate local random-walk MH for log_sigma2:
        ell' = ell + delta * xi,  xi ~ N(0,1)
        Tune delta to get reasonable acceptance rate (20-40%).

TASK 3: Rerun Stage 3 with the new log_sigma2 update.
        Check if matched-sample PPG error returns to oracle noise scale.

Usage
-----
python experiments/part1/check_log_sigma2.py --n-windows 5
"""

import sys, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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
from inference.vi import MeanFieldGaussian, load_lambda
from inference.mala import (MCMCState, MALAConfig,
                             phi_mh_step, z_mala_step,
                             x_diffusion_step, _set_likelihood_params)
from inference.sampler import make_gaussian_z_prior

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
DATA_DIR = ROOT / "experiments/part1/synthetic_data"
OUT_DIR  = ROOT / "experiments/part1/check_log_sigma2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS       = 500
PHI_TRUE = torch.tensor([0.2, 0.8, 0.05, -3.0])
Z_TRUE   = torch.tensor([[0.2]])

def phi_to_params(phi):
    return {"log_a": phi[1].abs().log(), "b": phi[2],
            "tau": phi[0], "log_diag": phi[3].expand(1)}

def apply_h_phi(x, phi_vec, fs=FS):
    phi_vec = phi_vec.to(x.device)
    a = phi_vec[1].abs(); b = phi_vec[2]
    half = 30
    t = torch.arange(-half, half+1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t**2/(2*10.0**2)); k=k/k.sum()
    h = F.conv1d(x, k.view(1,1,-1), padding=half)
    tau_s = int(round(float(phi_vec[0])*fs))
    if tau_s > 0:
        h = torch.roll(h, shifts=tau_s, dims=-1)
        h[..., :tau_s] = 0.0
    return a*h + b

def log_likelihood_at_ell(y, x, phi_fixed_3, ell, lik, device):
    """
    Compute log L(y | x, z_true, phi) where phi=[tau,a,b,ell].
    phi_fixed_3 = [tau, a, b] fixed at true values.
    ell = log_sigma2 to evaluate.
    """
    phi = torch.tensor([phi_fixed_3[0], phi_fixed_3[1],
                        phi_fixed_3[2], ell],
                       dtype=torch.float32, device=device)
    params = phi_to_params(phi)
    _set_likelihood_params(lik, params)
    with torch.no_grad():
        ll = lik.log_likelihood(y, x, None).sum()
    return float(ll)

def log_prior_ell(ell, mu=-3.0, sigma=1.0):
    """Log prior on ell = log_sigma2: N(mu, sigma^2)."""
    return -0.5 * ((ell - mu) / sigma)**2

def load_prior(device):
    ckpt = torch.load(str(CKPT_DIR/"prior_k00.pt"),
                      map_location=device, weights_only=False)
    unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
    unet.load_state_dict(ckpt["model"]); unet.eval()
    return ECGDiffusionPrior(denoiser=unet, T=1000, pred_type="x0",
                             device=device).to(device)


# ======================================================================
# Local random-walk MH for log_sigma2 only
# ======================================================================

def rw_mh_log_sigma2(
    ell_current : float,
    y           : torch.Tensor,
    x           : torch.Tensor,
    phi_rest    : torch.Tensor,   # [tau, a, b] — other phi components
    lik         : GaussianLikelihood,
    device      : torch.device,
    delta       : float = 0.3,
    prior_mu    : float = -3.0,
    prior_sigma : float = 1.0,
) -> tuple[float, bool]:
    """
    Local random-walk MH step for ell = log_sigma2.

    Proposal: ell' = ell + delta * xi,  xi ~ N(0,1)
    Accept with min(1, exp(log_post(ell') - log_post(ell))).

    Returns (ell_new, accepted).
    """
    xi   = float(torch.randn(1).item())
    ell_prop = ell_current + delta * xi

    phi_current = torch.tensor(
        [float(phi_rest[0]), float(phi_rest[1]),
         float(phi_rest[2]), ell_current],
        dtype=torch.float32, device=device)
    phi_prop = torch.tensor(
        [float(phi_rest[0]), float(phi_rest[1]),
         float(phi_rest[2]), ell_prop],
        dtype=torch.float32, device=device)

    ll_curr = log_likelihood_at_ell(y, x, phi_rest.tolist(), ell_current, lik, device)
    ll_prop = log_likelihood_at_ell(y, x, phi_rest.tolist(), ell_prop,    lik, device)
    lp_curr = log_prior_ell(ell_current, prior_mu, prior_sigma)
    lp_prop = log_prior_ell(ell_prop,    prior_mu, prior_sigma)

    log_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr)
    accepted  = math.log(max(float(torch.rand(1)), 1e-10)) < log_ratio

    return (ell_prop if accepted else ell_current), accepted


def tune_delta(y, x, phi_rest, lik, device, target_acc=0.30, n_steps=200):
    """
    Tune delta for rw_mh_log_sigma2 to hit target acceptance rate.
    Returns best delta found.
    """
    best_delta = 0.3
    best_gap   = 1.0

    for delta in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        ell   = -3.0
        n_acc = 0
        for _ in range(n_steps):
            ell, acc = rw_mh_log_sigma2(ell, y, x, phi_rest,
                                         lik, device, delta)
            if acc: n_acc += 1
        rate = n_acc / n_steps
        gap  = abs(rate - target_acc)
        if gap < best_gap:
            best_gap   = gap
            best_delta = delta

    return best_delta


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

    prior    = load_prior(device)
    diff     = GaussianDiffusion(T=1000, schedule="cosine",
                                  pred_type="x0").to(device)
    lik      = GaussianLikelihood(obs_dim=4000, fs=500.0,
                                   cov_mode=CovMode.ISOTROPIC,
                                   kernel_sigma=10.0,
                                   kernel_size=61).to(device)
    q_lambda = load_lambda(CKPT_DIR/"lambda_star.pt", device)

    # ================================================================
    # TASK 1: Conditional log posterior as function of ell
    # ================================================================
    print("="*60)
    print("TASK 1: Conditional log posterior vs ell = log_sigma2")
    print("Fix X=x_true, Z=z_true, tau=0.2, a=0.8, b=0.05")
    print("="*60)

    win_idx   = 0
    x_true_np = ecg_all[win_idx]
    y_obs_np  = ppg_all[win_idx]
    x_true_t  = torch.tensor(x_true_np).view(1,1,4000).to(device)
    y_t       = torch.tensor(y_obs_np).view(1,1,4000).to(device)

    ell_range = np.linspace(-8.0, 0.0, 100)
    phi_rest  = torch.tensor([0.2, 0.8, 0.05])   # tau, a, b at true values
    log_posts = []

    for ell in ell_range:
        ll = log_likelihood_at_ell(
            y_t, x_true_t, phi_rest.tolist(), float(ell), lik, device)
        lp = log_prior_ell(float(ell), mu=-3.0, sigma=1.0)
        log_posts.append(ll + lp)

    log_posts = np.array(log_posts)
    ell_max   = float(ell_range[np.argmax(log_posts)])

    print(f"  ell_true = {math.log(ns**2):.4f}  (noise_sigma={ns:.5f})")
    print(f"  ell_max  = {ell_max:.4f}  (conditional log posterior peak)")
    print(f"  gap      = {abs(ell_max - math.log(ns**2)):.4f}")
    if abs(ell_max - math.log(ns**2)) < 0.5:
        print(f"  ✓ Maximum is near ell_true — likelihood is correctly specified")
    else:
        print(f"  ⚠ Maximum is far from ell_true — likelihood may be misspecified")

    # Also compute at true phi for reference
    ll_true = log_likelihood_at_ell(
        y_t, x_true_t, phi_rest.tolist(), math.log(ns**2), lik, device)
    lp_true = log_prior_ell(math.log(ns**2))
    print(f"  log_post at ell_true: {ll_true + lp_true:.2f}")
    print(f"  log_post at ell_max:  {max(log_posts):.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ell_range, log_posts, 'steelblue', lw=1.5,
            label="log p(ell | x_true, z_true, tau_true, a_true, b_true)")
    ax.axvline(math.log(ns**2), color='red', ls='--', lw=1.5,
               label=f"ell_true = {math.log(ns**2):.3f}")
    ax.axvline(ell_max, color='orange', ls=':', lw=1.5,
               label=f"ell_max = {ell_max:.3f}")
    ax.set_xlabel("ell = log_sigma2")
    ax.set_ylabel("log posterior")
    ax.set_title("Conditional log posterior vs log_sigma2\n"
                 "(all other variables fixed at true values)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR/"log_sigma2_posterior.png"), dpi=120)
    plt.close(fig)
    print(f"  → saved log_sigma2_posterior.png\n")

    # ================================================================
    # TASK 2: Tune delta for local RW-MH on log_sigma2
    # ================================================================
    print("="*60)
    print("TASK 2: Tune delta for local RW-MH on log_sigma2")
    print("="*60)

    print("  Tuning delta on window 0 (x=x_true, tau/a/b fixed at true)...")
    # Set other phi params correctly first
    params = phi_to_params(PHI_TRUE.to(device))
    _set_likelihood_params(lik, params)

    delta_star = tune_delta(y_t, x_true_t, phi_rest, lik, device,
                             target_acc=0.30, n_steps=500)
    print(f"  Best delta = {delta_star:.4f}")

    # Verify acceptance rate with delta_star
    ell = -3.0
    n_acc = 0
    ell_trace = [ell]
    for _ in range(500):
        ell, acc = rw_mh_log_sigma2(ell, y_t, x_true_t,
                                     phi_rest, lik, device, delta_star)
        if acc: n_acc += 1
        ell_trace.append(ell)

    print(f"  Acceptance rate at delta={delta_star}: {n_acc/500:.3f}")
    print(f"  ell trace: mean={np.mean(ell_trace):.3f}  "
          f"std={np.std(ell_trace):.3f}  "
          f"(true={math.log(ns**2):.3f})")

    # Trace plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ell_trace[:200], 'steelblue', lw=0.8)
    axes[0].axhline(math.log(ns**2), color='red', ls='--',
                    label=f"ell_true={math.log(ns**2):.3f}")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("ell = log_sigma2")
    axes[0].set_title(f"RW-MH trace (delta={delta_star}, acc={n_acc/500:.3f})")
    axes[0].legend(fontsize=8)

    axes[1].hist(ell_trace[50:], bins=30, color="steelblue",
                  alpha=0.7, density=True)
    axes[1].axvline(math.log(ns**2), color='red', ls='--',
                    label=f"ell_true={math.log(ns**2):.3f}")
    axes[1].set_xlabel("ell = log_sigma2")
    axes[1].set_title("RW-MH posterior histogram")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR/"rw_mh_trace.png"), dpi=120)
    plt.close(fig)
    print(f"  → saved rw_mh_trace.png\n")

    # ================================================================
    # TASK 3: Rerun Stage 3 with RW-MH for log_sigma2
    # ================================================================
    print("="*60)
    print("TASK 3: Stage 3 with local RW-MH for log_sigma2")
    print(f"Windows: {args.n_windows}  N=50  delta={delta_star:.4f}")
    print("="*60)

    gamma_t  = {t: args.gamma for t in list(range(500, 0, -10)) + [1]}
    log_z_prior = make_gaussian_z_prior(mu=0.2, sigma=0.1)

    def score_fn(x_t, t_b):
        return prior.denoiser(x_t, t_b)

    ppg_s1_all, ppg_s3_all = [], []
    acc_phi_all, acc_ell_all = [], []
    corr_s3_all, rmse_s3_all = [], []

    for win_idx in range(args.n_windows):
        x_true_np = ecg_all[win_idx]
        y_obs_np  = ppg_all[win_idx]
        x_t_win   = torch.tensor(y_obs_np).view(1,1,4000).to(device)

        # Stage 1 PPG error for reference
        cfg_s1 = MALAConfig(n_inner=50, burn_in=10, step_size_z=1e-3,
                             t_anneal=list(range(500,0,-10))+[1],
                             gamma_t=gamma_t)
        params_true = phi_to_params(PHI_TRUE.to(device))
        params_true["log_diag"] = torch.tensor(
            [math.log(ns**2)], device=device)
        _set_likelihood_params(lik, params_true)

        phi0 = PHI_TRUE.clone().to(device)
        z0   = Z_TRUE.clone().to(device)
        state = MCMCState(x=torch.randn(1,1,4000,device=device),
                          z=z0, phi=phi0, log_lik=-1e6, step=0)
        x_s1, phi_s1 = [], []
        for r in range(50):
            _set_likelihood_params(lik, phi_to_params(phi0))
            state = x_diffusion_step(state, x_t_win, score_fn,
                                      lik, diff, cfg_s1, device)
            state = MCMCState(x=state.x, z=z0, phi=phi0,
                              log_lik=state.log_lik, step=r+1,
                              n_acc_phi=0, n_acc_z=0)
            if r >= 10:
                x_s1.append(state.x.squeeze().cpu())
                phi_s1.append(phi0.cpu())
        ppg_s1 = []
        for i in range(len(x_s1)):
            xi = x_s1[i].view(1,1,4000).to(device)
            with torch.no_grad():
                pi = apply_h_phi(xi, PHI_TRUE.to(device)).squeeze().cpu().numpy()
            ppg_s1.append(float(np.abs(pi - y_obs_np).mean()))
        ppg_s1_all.extend(ppg_s1)

        # Stage 3 with RW-MH for log_sigma2 + q_lambda* for (tau,a,b)
        phi_init = q_lambda.sample(1).squeeze(0).detach().to(device)
        z_init   = torch.tensor([[0.2]], dtype=torch.float32, device=device)
        ell_curr = float(phi_init[3])

        params_init = phi_to_params(phi_init)
        params_init["log_diag"] = torch.tensor([ell_curr], device=device)
        _set_likelihood_params(lik, params_init)

        state3 = MCMCState(x=torch.randn(1,1,4000,device=device),
                           z=z_init, phi=phi_init,
                           log_lik=-1e6, step=0)
        x_s3, phi_s3 = [], []
        n_acc_phi3, n_acc_ell3 = 0, 0
        N3, B3 = 50, 10

        for r in range(N3):
            # (tau, a, b) updated via q_lambda* MH as before
            state3, acc_phi = phi_mh_step(
                state3, x_t_win, lik, q_lambda,
                phi_to_params, device)
            if acc_phi: n_acc_phi3 += 1

            # log_sigma2 updated via local RW-MH separately
            phi_rest_curr = state3.phi[:3]   # [tau, a, b]
            x_curr = state3.x
            ell_new, acc_ell = rw_mh_log_sigma2(
                ell_curr, x_t_win, x_curr,
                phi_rest_curr, lik, device, delta_star)
            if acc_ell: n_acc_ell3 += 1
            ell_curr = ell_new

            # Update full phi with new ell
            phi_new = torch.tensor(
                [float(state3.phi[0]), float(state3.phi[1]),
                 float(state3.phi[2]), ell_curr],
                dtype=torch.float32, device=device)
            params_new = phi_to_params(phi_new)
            _set_likelihood_params(lik, params_new)
            state3 = MCMCState(x=state3.x, z=state3.z, phi=phi_new,
                               log_lik=state3.log_lik, step=r+1,
                               n_acc_phi=state3.n_acc_phi,
                               n_acc_z=state3.n_acc_z)

            # Z update
            state3, _ = z_mala_step(
                state3, x_t_win, lik, log_z_prior, cfg_s1, device)

            # X update
            state3 = x_diffusion_step(
                state3, x_t_win, score_fn, lik, diff, cfg_s1, device)
            state3 = MCMCState(x=state3.x, z=state3.z, phi=state3.phi,
                               log_lik=state3.log_lik, step=r+1,
                               n_acc_phi=state3.n_acc_phi,
                               n_acc_z=state3.n_acc_z)

            if r >= B3:
                x_s3.append(state3.x.squeeze().cpu())
                phi_s3.append(state3.phi.cpu())

        acc_phi_all.append(n_acc_phi3 / N3)
        acc_ell_all.append(n_acc_ell3 / N3)

        # Evaluate
        xs3   = torch.stack(x_s3).numpy()
        mean3 = xs3.mean(axis=0)
        rmse3 = float(np.sqrt(((mean3 - x_true_np)**2).mean()))
        corr3 = float(np.corrcoef(mean3, x_true_np)[0,1])
        corr_s3_all.append(corr3)
        rmse_s3_all.append(rmse3)

        ppg_s3 = []
        for i in range(len(x_s3)):
            xi   = x_s3[i].view(1,1,4000).to(device)
            phii = phi_s3[i].to(device)
            with torch.no_grad():
                pi = apply_h_phi(xi, phii).squeeze().cpu().numpy()
            ppg_s3.append(float(np.abs(pi - y_obs_np).mean()))
        ppg_s3_all.extend(ppg_s3)

        ell_samples = [float(p[3]) for p in phi_s3]
        print(f"  Win {win_idx}: acc_phi={n_acc_phi3/N3:.3f}  "
              f"acc_ell={n_acc_ell3/N3:.3f}  "
              f"ell_mean={np.mean(ell_samples):.3f}  "
              f"ppg_s1={np.mean(ppg_s1):.4f}  "
              f"ppg_s3={np.mean(ppg_s3):.4f}  "
              f"corr={corr3:.4f}")

    print()
    print("  Summary:")
    print(f"    Mean acc_phi (tau,a,b): {np.mean(acc_phi_all):.3f}")
    print(f"    Mean acc_ell (log_s2):  {np.mean(acc_ell_all):.3f}")
    print(f"    PPG error Stage 1 (oracle phi): "
          f"mean={np.mean(ppg_s1_all):.4f}  std={np.std(ppg_s1_all):.4f}")
    print(f"    PPG error Stage 3 (RW-MH ell): "
          f"mean={np.mean(ppg_s3_all):.4f}  std={np.std(ppg_s3_all):.4f}")
    print(f"    Reconstruction: RMSE={np.mean(rmse_s3_all):.4f}  "
          f"corr={np.mean(corr_s3_all):.4f}")
    print()
    ell_true = math.log(ns**2)
    if np.mean(ppg_s3_all) < 2 * np.mean(ppg_s1_all):
        print("  ✓ Stage 3 PPG error much closer to oracle — "
              "log_sigma2 mixing improved")
    else:
        print("  ⚠ Stage 3 PPG error still elevated — "
              "further tuning may be needed")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
