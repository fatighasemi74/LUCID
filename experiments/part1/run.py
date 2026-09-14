"""
experiments/part1/run.py
=========================
Full Part 1 pipeline: load K=10 priors + lambda*, run Phase 2 inference
on semi-synthetic test windows, compute all four evaluations, save results
and figures.

What this does
--------------
1. Load K=10 bootstrap prior checkpoints + frozen lambda*
2. Load synthetic PPG dataset (generate.py output) for each SNR level
3. For each SNR level and each test window:
   a. Run Phase 2 inference: outer loop over k, inner MCMC (phi-MH, z-MALA, x-diffusion)
   b. Estimate evidence Ẑ_k via SMC, compute weights omega_k
   c. Collect posterior samples {x, z, phi} per k
4. Compute four evaluations:
   Q1 Calibration  — coverage at {50, 80, 90, 95}% vs known truth
   Q2 Sharpness    — credible interval widths per clinical feature
   Q3 Failure pred — uncertainty-error correlation, AUROC, risk-coverage
   Q4 PPG consist  — push posterior ECG through H_phi, compare to observed y
   + Parameter recovery: ||phi_hat - phi_true||, phi coverage
5. Save results to experiments/part1/results/results_snr{N}dB.npz
6. Save figures to experiments/part1/results/figures/

Usage
-----
# First run — 50 test windows, fast
python experiments/part1/run.py --n-windows 50

# Full run — all 2198 windows
python experiments/part1/run.py --n-windows -1

# Single SNR level
python experiments/part1/run.py --snr-db 20 --n-windows 50
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.prior import ECGDiffusionPrior, BootstrapEnsemble
from models.likelihood import GaussianLikelihood
from models.noise_cov import CovMode
from models.diffusion_core import GaussianDiffusion
from models.unet1d import UNet1D
from inference.vi import MeanFieldGaussian, load_lambda
from inference.mala import (MCMCState, MALAConfig,
    z_to_params, phi_to_params, make_z_prior,
    _set_likelihood_params, set_state_params)
from inference.sampler import run_inference, SampleSet

# ======================================================================
# Paths
# ======================================================================

CKPT_DIR    = ROOT / "experiments" / "part1" / "checkpoints"
DATA_DIR    = ROOT / "experiments" / "part1" / "synthetic_data"
RESULTS_DIR = ROOT / "experiments" / "part1" / "results"
FIG_DIR     = RESULTS_DIR / "figures"

LAMBDA_PATH = CKPT_DIR / "lambda_star.pt"

# ======================================================================
# Configuration
# ======================================================================

K           = 10
SNR_LEVELS  = [5, 10, 20, 40]
N_WINDOWS   = 50      # default subset; -1 = all
N_INNER     = 30      # inner MCMC steps per k
BURN_IN     = 10      # burn-in steps
N_PARTICLES = 32      # SMC particles for evidence estimation
N_TEMPS     = 6       # SMC temperature levels

# φ_true from generate.py
PHI_TRUE = {
    "tau"  : 0.2,
    "a"    : 0.8,
    "b"    : 0.05,
    "sigma": 10.0,
}
# New structure: Z=[tau,a,b]  Phi=[log_sigma2]
Z_TRUE_VEC   = torch.tensor([PHI_TRUE["tau"], PHI_TRUE["a"], PHI_TRUE["b"]])
# PHI_TRUE_VEC kept for reference but log_sigma2 set per SNR
TAU_TRUE = PHI_TRUE["tau"]
A_TRUE   = PHI_TRUE["a"]
B_TRUE   = PHI_TRUE["b"]

# Clinical features — RR interval only for Part 1 (QRS/QTc need full pipeline)
# Simplified: use peak-to-peak interval as proxy for RR
FS = 500


# ======================================================================
# φ layout — must match train_vi.py
# ======================================================================

# phi_to_params removed — use z_to_params and phi_to_params from mala.py



# ======================================================================
# Functional forward operator (for PPG consistency check)
# ======================================================================

def _gaussian_kernel(sigma: float, size: int, device: torch.device) -> torch.Tensor:
    half = size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    k = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return k / k.sum()


def apply_h_phi(x: torch.Tensor, z_vec: torch.Tensor, fs: int = 500) -> torch.Tensor:
    """H_phi(x) = a * lowpass(shift(x, tau)) + b. z_vec=[tau,a,b], shape (B,1,L)."""
    z_vec = z_vec.squeeze()
    a = z_vec[1].abs()
    b = z_vec[2]
    k = _gaussian_kernel(10.0, 61, x.device).view(1, 1, -1)
    h = F_nn.conv1d(x, k, padding=30)
    # Apply delay — must match generate.py convention
    tau_samples = int(round(float(z_vec[0]) * fs))
    if tau_samples > 0:
        h = torch.roll(h, shifts=tau_samples, dims=-1)
        h[..., :tau_samples] = 0.0
    return a * h + b


# ======================================================================
# Load checkpoints
# ======================================================================

def load_ensemble(device: torch.device) -> BootstrapEnsemble:
    """Load K=10 trained priors into a BootstrapEnsemble."""
    priors = []
    for k in range(K):
        path = CKPT_DIR / f"prior_k{k:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint {path} not found. "
                "Run experiments/part1/train_bootstrap.py first."
            )
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        denoiser = UNet1D(
            base_ch =ckpt.get("base_ch", 64),
            time_dim=ckpt.get("time_dim", 128),
            n_res   =ckpt.get("n_res", 2),
        ).to(device)
        denoiser.load_state_dict(ckpt["model"])
        denoiser.eval()
        prior = ECGDiffusionPrior(
            denoiser  =denoiser,
            T         =ckpt.get("T", 1000),
            pred_type =ckpt.get("pred_type", "x0"),
            device    =device,
        ).to(device)
        priors.append(prior)
        print(f"  loaded prior_k{k:02d}.pt  val_loss={ckpt.get('val_loss',0):.4f}")

    ensemble = BootstrapEnsemble(priors)
    print(f"[ensemble] K={ensemble.B} priors loaded")
    return ensemble


def load_synthetic(snr_db: int, n_windows: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Load ECG and PPG from synthetic dataset for one SNR level."""
    path = DATA_DIR / f"snr_{snr_db:02d}dB.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run experiments/part1/generate.py first."
        )
    data = np.load(str(path))
    ecg  = data["ecg"]   # (N, 4000) normalized
    ppg  = data["ppg"]   # (N, 4000) synthetic PPG
    noise_sigma = float(data["noise_sigma"])

    if n_windows > 0 and n_windows < len(ecg):
        idx = np.random.choice(len(ecg), n_windows, replace=False)
        ecg = ecg[idx]
        ppg = ppg[idx]

    print(f"  loaded snr_{snr_db:02d}dB.npz: {len(ecg)} windows  "
          f"noise_sigma={noise_sigma:.5f}")
    return ecg, ppg, noise_sigma


# ======================================================================
# Clinical feature extraction (simplified — RR from peak detection)
# ======================================================================

def extract_rr(x_np: np.ndarray, fs: int = 500) -> float:
    """
    Estimate mean RR interval from a 1-D ECG window via peak detection.
    Returns RR in seconds, or NaN if fewer than 2 peaks found.
    """
    from scipy.signal import find_peaks
    x = x_np.ravel()
    # Threshold: peaks above 0.5 std, min distance 0.3s
    peaks, _ = find_peaks(x, height=0.5 * x.std(), distance=int(0.3 * fs))
    if len(peaks) < 2:
        return float("nan")
    rr_intervals = np.diff(peaks) / fs   # seconds
    return float(np.median(rr_intervals))


# ======================================================================
# Single-window inference
# ======================================================================

def infer_window(
    y               : torch.Tensor,
    ensemble        : BootstrapEnsemble,
    q_lambda        : MeanFieldGaussian,
    likelihood      : GaussianLikelihood,
    diffusion       : GaussianDiffusion,
    cfg             : MALAConfig,
    device          : torch.device,
    fix_log_sigma2  : float | None = None,
) -> tuple[list[SampleSet], torch.Tensor, torch.Tensor]:
    """Run Phase 2 inference for one PPG window.
    
    fix_log_sigma2: if set, overrides log_sigma2 at each MH step
    to the true noise variance (Part 1 controlled experiments).
    """
    return run_inference(
        y=y,
        ensemble=ensemble,
        q_lambda_star=q_lambda,
        likelihood=likelihood,
        diffusion=diffusion,
        phi_to_params_fn=phi_to_params,   # Phi=[log_sigma2] only
        phi_dim=1,
        device=device,
        cfg=cfg,
        verbose=False,
        fix_log_sigma2=fix_log_sigma2,
    )


# ======================================================================
# Evaluation functions
# ======================================================================

def posterior_mean_std(
    sample_sets : list[SampleSet],
    weights     : torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted posterior mean and std over x samples.

    Returns (mean, std) each shape (4000,).
    """
    all_x = []
    all_w = []
    for k, ss in enumerate(sample_sets):
        w_k = float(weights[k])
        for r in range(ss.n_samples):
            all_x.append(ss.x_samples[r].squeeze().numpy())
            all_w.append(w_k / ss.n_samples)

    all_x = np.stack(all_x)         # (N_total, 4000)
    all_w = np.array(all_w)
    all_w = all_w / all_w.sum()

    mean = (all_x * all_w[:, None]).sum(axis=0)
    var  = ((all_x - mean) ** 2 * all_w[:, None]).sum(axis=0)
    return mean, np.sqrt(var)


def posterior_phi_stats(
    sample_sets : list[SampleSet],
    weights     : torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and std over Z=[tau,a,b] samples."""
    all_z = []
    all_w = []
    for k, ss in enumerate(sample_sets):
        w_k = float(weights[k])
        for r in range(ss.n_samples):
            all_z.append(ss.z_samples[r].detach().numpy())
            all_w.append(w_k / ss.n_samples)
    all_z = np.stack(all_z)
    all_w = np.array(all_w) / sum(all_w)
    mean = (all_z * all_w[:, None]).sum(axis=0)
    var  = ((all_z - mean) ** 2 * all_w[:, None]).sum(axis=0)
    return mean, np.sqrt(var)


def variance_decomposition(
    sample_sets : list,
    weights     : torch.Tensor,
) -> dict:
    """
    Nested law-of-total-variance decomposition.
    Returns U_X, U_Z, U_Phi, U_Theta — four uncertainty sources.

    U_Theta = Var_theta[E[X|theta,y]]     model uncertainty
    U_Phi   = E_theta[Var_phi[E[X|phi,theta,y]]]  operator uncertainty
    U_Z     = E[Var_z[E[X|z,phi,theta,y]]]  nuisance uncertainty
    U_X     = E[Var[X|z,phi,theta,y]]     irreducible posterior uncertainty

    Approximation from samples:
    - All x samples pooled with weights give total posterior mean/var
    - Per-k means give between-theta variance (U_Theta)
    - Within each k, per-phi means give U_Phi (approximated as phi variance)
    - Residual split between U_Z and U_X

    Simplified implementation using sample variances.
    """
    K = len(sample_sets)
    all_w = weights.numpy()

    # Collect per-k sample means
    k_means = []
    k_vars  = []
    k_phi_vars = []
    k_z_vars   = []

    for k, ss in enumerate(sample_sets):
        xs = ss.x_samples.detach().squeeze(1).numpy()   # (N_s, 4000)
        k_means.append(xs.mean(axis=0))
        k_vars.append(xs.var(axis=0))

        # phi variance across samples (proxy for U_Phi per k)
        phis = ss.phi_samples.detach().numpy()           # (N_s, 4)
        k_phi_vars.append(phis.var(axis=0).mean())

        # z variance across samples (proxy for U_Z per k)
        zs = ss.z_samples.detach().numpy()              # (N_s, d_z)
        k_z_vars.append(zs.var(axis=0).mean())

    k_means   = np.stack(k_means)   # (K, 4000)
    k_vars    = np.stack(k_vars)    # (K, 4000)
    k_phi_vars = np.array(k_phi_vars)
    k_z_vars   = np.array(k_z_vars)

    # Weighted grand mean
    grand_mean = (k_means * all_w[:, None]).sum(axis=0)

    # U_Theta: between-model variance (variance of per-k means)
    u_theta = float(
        ((k_means - grand_mean)**2 * all_w[:, None]).sum(axis=0).mean()
    )

    # U_X: mean within-k posterior variance (irreducible)
    u_x = float((k_vars * all_w[:, None]).sum(axis=0).mean())

    # U_Phi: mean phi variance across k (operator uncertainty)
    u_phi = float((k_phi_vars * all_w).sum())

    # U_Z: mean z variance across k (nuisance uncertainty)
    u_z = float((k_z_vars * all_w).sum())

    # Total
    u_total = u_theta + u_x + u_phi + u_z

    return {
        "U_X"    : u_x,
        "U_Z"    : u_z,
        "U_Phi"  : u_phi,
        "U_Theta": u_theta,
        "U_total": u_total,
    }


def check_coverage(
    x_true     : np.ndarray,
    mean       : np.ndarray,
    std        : np.ndarray,
    level      : float,
) -> float:
    """Fraction of x_true timesteps inside ± z_alpha * std band."""
    from scipy import stats
    z = stats.norm.ppf((1 + level) / 2)
    inside = np.abs(x_true - mean) <= z * std
    return float(inside.mean())


def ppg_consistency(
    sample_sets : list[SampleSet],
    weights     : torch.Tensor,
    y_obs       : np.ndarray,
    device      : torch.device,
) -> float:
    """
    Push posterior x samples through H_phi and compare to observed y.
    Returns mean absolute error between regenerated PPG and y_obs.
    """
    errors = []
    for k, ss in enumerate(sample_sets):
        w_k = float(weights[k])
        for r in range(ss.n_samples):
            x_r   = torch.tensor(ss.x_samples[r]).unsqueeze(0).to(device)
            z_r   = ss.z_samples[r].to(device)   # Z=[tau,a,b]
            with torch.no_grad():
                ppg_pred = apply_h_phi(x_r, z_r).squeeze().cpu().numpy()
            err = float(np.abs(ppg_pred - y_obs).mean())
            errors.append(err * w_k / ss.n_samples)
    return float(sum(errors))


# ======================================================================
# Figures
# ======================================================================

def plot_calibration(coverage_results: dict, snr_db: int, out_dir: Path) -> None:
    """Coverage curve: empirical vs nominal."""
    levels = [0.50, 0.80, 0.90, 0.95]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    emp = [np.mean(coverage_results[l]) for l in levels]
    ax.plot(levels, emp, "o-", color="steelblue", label="empirical")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Calibration — SNR {snr_db} dB")
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"calibration_snr{snr_db:02d}dB.png"
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_reconstruction(
    x_true : np.ndarray,
    y_obs  : np.ndarray,
    mean   : np.ndarray,
    std    : np.ndarray,
    snr_db : int,
    win_idx: int,
    out_dir: Path,
) -> None:
    """Posterior mean ± 1std vs ground truth ECG."""
    t = np.arange(len(x_true)) / FS
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, x_true, "k", lw=0.8, label="ECG true")
    ax.plot(t, mean,   "steelblue", lw=0.8, label="posterior mean")
    ax.fill_between(t, mean - std, mean + std,
                    alpha=0.3, color="steelblue", label="±1 std")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"ECG reconstruction — SNR {snr_db} dB  (window {win_idx})")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(t, y_obs, "darkorange", lw=0.8, label="PPG observed")
    ax.set_ylabel("Normalized amplitude")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = out_dir / f"recon_snr{snr_db:02d}dB_win{win_idx:03d}.png"
    fig.savefig(str(out), dpi=120)
    plt.close(fig)


def plot_uncertainty_vs_error(
    errors       : list[float],
    uncertainties: list[float],
    snr_db       : int,
    out_dir      : Path,
) -> None:
    """Scatter: posterior std (uncertainty) vs reconstruction error."""
    from scipy.stats import pearsonr
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(uncertainties, errors, alpha=0.5, s=20, color="steelblue")
    if len(errors) > 2:
        r, p = pearsonr(uncertainties, errors)
        ax.set_title(f"Uncertainty vs Error — SNR {snr_db} dB\n"
                     f"r={r:.3f}  p={p:.3f}")
    ax.set_xlabel("Posterior std (uncertainty)")
    ax.set_ylabel("|mean - truth| (error)")
    fig.tight_layout()
    out = out_dir / f"uncertainty_error_snr{snr_db:02d}dB.png"
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Part 1 full inference pipeline")
    ap.add_argument("--snr-db", nargs="+", type=int, default=SNR_LEVELS,
                    help="SNR levels to run (default: 5 10 20 40)")
    ap.add_argument("--n-windows", type=int, default=N_WINDOWS,
                    help="test windows per SNR level (-1 = all)")
    ap.add_argument("--n-inner", type=int, default=N_INNER,
                    help="inner MCMC steps N")
    ap.add_argument("--burn-in", type=int, default=BURN_IN,
                    help="burn-in steps B")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pulse2Posterior — Part 1 Inference")
    print("=" * 60)
    print(f"Device    : {device}")
    print(f"SNR levels: {args.snr_db}")
    print(f"Windows   : {args.n_windows} per SNR")
    print(f"N inner   : {args.n_inner}  burn-in: {args.burn_in}")
    print()

    # ── Load shared components ────────────────────────────────────────
    print("[setup] loading ensemble …")
    ensemble = load_ensemble(device)

    print("[setup] loading lambda* …")
    q_lambda = load_lambda(LAMBDA_PATH, device)

    likelihood = GaussianLikelihood(
        obs_dim=4000, fs=500.0,
        cov_mode=CovMode.ISOTROPIC,
        kernel_sigma=10.0, kernel_size=61,
    ).to(device)

    diffusion = GaussianDiffusion(
        T=1000, schedule="cosine", pred_type="x0"
    ).to(device)

    cfg = MALAConfig(
        n_inner  =args.n_inner,
        burn_in  =args.burn_in,
        step_size_z=1e-3,
        t_anneal =list(range(500, 0, -10)) + [1],   # 50 steps
    )

    # ── Run per SNR level ─────────────────────────────────────────────
    for snr_db in args.snr_db:
        print()
        print(f"{'='*60}")
        print(f"SNR = {snr_db} dB")
        print(f"{'='*60}")

        ecg_np, ppg_np, noise_sigma = load_synthetic(snr_db, args.n_windows)
        n_win = len(ecg_np)

        # Fix sigma2 at true value for Part 1 controlled experiments
        # Professor: keep sigma2=sigma2_true for SNR sweep
        import math as _math
        true_log_sigma2 = float(_math.log(noise_sigma**2))
        print(f"  [sigma2] Fixed log_sigma2={true_log_sigma2:.4f}  "
              f"(noise_sigma={noise_sigma:.5f})")

        # Per-window results
        rmse_list, corr_list   = [], []
        unc_list               = []
        ppg_consist_list       = []
        phi_errors             = []
        coverage_results       = {l: [] for l in [0.50, 0.80, 0.90, 0.95]}
        ux_list, uz_list, uphi_list, utheta_list = [], [], [], []

        t_snr = time.time()
        for i in range(n_win):
            x_true = ecg_np[i]                            # (4000,)
            y_obs  = ppg_np[i]                            # (4000,)

            y_t = torch.tensor(y_obs, dtype=torch.float32
                               ).view(1, 1, 4000).to(device)

            t_win = time.time()
            sample_sets, weights, evidences = infer_window(
                y_t, ensemble, q_lambda,
                likelihood, diffusion, cfg, device,
                fix_log_sigma2=true_log_sigma2,
            )

            # Posterior stats
            mean_x, std_x = posterior_mean_std(sample_sets, weights)

            # Q1 Calibration
            for level in [0.50, 0.80, 0.90, 0.95]:
                cov = check_coverage(x_true, mean_x, std_x, level)
                coverage_results[level].append(cov)

            # Q2 Sharpness — mean posterior std
            unc = float(std_x.mean())
            unc_list.append(unc)

            # Reconstruction quality
            err   = float(np.abs(mean_x - x_true).mean())
            rmse  = float(np.sqrt(((mean_x - x_true) ** 2).mean()))
            corr  = float(np.corrcoef(mean_x, x_true)[0, 1])
            rmse_list.append(rmse)
            corr_list.append(corr)

            # Q4 PPG consistency
            ppg_err = ppg_consistency(sample_sets, weights, y_obs, device)
            ppg_consist_list.append(ppg_err)

            # Parameter recovery — Z=[tau,a,b]
            phi_mean, phi_std = posterior_phi_stats(sample_sets, weights)
            z_true_np = Z_TRUE_VEC.numpy()  # [tau, a, b]
            phi_err = float(np.abs(phi_mean - z_true_np).mean())
            phi_errors.append(phi_err)

            elapsed = time.time() - t_win
            if i == 0 or (i + 1) % 10 == 0:
                print(f"  window {i+1:03d}/{n_win}  "
                      f"rmse={rmse:.4f}  corr={corr:.3f}  "
                      f"unc={unc:.4f}  ppg_err={ppg_err:.4f}  "
                      f"phi_err={phi_err:.4f}  {elapsed:.1f}s")

            # Variance decomposition U_X, U_Z, U_Phi, U_Theta
            vd = variance_decomposition(sample_sets, weights)
            ux_list.append(vd['U_X'])
            uz_list.append(vd['U_Z'])
            uphi_list.append(vd['U_Phi'])
            utheta_list.append(vd['U_Theta'])
            if i == 0:
                print(f"  [UQ] U_X={vd['U_X']:.4f} U_Z={vd['U_Z']:.6f} "
                      f"U_Phi={vd['U_Phi']:.6f} U_Theta={vd['U_Theta']:.4f}")

            # Save one reconstruction plot
            if i == 0:
                plot_reconstruction(
                    x_true, y_obs, mean_x, std_x,
                    snr_db, i, FIG_DIR
                )

        # ── Summary ───────────────────────────────────────────────────
        print()
        print(f"SNR {snr_db} dB — {n_win} windows in "
              f"{(time.time()-t_snr)/60:.1f} min")
        print(f"  RMSE          : {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
        print(f"  Correlation   : {np.mean(corr_list):.4f} ± {np.std(corr_list):.4f}")
        print(f"  Uncertainty   : {np.mean(unc_list):.4f} ± {np.std(unc_list):.4f}")
        print(f"  PPG consist   : {np.mean(ppg_consist_list):.4f}")
        print(f"  Phi error     : {np.mean(phi_errors):.4f}")
        print("  Coverage:")
        for level in [0.50, 0.80, 0.90, 0.95]:
            emp = np.mean(coverage_results[level])
            print(f"    {int(level*100):2d}% nominal → {emp*100:.1f}% empirical")

        # ── Save results ──────────────────────────────────────────────
        out_path = RESULTS_DIR / f"results_snr{snr_db:02d}dB.npz"
        np.savez(str(out_path),
                 rmse=np.array(rmse_list),
                 corr=np.array(corr_list),
                 uncertainty=np.array(unc_list),
                 ppg_consistency=np.array(ppg_consist_list),
                 phi_error=np.array(phi_errors),
                 U_X=np.array(ux_list),
                 U_Z=np.array(uz_list),
                 U_Phi=np.array(uphi_list),
                 U_Theta=np.array(utheta_list),
                 coverage_50=np.array(coverage_results[0.50]),
                 coverage_80=np.array(coverage_results[0.80]),
                 coverage_90=np.array(coverage_results[0.90]),
                 coverage_95=np.array(coverage_results[0.95]),
                 snr_db=np.float32(snr_db),
                 n_windows=np.int32(n_win),
        )
        print(f"  results → {out_path.name}")

        # ── Figures ───────────────────────────────────────────────────
        plot_calibration(coverage_results, snr_db, FIG_DIR)
        plot_uncertainty_vs_error(rmse_list, unc_list, snr_db, FIG_DIR)

    print()
    print("=" * 60)
    print("Part 1 complete.")
    print(f"Results : {RESULTS_DIR}/")
    print(f"Figures : {FIG_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
