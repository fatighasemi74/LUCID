"""
experiments/part1/run_vitaldb.py
==================================
Run LUCID inference on held-out VitalDB patients.
Reports ECG RMSE/correlation, HR MAE, uncertainty vs PPG error.

Usage
-----
python experiments/part1/run_vitaldb.py --n-windows 20 --n-inner 100 --burn-in 20
"""

import argparse
import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
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
from inference.mala import (MALAConfig, z_to_params, phi_to_params,
                             make_z_prior, _set_likelihood_params)
from inference.sampler import run_inference
from data.vitaldb import (find_paired_cases, split_case_ids,
                           load_case_signals, is_physical_units,
                           qc_paired_window, WINDOW_SAMPLES, FS)

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
OUT_DIR  = ROOT / "experiments/part1/vitaldb_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Clinical quantities
# ======================================================================

def estimate_heart_rate(ecg_np: np.ndarray, fs: int = FS) -> float:
    try:
        peaks, _ = find_peaks(ecg_np,
                              height=0.3 * np.percentile(ecg_np, 95),
                              distance=int(0.4 * fs))
        if len(peaks) < 2:
            return float('nan')
        rr = np.diff(peaks) / fs
        hr = 60.0 / np.mean(rr)
        return float(hr) if 30 < hr < 220 else float('nan')
    except Exception:
        return float('nan')


def estimate_rr_interval(ecg_np: np.ndarray, fs: int = FS) -> float:
    try:
        peaks, _ = find_peaks(ecg_np,
                              height=0.3 * np.percentile(ecg_np, 95),
                              distance=int(0.4 * fs))
        if len(peaks) < 2:
            return float('nan')
        return float(np.mean(np.diff(peaks)) / fs)
    except Exception:
        return float('nan')


def posterior_clinical(x_samples: list) -> dict:
    hrs, rrs = [], []
    for xs in x_samples:
        xn = xs.numpy()
        hr = estimate_heart_rate(xn)
        rr = estimate_rr_interval(xn)
        if not math.isnan(hr): hrs.append(hr)
        if not math.isnan(rr): rrs.append(rr)
    return {
        'hr_mean': float(np.mean(hrs)) if hrs else float('nan'),
        'hr_std':  float(np.std(hrs))  if hrs else float('nan'),
        'rr_mean': float(np.mean(rrs)) if rrs else float('nan'),
        'rr_std':  float(np.std(rrs))  if rrs else float('nan'),
    }


# ======================================================================
# Forward operator
# ======================================================================

def compute_case_lag(ecg_raw: np.ndarray, ppg_raw: np.ndarray,
                    fs: int = FS, max_lag_s: float = 1.5) -> int:
    """
    Compute ECG-PPG synchronization lag for one case using
    cross-correlation on first 30 seconds of data.
    Returns lag in samples (positive = PPG lags ECG).
    Applied consistently to all windows from this case.
    """
    from scipy.signal import correlate
    n = min(len(ecg_raw), len(ppg_raw), fs * 30)  # 30s
    ecg_s = ecg_raw[:n]; ppg_s = ppg_raw[:n]
    ecg_n = (ecg_s - ecg_s.mean()) / (ecg_s.std() + 1e-8)
    ppg_n = (ppg_s - ppg_s.mean()) / (ppg_s.std() + 1e-8)
    xcorr = correlate(ppg_n, ecg_n, mode='full')
    lags  = np.arange(-(len(ecg_n)-1), len(ecg_n))
    # Only search within physiological range [0, max_lag_s]
    max_lag = int(max_lag_s * fs)
    mask = (lags >= 0) & (lags <= max_lag)
    if mask.sum() == 0:
        return 0
    best_lag = int(lags[mask][np.argmax(xcorr[mask])])
    return best_lag


def apply_h_phi(x: torch.Tensor, z_vec: torch.Tensor,
                fs: int = FS) -> torch.Tensor:
    z_vec = z_vec.squeeze()
    a = z_vec[1].abs(); b = z_vec[2]
    k_size = 61; half = k_size // 2
    t = torch.arange(-half, half+1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t**2 / (2*10.0**2)); k = k/k.sum()
    h = F.conv1d(x, k.view(1,1,-1), padding=half)
    tau_s = int(round(float(z_vec[0]) * fs))
    if tau_s > 0:
        h = torch.roll(h, shifts=tau_s, dims=-1)
        h[..., :tau_s] = 0.0
    return a * h + b


# ======================================================================
# Load ensemble
# ======================================================================

def load_ensemble(device):
    priors = []
    for k in range(10):
        p = CKPT_DIR / f"prior_k{k:02d}.pt"
        if not p.exists(): continue
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
        unet.load_state_dict(ckpt["model"]); unet.eval()
        prior = ECGDiffusionPrior(denoiser=unet, T=1000,
                                   pred_type="x0", device=device).to(device)
        priors.append(prior)
    print(f"  {len(priors)} priors loaded")
    return priors


# ======================================================================
# Load test windows — NOW SAVES TRUE ECG TOO
# ======================================================================

def load_test_windows(n_windows, seed=42):
    test_ids_path = CKPT_DIR / "vitaldb_test_ids.npy"
    stats_path    = CKPT_DIR / "vitaldb_stats.npy"

    test_ids = list(np.load(str(test_ids_path)).astype(int)) \
               if test_ids_path.exists() else []

    if stats_path.exists():
        stats = np.load(str(stats_path))
        ecg_mean, ecg_std = float(stats[0,0]), float(stats[0,1])
        ppg_mean, ppg_std = float(stats[1,0]), float(stats[1,1])
    else:
        ecg_mean, ecg_std = 0.0, 1.0
        ppg_mean, ppg_std = 0.0, 1.0

    print(f"  ECG norm: mean={ecg_mean:.4f} std={ecg_std:.4f}")
    print(f"  PPG norm: mean={ppg_mean:.4f} std={ppg_std:.4f}")

    windows = []
    # One window per patient with per-case lag correction
    # Exclusion list: anomalous or severely misaligned cases
    EXCLUDE_CASES = {24}  # r=1.000 anomaly (identical ECG/PPG signals)
    MAX_LAG_EXCLUDE = 2.0  # exclude cases with lag > 2s (irrecoverable sync error)

    for cid in test_ids:
        if cid in EXCLUDE_CASES:
            print(f"  case {cid}: excluded (known anomaly)")
            continue

        result = load_case_signals(cid)
        if result is None:
            print(f"  case {cid}: load failed, skipping")
            continue
        ecg_raw, ppg_raw = result
        if not is_physical_units(ecg_raw, ppg_raw):
            print(f"  case {cid}: ADC-scale, skipping")
            continue

        # Compute case-level lag correction from first 30s
        lag_samples = compute_case_lag(ecg_raw, ppg_raw, FS)
        lag_s = lag_samples / FS

        if lag_s > MAX_LAG_EXCLUDE:
            print(f"  case {cid}: lag={lag_s:.2f}s > {MAX_LAG_EXCLUDE}s, excluding")
            continue

        # Apply lag correction: shift PPG back by lag_samples
        # so that PPG[t] aligns with ECG[t]
        if lag_samples > 0:
            ppg_corrected = np.roll(ppg_raw, -lag_samples)
            ppg_corrected[-lag_samples:] = np.nan
        else:
            ppg_corrected = ppg_raw.copy()

        print(f"  case {cid}: lag={lag_s:.3f}s ({lag_samples} samples) corrected")

        # Pick middle window
        total_wins = (len(ecg_raw) - WINDOW_SAMPLES) // WINDOW_SAMPLES
        mid = max(0, total_wins // 2)
        start = mid * WINDOW_SAMPLES

        ecg_w = ecg_raw[start:start+WINDOW_SAMPLES]
        ppg_w = ppg_corrected[start:start+WINDOW_SAMPLES]

        if not qc_paired_window(ecg_w, ppg_w):
            found = False
            for s in range(0, len(ecg_raw)-WINDOW_SAMPLES+1, WINDOW_SAMPLES):
                ew = ecg_raw[s:s+WINDOW_SAMPLES]
                pw = ppg_corrected[s:s+WINDOW_SAMPLES]
                if qc_paired_window(ew, pw):
                    ecg_w, ppg_w = ew, pw
                    found = True
                    break
            if not found:
                print(f"  case {cid}: no valid window after lag correction, skipping")
                continue

        ecg_n = (ecg_w - ecg_mean) / max(ecg_std, 1e-8)
        ppg_n = (ppg_w - ppg_mean) / max(ppg_std, 1e-8)
        windows.append((ecg_n, ppg_n, ecg_w, cid))

    print(f"  Loaded {len(windows)} test windows")
    return windows


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--n-inner",   type=int, default=100)
    ap.add_argument("--burn-in",   type=int, default=20)
    ap.add_argument("--gamma",     type=float, default=1.0)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[setup] loading ensemble...")
    priors = load_ensemble(device)

    lam_path = CKPT_DIR / "lambda_star_vitaldb.pt"
    if not lam_path.exists():
        raise FileNotFoundError("Run train_vi_vitaldb.py first.")
    q_lambda = load_lambda(lam_path, device)
    log_sigma2 = float(q_lambda.mu_lambda[0])
    print(f"  log_sigma2={log_sigma2:.4f} sigma={math.exp(log_sigma2/2):.4f}")

    diff = GaussianDiffusion(T=1000, schedule="cosine",
                              pred_type="x0").to(device)
    lik  = GaussianLikelihood(obs_dim=4000, fs=float(FS),
                               cov_mode=CovMode.ISOTROPIC,
                               kernel_sigma=10.0, kernel_size=61).to(device)

    t_anneal = list(range(500,0,-10)) + [1]
    cfg = MALAConfig(n_inner=args.n_inner, burn_in=args.burn_in,
                     step_size_z=1e-3, step_size_tau=0.02,
                     t_anneal=t_anneal,
                     gamma_t={t: args.gamma for t in t_anneal})

    # Load VitalDB Z stats
    z_stats_path = CKPT_DIR / "vitaldb_z_stats.npy"
    if z_stats_path.exists():
        zs = np.load(str(z_stats_path), allow_pickle=True).item()
        log_z_prior = make_z_prior(
            tau_mu=zs['tau_mean'], tau_sigma=max(zs['tau_std'], 0.05),
            a_mu=zs['a_mean'],     a_sigma=max(zs['a_std'], 0.20),
            b_mu=zs['b_mean'],     b_sigma=max(zs['b_std'], 0.10))
    else:
        log_z_prior = make_z_prior(tau_mu=0.25, tau_sigma=0.10,
                                    a_mu=0.85, a_sigma=0.40,
                                    b_mu=0.03, b_sigma=0.20)

    print("\n[data] loading VitalDB test windows...")
    windows = load_test_windows(args.n_windows, seed=args.seed)

    # Storage
    ppg_errs, uncertainties = [], []
    ecg_rmses, ecg_corrs   = [], []
    hr_post, hr_true        = [], []
    rr_post, rr_true        = [], []
    z_estimates             = []
    posterior_means         = []
    true_ecgs               = []

    class SimpleEnsemble:
        def __init__(self, p): self.priors=p; self.B=len(p)
    ensemble = SimpleEnsemble(priors)

    print(f"\n{'Win':>4} {'PPG_err':>9} {'ECG_RMSE':>10} "
          f"{'ECG_corr':>10} {'HR_post':>9} {'HR_true':>9} "
          f"{'HR_MAE':>8} {'Unc':>8}")
    print("-"*85)

    for i, (ecg_np, ppg_np, ecg_raw, cid) in enumerate(windows):
        t0 = time.time()
        y_t = torch.tensor(ppg_np, dtype=torch.float32).view(1,1,4000).to(device)

        sample_sets, weights, _ = run_inference(
            y=y_t, ensemble=ensemble, q_lambda_star=q_lambda,
            likelihood=lik, diffusion=diff, phi_to_params_fn=phi_to_params,
            phi_dim=1, device=device, cfg=cfg, log_z_prior=log_z_prior,
            verbose=False, fix_log_sigma2=log_sigma2)

        # Collect samples
        all_x, all_z, all_w = [], [], []
        for k, ss in enumerate(sample_sets):
            w_k = float(weights[k])
            for r in range(ss.n_samples):
                all_x.append(ss.x_samples[r].squeeze())
                all_z.append(ss.z_samples[r].squeeze())
                all_w.append(w_k / ss.n_samples)

        xs_np = torch.stack(all_x).numpy()
        post_mean = xs_np.mean(axis=0)
        posterior_means.append(post_mean)
        true_ecgs.append(ecg_np)

        # ECG RMSE and correlation vs true ECG
        ecg_rmse = float(np.sqrt(np.mean((post_mean - ecg_np)**2)))
        ecg_corr = float(np.corrcoef(post_mean, ecg_np)[0,1])
        ecg_rmses.append(ecg_rmse)
        ecg_corrs.append(ecg_corr)

        # PPG consistency
        ppg_win_errs = []
        for j in range(len(all_x)):
            x_j = all_x[j].view(1,1,4000).to(device)
            z_j = all_z[j].to(device)
            with torch.no_grad():
                ppg_pred = apply_h_phi(x_j, z_j).squeeze().cpu().numpy()
            ppg_win_errs.append(float(np.abs(ppg_pred - ppg_np).mean()))
        ppg_err = float(np.mean(ppg_win_errs))
        ppg_errs.append(ppg_err)

        # Uncertainty
        unc = float(xs_np.std(axis=0).mean())
        uncertainties.append(unc)

        # Clinical quantities
        clin = posterior_clinical(all_x)
        hr_post.append(clin['hr_mean'])
        rr_post.append(clin['rr_mean'])

        # True ECG clinical quantities (from raw ECG)
        hr_t = estimate_heart_rate(ecg_np)
        rr_t = estimate_rr_interval(ecg_np)
        hr_true.append(hr_t)
        rr_true.append(rr_t)

        hr_mae = abs(clin['hr_mean'] - hr_t) if not math.isnan(hr_t) \
                 and not math.isnan(clin['hr_mean']) else float('nan')

        # Z estimate
        z_mean = np.mean([z.numpy() for z in all_z], axis=0)
        z_estimates.append(z_mean)

        elapsed = time.time() - t0
        print(f"{i+1:>4}  {ppg_err:>9.4f}  {ecg_rmse:>10.4f}  "
              f"{ecg_corr:>10.4f}  {clin['hr_mean']:>9.1f}  "
              f"{hr_t:>9.1f}  {hr_mae:>8.1f}  {unc:>8.4f}  "
              f"[case {cid}]")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VitalDB held-out evaluation summary")
    print(f"{'='*60}")

    hr_maes = [abs(h-t) for h,t in zip(hr_post, hr_true)
               if not math.isnan(h) and not math.isnan(t)]
    hr_pairs = [(h,t) for h,t in zip(hr_post, hr_true)
                if not math.isnan(h) and not math.isnan(t)]

    rr_maes = [abs(h-t) for h,t in zip(rr_post, rr_true)
               if not math.isnan(h) and not math.isnan(t)]

    print(f"PPG consistency:   {np.nanmean(ppg_errs):.4f} ± {np.nanstd(ppg_errs):.4f}")
    print(f"ECG RMSE:          {np.nanmean(ecg_rmses):.4f} ± {np.nanstd(ecg_rmses):.4f}")
    print(f"ECG correlation:   {np.nanmean(ecg_corrs):.4f} ± {np.nanstd(ecg_corrs):.4f}")
    print(f"HR MAE (bpm):      {np.mean(hr_maes):.2f} ± {np.std(hr_maes):.2f}")
    if len(hr_pairs) >= 3:
        hr_r, hr_p = pearsonr([h for h,_ in hr_pairs], [t for _,t in hr_pairs])
        print(f"HR correlation:    r={hr_r:.3f} p={hr_p:.3f}")
    if len(rr_maes) >= 3:
        print(f"RR MAE (s):        {np.mean(rr_maes):.4f} ± {np.std(rr_maes):.4f}")

    # Spearman correlation: uncertainty vs PPG error
    if len(ppg_errs) >= 3:
        sp_r, sp_p = spearmanr(uncertainties, ppg_errs)
        print(f"Uncertainty vs PPG err (Spearman): r={sp_r:.3f} p={sp_p:.3f}")

    z_arr = np.array(z_estimates)
    print(f"tau (s):           {z_arr[:,0].mean():.3f} ± {z_arr[:,0].std():.3f}")
    print(f"a:                 {z_arr[:,1].mean():.3f} ± {z_arr[:,1].std():.3f}")
    print(f"b:                 {z_arr[:,2].mean():.3f} ± {z_arr[:,2].std():.3f}")

    # ── Save results ─────────────────────────────────────────────────
    np.savez(str(OUT_DIR / "vitaldb_results.npz"),
             ppg_consistency = np.array(ppg_errs),
             uncertainty     = np.array(uncertainties),
             ecg_rmse        = np.array(ecg_rmses),
             ecg_corr        = np.array(ecg_corrs),
             hr_post         = np.array(hr_post),
             hr_true         = np.array(hr_true),
             rr_post         = np.array(rr_post),
             rr_true         = np.array(rr_true),
             z_estimates     = z_arr,
             posterior_means = np.array(posterior_means),
             true_ecgs       = np.array(true_ecgs))
    print(f"\n→ saved vitaldb_results.npz")

    # ── Figures ──────────────────────────────────────────────────────
    # Fig 1: Uncertainty vs PPG error scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.scatter(uncertainties, ppg_errs, color='steelblue', alpha=0.7, s=60)
    for j, (u, p) in enumerate(zip(uncertainties, ppg_errs)):
        if p > 0.1:
            ax.annotate(f"win {j+1}", (u, p), textcoords='offset points',
                        xytext=(5, 3), fontsize=7, color='red')
    if len(ppg_errs) >= 3:
        sp_r, sp_p = spearmanr(uncertainties, ppg_errs)
        ax.set_title(f"Uncertainty vs PPG error — VitalDB\n"
                     f"Spearman r={sp_r:.3f}, p={sp_p:.3f}")
    ax.set_xlabel("Posterior std (uncertainty)")
    ax.set_ylabel("PPG consistency MAE")

    # Fig 2: HR posterior vs true
    ax2 = axes[1]
    hr_p_arr = np.array(hr_post); hr_t_arr = np.array(hr_true)
    valid = ~(np.isnan(hr_p_arr) | np.isnan(hr_t_arr))
    if valid.sum() >= 2:
        ax2.scatter(hr_t_arr[valid], hr_p_arr[valid],
                    color='darkorange', alpha=0.7, s=60)
        lim = [min(hr_t_arr[valid].min(), hr_p_arr[valid].min()) - 5,
               max(hr_t_arr[valid].max(), hr_p_arr[valid].max()) + 5]
        ax2.plot(lim, lim, 'k--', lw=1, label='Ideal')
        if valid.sum() >= 3:
            hr_r, hr_p = pearsonr(hr_t_arr[valid], hr_p_arr[valid])
            ax2.set_title(f"HR: posterior vs true ECG — VitalDB\n"
                          f"r={hr_r:.3f}, MAE={np.mean(hr_maes):.1f} bpm")
        ax2.set_xlabel("True HR from ECG (bpm)")
        ax2.set_ylabel("Posterior HR estimate (bpm)")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "vitaldb_analysis.png"), dpi=150)
    plt.close(fig)
    print(f"→ saved vitaldb_analysis.png")

    # Fig 3: Inspect the 3 high-error windows
    high_err_idx = [j for j, p in enumerate(ppg_errs) if p > 0.1]
    if high_err_idx:
        fig, axes2 = plt.subplots(len(high_err_idx), 2,
                                   figsize=(12, 4*len(high_err_idx)))
        if len(high_err_idx) == 1:
            axes2 = axes2.reshape(1, 2)
        t_ax = np.arange(4000) / FS
        for row, j in enumerate(high_err_idx):
            axes2[row, 0].plot(t_ax, true_ecgs[j], 'k', lw=0.6,
                               label='True ECG')
            axes2[row, 0].plot(t_ax, posterior_means[j], 'steelblue',
                               lw=0.6, alpha=0.8, label='Posterior mean')
            axes2[row, 0].set_title(f"Window {j+1} — ECG "
                                     f"(PPG err={ppg_errs[j]:.3f})")
            axes2[row, 0].legend(fontsize=7)
            axes2[row, 0].set_xlabel("Time (s)")

            axes2[row, 1].plot(t_ax, windows[j][1], 'darkorange',
                               lw=0.6, label='Observed PPG')
            axes2[row, 1].set_title(f"Window {j+1} — PPG "
                                     f"(unc={uncertainties[j]:.3f})")
            axes2[row, 1].legend(fontsize=7)
            axes2[row, 1].set_xlabel("Time (s)")

        fig.suptitle("High-error windows inspection — VitalDB", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(OUT_DIR / "vitaldb_outliers.png"), dpi=150)
        plt.close(fig)
        print(f"→ saved vitaldb_outliers.png")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
