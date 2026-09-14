"""
experiments/part1/convergence_and_phi.py
==========================================
Professor's two checks before moving to K=10:

TASK A — N convergence study with N in {50, 100, 200}
  Run Stage 1 (oracle Z, phi) on a fixed set of windows.
  Report RMSE, correlation, uncertainty per N.
  Determine where posterior summaries stabilize.

TASK B — phi MH acceptance rate + PPG error distribution
  Run Stage 3 (infer Z and phi) with N=50.
  Report phi MH acceptance rate.
  Report distribution of ||y - H_phi^(m)(x^(m), z^(m))|| per matched sample.
  Distinguish genuine forward-model uncertainty from insufficient phi mixing.

Usage
-----
python experiments/part1/convergence_and_phi.py --n-windows 5
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
OUT_DIR  = ROOT / "experiments/part1/convergence_and_phi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS      = 500
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
    k = torch.exp(-t**2/(2*10.0**2)); k = k/k.sum()
    h = F.conv1d(x, k.view(1,1,-1), padding=half)
    tau_s = int(round(float(phi_vec[0])*fs))
    if tau_s > 0:
        h = torch.roll(h, shifts=tau_s, dims=-1)
        h[..., :tau_s] = 0.0
    return a*h + b


def load_prior(device):
    ckpt = torch.load(str(CKPT_DIR/"prior_k00.pt"),
                      map_location=device, weights_only=False)
    unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
    unet.load_state_dict(ckpt["model"]); unet.eval()
    return ECGDiffusionPrior(denoiser=unet, T=1000, pred_type="x0",
                             device=device).to(device)


def run_mcmc_tracked(y, prior, lik, diff, cfg, device, q_lambda,
                     infer_phi, infer_z,
                     phi_fixed=None, z_fixed=None):
    """
    Run MCMC and return samples + diagnostics.
    Returns x_samples, phi_samples, z_samples, n_acc_phi, n_acc_z
    """
    phi0 = phi_fixed.clone().to(device) if phi_fixed is not None \
           else q_lambda.sample(1).squeeze(0).detach()
    z0   = z_fixed.clone().to(device) if z_fixed is not None \
           else torch.tensor([[0.2]], dtype=torch.float32, device=device)

    _set_likelihood_params(lik, phi_to_params(phi0))
    log_z_prior = make_gaussian_z_prior(mu=0.2, sigma=0.1)

    def score_fn(x_t, t_b):
        return prior.denoiser(x_t, t_b)

    state = MCMCState(x=torch.randn(1,1,4000,device=device),
                      z=z0, phi=phi0, log_lik=-1e6, step=0)

    x_samples, phi_samples, z_samples = [], [], []
    n_acc_phi_total = 0
    n_phi_proposals = 0

    for r in range(cfg.n_inner):
        if infer_phi:
            state, acc_phi = phi_mh_step(
                state, y, lik, q_lambda, phi_to_params, device)
            n_phi_proposals += 1
            if acc_phi:
                n_acc_phi_total += 1
        else:
            _set_likelihood_params(lik, phi_to_params(phi0))

        if infer_z:
            state, _ = z_mala_step(state, y, lik, log_z_prior, cfg, device)

        state = x_diffusion_step(state, y, score_fn, lik, diff, cfg, device)
        state = MCMCState(x=state.x, z=state.z, phi=state.phi,
                          log_lik=state.log_lik, step=r+1,
                          n_acc_phi=state.n_acc_phi,
                          n_acc_z=state.n_acc_z)

        if r >= cfg.burn_in:
            x_samples.append(state.x.squeeze().cpu())
            phi_samples.append(state.phi.cpu())
            z_samples.append(state.z.cpu())

    acc_rate_phi = n_acc_phi_total / max(1, n_phi_proposals)
    return x_samples, phi_samples, z_samples, acc_rate_phi


def evaluate(x_samples, phi_samples, z_samples,
             x_true_np, y_obs_np, device):
    xs   = torch.stack(x_samples).numpy()
    mean = xs.mean(axis=0)
    std  = xs.std(axis=0)
    rmse = float(np.sqrt(((mean - x_true_np)**2).mean()))
    corr = float(np.corrcoef(mean, x_true_np)[0,1])
    unc  = float(std.mean())

    # Per-sample PPG error using matched (x_i, phi_i, z_i)
    ppg_errs = []
    for i in range(len(x_samples)):
        x_i   = x_samples[i].view(1,1,4000).to(device)
        phi_i = phi_samples[i].to(device)
        with torch.no_grad():
            ppg_i = apply_h_phi(x_i, phi_i).squeeze().cpu().numpy()
        ppg_errs.append(float(np.abs(ppg_i - y_obs_np).mean()))

    return dict(rmse=rmse, corr=corr, unc=unc,
                ppg_err=float(np.mean(ppg_errs)),
                ppg_errs=ppg_errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=5)
    ap.add_argument("--snr-db",    type=int, default=20)
    ap.add_argument("--gamma",     type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  SNR={args.snr_db}dB  gamma={args.gamma}\n")

    data     = np.load(str(DATA_DIR/f"snr_{args.snr_db:02d}dB.npz"))
    ecg_all  = data["ecg"]
    ppg_all  = data["ppg"]
    ns       = float(data["noise_sigma"])

    prior    = load_prior(device)
    diff     = GaussianDiffusion(T=1000, schedule="cosine",
                                  pred_type="x0").to(device)
    lik      = GaussianLikelihood(obs_dim=4000, fs=500.0,
                                   cov_mode=CovMode.ISOTROPIC,
                                   kernel_sigma=10.0,
                                   kernel_size=61).to(device)
    params   = phi_to_params(PHI_TRUE.to(device))
    params["log_diag"] = torch.tensor([math.log(ns**2)], device=device)
    _set_likelihood_params(lik, params)

    q_lambda = load_lambda(CKPT_DIR/"lambda_star.pt", device)
    gamma_t  = {t: args.gamma for t in list(range(500, 0, -10)) + [1]}

    # ================================================================
    # TASK A: N convergence {50, 100, 200} — Stage 1 (oracle Z, phi)
    # ================================================================
    print("="*60)
    print("TASK A: N convergence — Stage 1 (oracle Z, phi)")
    print(f"Windows: {args.n_windows}  SNR={args.snr_db}dB")
    print("="*60)
    print(f"{'N':>6} {'B':>5} {'RMSE':>10} {'Corr':>10} "
          f"{'Unc':>10} {'PPG_err':>10}")
    print("-"*55)

    conv_results = {}
    windows = [(ecg_all[i], ppg_all[i]) for i in range(args.n_windows)]

    for n_inner in [50, 100, 200]:
        burn_in = n_inner // 5
        cfg = MALAConfig(n_inner=n_inner, burn_in=burn_in,
                         step_size_z=1e-3,
                         t_anneal=list(range(500, 0, -10)) + [1],
                         gamma_t=gamma_t)
        rmse_list, corr_list, unc_list, ppg_list = [], [], [], []

        for win_idx, (x_true_np, y_obs_np) in enumerate(windows):
            y_t = torch.tensor(y_obs_np).view(1,1,4000).to(device)
            x_s, phi_s, z_s, _ = run_mcmc_tracked(
                y_t, prior, lik, diff, cfg, device, q_lambda,
                infer_phi=False, infer_z=False,
                phi_fixed=PHI_TRUE.to(device),
                z_fixed=Z_TRUE.to(device))
            m = evaluate(x_s, phi_s, z_s, x_true_np, y_obs_np, device)
            rmse_list.append(m["rmse"])
            corr_list.append(m["corr"])
            unc_list.append(m["unc"])
            ppg_list.append(m["ppg_err"])

        conv_results[n_inner] = dict(
            rmse=np.array(rmse_list),
            corr=np.array(corr_list),
            unc =np.array(unc_list),
            ppg =np.array(ppg_list),
        )
        print(f"{n_inner:6d} {burn_in:5d} "
              f"{np.mean(rmse_list):>10.4f}±{np.std(rmse_list):.4f}  "
              f"{np.mean(corr_list):>10.4f}±{np.std(corr_list):.4f}  "
              f"{np.mean(unc_list):>10.4f}  "
              f"{np.mean(ppg_list):>10.4f}")

    # Convergence plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    Ns = [50, 100, 200]
    for ax, key, label in zip(axes,
                               ["rmse", "corr", "unc"],
                               ["RMSE", "Correlation", "Uncertainty (std)"]):
        means = [conv_results[n][key].mean() for n in Ns]
        stds  = [conv_results[n][key].std()  for n in Ns]
        ax.errorbar(Ns, means, yerr=stds, marker='o', capsize=4,
                    color="steelblue")
        ax.set_xlabel("N (inner steps)"); ax.set_ylabel(label)
        ax.set_title(f"{label} vs N")
        ax.set_xticks(Ns)
    fig.suptitle(f"Stage 1 convergence — {args.n_windows} windows, "
                 f"SNR={args.snr_db}dB", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR/"convergence_N.png"), dpi=120)
    plt.close(fig)
    print(f"\n→ saved convergence_N.png")

    # ================================================================
    # TASK B: phi MH acceptance + PPG error distribution (Stage 3)
    # ================================================================
    print()
    print("="*60)
    print("TASK B: phi MH acceptance + PPG error distribution")
    print("Stage 3: infer X, Z, phi  (N=50)")
    print("="*60)

    cfg_b = MALAConfig(n_inner=50, burn_in=10,
                       step_size_z=1e-3,
                       t_anneal=list(range(500, 0, -10)) + [1],
                       gamma_t=gamma_t)

    all_ppg_stage1 = []
    all_ppg_stage3 = []
    all_acc_phi    = []
    all_phi_samples = []

    for win_idx, (x_true_np, y_obs_np) in enumerate(windows):
        y_t = torch.tensor(y_obs_np).view(1,1,4000).to(device)

        # Stage 1 PPG errors (oracle phi) for comparison
        x_s1, phi_s1, z_s1, _ = run_mcmc_tracked(
            y_t, prior, lik, diff, cfg_b, device, q_lambda,
            infer_phi=False, infer_z=False,
            phi_fixed=PHI_TRUE.to(device), z_fixed=Z_TRUE.to(device))
        m1 = evaluate(x_s1, phi_s1, z_s1, x_true_np, y_obs_np, device)
        all_ppg_stage1.extend(m1["ppg_errs"])

        # Stage 3: infer phi
        x_s3, phi_s3, z_s3, acc = run_mcmc_tracked(
            y_t, prior, lik, diff, cfg_b, device, q_lambda,
            infer_phi=True, infer_z=True)
        m3 = evaluate(x_s3, phi_s3, z_s3, x_true_np, y_obs_np, device)
        all_ppg_stage3.extend(m3["ppg_errs"])
        all_acc_phi.append(acc)
        all_phi_samples.extend(phi_s3)

        print(f"  Window {win_idx}: acc_phi={acc:.3f}  "
              f"ppg_err_s1={np.mean(m1['ppg_errs']):.4f}  "
              f"ppg_err_s3={np.mean(m3['ppg_errs']):.4f}  "
              f"RMSE_s3={m3['rmse']:.4f}  corr_s3={m3['corr']:.4f}")

    mean_acc = float(np.mean(all_acc_phi))
    print(f"\n  Mean phi MH acceptance rate: {mean_acc:.4f}")
    print(f"  (ideal range: 0.20-0.40 for MH with q_lambda* proposal)")
    print()
    print(f"  PPG error distribution:")
    print(f"    Stage 1 (oracle phi): "
          f"mean={np.mean(all_ppg_stage1):.4f}  "
          f"std={np.std(all_ppg_stage1):.4f}  "
          f"median={np.median(all_ppg_stage1):.4f}")
    print(f"    Stage 3 (infer phi):  "
          f"mean={np.mean(all_ppg_stage3):.4f}  "
          f"std={np.std(all_ppg_stage3):.4f}  "
          f"median={np.median(all_ppg_stage3):.4f}")
    print()

    # Interpret acceptance rate
    if mean_acc < 0.05:
        print("  ⚠ Acceptance rate very low (<5%) — phi mixing is poor.")
        print("    Proposals from q_lambda* are too far from the posterior.")
        print("    The PPG error increase reflects insufficient phi mixing,")
        print("    not genuine forward-model uncertainty.")
    elif mean_acc < 0.20:
        print("  ⚠ Acceptance rate low (<20%) — phi mixing is moderate.")
        print("    Some phi samples are useful but chain mixes slowly.")
    else:
        print("  ✓ Acceptance rate reasonable (>20%).")
        print("    PPG error increase at Stage 3 reflects genuine")
        print("    forward-model uncertainty under phi inference.")

    # Phi sample distribution
    phi_arr = torch.stack(all_phi_samples).numpy()
    phi_names = ["tau", "a", "b", "log_sigma2"]
    phi_true_vals = [0.2, 0.8, 0.05, -3.0]
    print()
    print(f"  Phi posterior (Stage 3):")
    print(f"  {'param':>12} {'true':>8} {'mean':>8} {'std':>8}")
    for j, (name, true_val) in enumerate(zip(phi_names, phi_true_vals)):
        print(f"  {name:>12} {true_val:>8.3f} "
              f"{phi_arr[:,j].mean():>8.3f} "
              f"{phi_arr[:,j].std():>8.3f}")

    # PPG error distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.hist(all_ppg_stage1, bins=20, alpha=0.6, color="steelblue",
            label=f"Stage 1 oracle φ\n(mean={np.mean(all_ppg_stage1):.3f})")
    ax.hist(all_ppg_stage3, bins=20, alpha=0.6, color="darkorange",
            label=f"Stage 3 infer φ\n(mean={np.mean(all_ppg_stage3):.3f})")
    ax.set_xlabel("||y - H_phi(x, z)|| per sample")
    ax.set_ylabel("Count")
    ax.set_title(f"PPG error distribution\nacc_phi={mean_acc:.3f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    for j, (name, true_val) in enumerate(zip(phi_names, phi_true_vals)):
        ax.violinplot(phi_arr[:,j], positions=[j], showmeans=True)
        ax.scatter([j], [true_val], color='red', zorder=5, s=50,
                   label="φ_true" if j==0 else "")
    ax.set_xticks(range(4))
    ax.set_xticklabels(phi_names, fontsize=8)
    ax.set_title("Phi posterior distribution\n(red = φ_true)")
    ax.legend(fontsize=8)

    fig.suptitle(f"Stage 3 phi diagnostics — {args.n_windows} windows, "
                 f"SNR={args.snr_db}dB", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR/"phi_diagnostics.png"), dpi=120)
    plt.close(fig)
    print(f"\n→ saved phi_diagnostics.png")
    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
