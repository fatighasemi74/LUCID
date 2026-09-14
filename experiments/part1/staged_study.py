"""
experiments/part1/staged_study.py
===================================
Professor's Tasks 3 and 4:

Task 3: Repeat staged check over multiple windows/seeds.
        Report mean ± std across windows.

Task 4: Convergence study with N in {5, 10, 20, 50} on same windows.
        Check that posterior summaries stabilize before fixing N.

Usage
-----
python experiments/part1/staged_study.py
"""

import sys, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet1d import UNet1D
from models.diffusion_core import GaussianDiffusion
from models.prior import ECGDiffusionPrior
from models.likelihood import GaussianLikelihood
from models.noise_cov import CovMode
from inference.vi import MeanFieldGaussian, load_lambda
from inference.mala import MCMCState, MALAConfig, phi_mh_step, z_mala_step, x_diffusion_step, _set_likelihood_params
from inference.sampler import make_gaussian_z_prior

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
DATA_DIR = ROOT / "experiments/part1/synthetic_data"
OUT_DIR  = ROOT / "experiments/part1/staged_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
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

def load_prior(k, device):
    path = CKPT_DIR / f"prior_k{k:02d}.pt"
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
    unet.load_state_dict(ckpt["model"]); unet.eval()
    return ECGDiffusionPrior(denoiser=unet, T=1000, pred_type="x0",
                             device=device).to(device)

def run_mcmc(y, prior, lik, diff, cfg, device, q_lambda,
             infer_phi, infer_z, phi_fixed=None, z_fixed=None):
    phi0 = phi_fixed.clone().to(device) if phi_fixed is not None \
           else q_lambda.sample(1).squeeze(0).detach()
    z0   = z_fixed.clone().to(device) if z_fixed is not None \
           else torch.tensor([[0.2]], dtype=torch.float32, device=device)
    _set_likelihood_params(lik, phi_to_params(phi0))
    log_z_prior = make_gaussian_z_prior(mu=0.2, sigma=0.1)

    def prior_score_fn(x_t, t_batch):
        return prior.denoiser(x_t, t_batch)

    x_samples, z_samples, phi_samples = [], [], []
    state = MCMCState(x=torch.randn(1,1,4000,device=device),
                      z=z0, phi=phi0, log_lik=-1e6, step=0)

    for r in range(cfg.n_inner):
        if infer_phi:
            state, _ = phi_mh_step(state, y, lik, q_lambda, phi_to_params, device)
        else:
            _set_likelihood_params(lik, phi_to_params(phi0))
        if infer_z:
            state, _ = z_mala_step(state, y, lik, log_z_prior, cfg, device)
        state = x_diffusion_step(state, y, prior_score_fn, lik, diff, cfg, device)
        state = MCMCState(x=state.x, z=state.z, phi=state.phi,
                          log_lik=state.log_lik, step=r+1,
                          n_acc_phi=state.n_acc_phi, n_acc_z=state.n_acc_z)
        if r >= cfg.burn_in:
            x_samples.append(state.x.squeeze().cpu())
            z_samples.append(state.z.cpu())
            phi_samples.append(state.phi.cpu())

    return x_samples, z_samples, phi_samples

def evaluate_samples(x_samples, phi_samples, x_true_np, y_obs_np, device):
    xs    = torch.stack(x_samples).numpy()
    mean  = xs.mean(axis=0)
    std   = xs.std(axis=0)
    rmse  = float(np.sqrt(((mean - x_true_np)**2).mean()))
    corr  = float(np.corrcoef(mean, x_true_np)[0,1])
    unc   = float(std.mean())
    # Coverage at 95%
    from scipy import stats
    z95 = stats.norm.ppf(0.975)
    cov95 = float((np.abs(x_true_np - mean) <= z95*std).mean())
    # PPG consistency: matched (x_i, phi_i) pairs
    ppg_errs = []
    for i in range(len(x_samples)):
        x_i   = x_samples[i].view(1,1,4000).to(device)
        phi_i = phi_samples[i].to(device)
        with torch.no_grad():
            ppg_i = apply_h_phi(x_i, phi_i).squeeze().cpu().numpy()
        ppg_errs.append(float(np.abs(ppg_i - y_obs_np).mean()))
    ppg_err = float(np.mean(ppg_errs))
    return dict(rmse=rmse, corr=corr, unc=unc, cov95=cov95, ppg_err=ppg_err)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=5,
                    help="windows for multi-window study (Task 3)")
    ap.add_argument("--snr-db", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  SNR={args.snr_db}dB  gamma={args.gamma}\n")

    # Load data
    data   = np.load(str(DATA_DIR/f"snr_{args.snr_db:02d}dB.npz"))
    ecg_all = data["ecg"]
    ppg_all = data["ppg"]
    ns      = float(data["noise_sigma"])

    # Load models (one prior for staged check)
    prior = load_prior(0, device)
    diff  = GaussianDiffusion(T=1000, schedule="cosine", pred_type="x0").to(device)
    lik   = GaussianLikelihood(obs_dim=4000, fs=500.0,
                               cov_mode=CovMode.ISOTROPIC,
                               kernel_sigma=10.0, kernel_size=61).to(device)
    params = phi_to_params(PHI_TRUE.to(device))
    params["log_diag"] = torch.tensor([math.log(ns**2)], device=device)
    _set_likelihood_params(lik, params)
    q_lambda = load_lambda(CKPT_DIR/"lambda_star.pt", device)

    gamma_schedule = {t: args.gamma for t in list(range(500, 0, -10)) + [1]}

    # ================================================================
    # TASK 4: Convergence study N in {5, 10, 20, 50} — ONE window
    # ================================================================
    print("="*60)
    print("TASK 4: Convergence study — N in {5, 10, 20, 50}")
    print("="*60)

    win_idx = 0
    x_true_np = ecg_all[win_idx]
    y_obs_np  = ppg_all[win_idx]
    y_t = torch.tensor(y_obs_np).view(1,1,4000).to(device)

    print(f"Window {win_idx}: ECG std={x_true_np.std():.3f}  "
          f"PPG std={y_obs_np.std():.3f}\n")
    print(f"{'N':>5} {'burn':>5} {'RMSE':>8} {'Corr':>8} "
          f"{'Unc':>8} {'Cov95':>8} {'PPG_err':>10}")
    print("-"*55)

    for n_inner in [5, 10, 20, 50]:
        burn_in = max(1, n_inner // 5)
        cfg = MALAConfig(n_inner=n_inner, burn_in=burn_in,
                         step_size_z=1e-3,
                         t_anneal=list(range(500, 0, -10)) + [1],
                         gamma_t=gamma_schedule)
        x_s, z_s, phi_s = run_mcmc(y_t, prior, lik, diff, cfg, device,
                                    q_lambda, infer_phi=False, infer_z=False,
                                    phi_fixed=PHI_TRUE.to(device),
                                    z_fixed=Z_TRUE.to(device))
        m = evaluate_samples(x_s, phi_s, x_true_np, y_obs_np, device)
        print(f"{n_inner:5d} {burn_in:5d} {m['rmse']:8.4f} {m['corr']:8.4f} "
              f"{m['unc']:8.4f} {m['cov95']*100:7.1f}% {m['ppg_err']:10.4f}")

    # ================================================================
    # TASK 3: Multi-window staged check — mean ± std
    # ================================================================
    print()
    print("="*60)
    print(f"TASK 3: Multi-window staged check ({args.n_windows} windows)")
    print("="*60)

    N_INNER = 20
    BURN_IN = 4
    cfg = MALAConfig(n_inner=N_INNER, burn_in=BURN_IN,
                     step_size_z=1e-3,
                     t_anneal=list(range(500, 0, -10)) + [1],
                     gamma_t=gamma_schedule)

    stage_results = {f"Stage {i}": [] for i in range(1, 5)}

    for win_idx in range(args.n_windows):
        x_true_np = ecg_all[win_idx]
        y_obs_np  = ppg_all[win_idx]
        y_t = torch.tensor(y_obs_np).view(1,1,4000).to(device)

        # Stage 1: X only, oracle Z and phi
        x_s, z_s, phi_s = run_mcmc(y_t, prior, lik, diff, cfg, device,
                                    q_lambda, infer_phi=False, infer_z=False,
                                    phi_fixed=PHI_TRUE.to(device),
                                    z_fixed=Z_TRUE.to(device))
        stage_results["Stage 1"].append(
            evaluate_samples(x_s, phi_s, x_true_np, y_obs_np, device))

        # Stage 2: X and Z, oracle phi
        x_s, z_s, phi_s = run_mcmc(y_t, prior, lik, diff, cfg, device,
                                    q_lambda, infer_phi=False, infer_z=True,
                                    phi_fixed=PHI_TRUE.to(device))
        stage_results["Stage 2"].append(
            evaluate_samples(x_s, phi_s, x_true_np, y_obs_np, device))

        # Stage 3: X, Z, and phi
        x_s, z_s, phi_s = run_mcmc(y_t, prior, lik, diff, cfg, device,
                                    q_lambda, infer_phi=True, infer_z=True)
        stage_results["Stage 3"].append(
            evaluate_samples(x_s, phi_s, x_true_np, y_obs_np, device))

        # Stage 4: same as Stage 3 with K=1 (should be identical)
        x_s, z_s, phi_s = run_mcmc(y_t, prior, lik, diff, cfg, device,
                                    q_lambda, infer_phi=True, infer_z=True)
        stage_results["Stage 4"].append(
            evaluate_samples(x_s, phi_s, x_true_np, y_obs_np, device))

        print(f"  window {win_idx+1}/{args.n_windows} done")

    print()
    print(f"{'Stage':<12} {'RMSE':>12} {'Corr':>12} {'Unc':>10} "
          f"{'Cov95':>10} {'PPG_err':>12}")
    print("-"*65)
    for stage, rows in stage_results.items():
        rmse  = np.array([r['rmse']    for r in rows])
        corr  = np.array([r['corr']    for r in rows])
        unc   = np.array([r['unc']     for r in rows])
        cov95 = np.array([r['cov95']   for r in rows])
        ppg   = np.array([r['ppg_err'] for r in rows])
        print(f"{stage:<12} "
              f"{rmse.mean():6.3f}±{rmse.std():.3f}  "
              f"{corr.mean():6.3f}±{corr.std():.3f}  "
              f"{unc.mean():6.3f}±{unc.std():.3f}  "
              f"{cov95.mean()*100:5.1f}%  "
              f"{ppg.mean():6.3f}±{ppg.std():.3f}")

    print(f"\nNote: Stage 3 and Stage 4 both use K=1, infer_phi=True, infer_z=True.")
    print(f"They should agree closely — any difference is sampling noise.")

if __name__ == "__main__":
    main()
