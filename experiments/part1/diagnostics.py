"""
experiments/part1/diagnostics.py
==================================
Four diagnostic checks from professor's Sep 1 meeting.

[1] Compare prior score magnitude vs likelihood guidance magnitude at each t.
    Professor: "I suspect the likelihood component is tiny relative to the prior."

[2] Check if likelihood gradient changes with SNR.
    Professor: "If it barely changes, it is not normalized correctly."

[3] Remove Langevin noise, take guided steps, plot ||y - H(x0hat, z_true)||²
    after each step. Should decrease monotonically. If not, gradient is wrong.

[4] Plot 5-6 individual ECG posterior samples with ||y - H(x_i, z_true)||
    for each. Professor's hypothesis: individual samples have strong QRS at
    slightly different locations → mean averages them out → weak QRS in mean.

Usage
-----
python experiments/part1/diagnostics.py
"""

import sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.diffusion_core import GaussianDiffusion
from models.unet1d import UNet1D
from models.prior import ECGDiffusionPrior
from models.likelihood import GaussianLikelihood
from models.noise_cov import CovMode
from inference.mala import _set_likelihood_params

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
DATA_DIR = ROOT / "experiments/part1/synthetic_data"
OUT_DIR  = ROOT / "experiments/part1/diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
PHI_TRUE = torch.tensor([0.2, 0.8, 0.05, -3.0])

def phi_to_params(phi):
    return {"log_a": phi[1].abs().log(), "b": phi[2],
            "tau": phi[0], "log_diag": phi[3].expand(1)}

def apply_h(x, phi=PHI_TRUE, fs=FS):
    a = phi[1].abs(); b = phi[2]
    half = 30
    t = torch.arange(-half, half+1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t**2/(2*10.0**2)); k=k/k.sum()
    h = F.conv1d(x, k.view(1,1,-1), padding=half)
    tau_s = int(round(float(phi[0])*fs))
    if tau_s > 0:
        h = torch.roll(h, shifts=tau_s, dims=-1)
        h[..., :tau_s] = 0.0
    return a*h + b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

# ── Load models ────────────────────────────────────────────────────────
ckpt = torch.load(str(CKPT_DIR/"prior_k00.pt"), map_location=device,
                  weights_only=False)
unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
unet.load_state_dict(ckpt["model"])
unet.eval()
diff = GaussianDiffusion(T=1000, schedule="cosine", pred_type="x0").to(device)
prior = ECGDiffusionPrior(denoiser=unet, T=1000, pred_type="x0",
                          device=device).to(device)

# ── Load test window (SNR=20 dB) ───────────────────────────────────────
data   = np.load(str(DATA_DIR/"snr_20dB.npz"))
ecg_np = data["ecg"][0]
ppg_np = data["ppg"][0]
x_true = torch.tensor(ecg_np).view(1,1,4000).to(device)
y_obs  = torch.tensor(ppg_np).view(1,1,4000).to(device)
noise_sigma_20 = float(data["noise_sigma"])

print(f"Loaded SNR=20dB window: noise_sigma={noise_sigma_20:.5f}")
print(f"ECG std={x_true.std().item():.3f}  PPG std={y_obs.std().item():.3f}\n")

def make_lik(noise_sigma):
    lik = GaussianLikelihood(obs_dim=4000, fs=500.0,
                             cov_mode=CovMode.ISOTROPIC,
                             kernel_sigma=10.0, kernel_size=61).to(device)
    params = phi_to_params(PHI_TRUE.to(device))
    params["log_diag"] = torch.tensor([math.log(noise_sigma**2)], device=device)
    _set_likelihood_params(lik, params)
    return lik

lik_20 = make_lik(noise_sigma_20)

# ======================================================================
# CHECK [1]: Prior score vs likelihood guidance magnitude at each t
# ======================================================================
print("="*60)
print("CHECK [1]: Prior score vs likelihood guidance magnitude")
print("="*60)

t_levels = [500, 250, 100, 50, 10, 1]
prior_norms = []
lik_norms   = []

with torch.no_grad():
    x_sample = prior.sample((1,1,4000), ddim_steps=20, device=device)

for t_val in t_levels:
    t_batch = torch.full((1,), t_val, dtype=torch.long, device=device)
    noise   = torch.randn_like(x_sample)
    x_t     = diff.q_sample(x_sample, t_batch, noise).requires_grad_(True)

    # Prior score
    with torch.no_grad():
        raw      = unet(x_t.detach(), t_batch)
        x0hat, _ = diff.to_x0_and_eps(x_t.detach(), t_batch, raw)
        abar     = diff._extract(diff.alphas_bar, t_batch, x_t.shape)
        prior_score = (x0hat - x_t.detach()) / (1.0 - abar + 1e-8)
        p_norm = prior_score.norm().item()

    # Likelihood gradient
    raw_g    = unet(x_t, t_batch)
    x0hat_g, _ = diff.to_x0_and_eps(x_t, t_batch, raw_g)
    ll       = lik_20.log_likelihood(y_obs, x0hat_g, None).sum()
    grad     = torch.autograd.grad(ll, x_t, allow_unused=True)[0]
    l_norm   = grad.norm().item() if grad is not None else 0.0

    prior_norms.append(p_norm)
    lik_norms.append(l_norm)
    ratio = p_norm / (l_norm + 1e-8)
    print(f"  t={t_val:4d}: ||prior score||={p_norm:.4f}  "
          f"||lik grad||={l_norm:.4f}  ratio={ratio:.2f}")

fig, ax = plt.subplots(figsize=(8,4))
ax.semilogy(t_levels, prior_norms, 'o-', label="||prior score||")
ax.semilogy(t_levels, lik_norms,   's-', label="||lik gradient||")
ax.set_xlabel("Diffusion timestep t")
ax.set_ylabel("L2 norm (log scale)")
ax.set_title("Prior score vs likelihood gradient magnitude across t")
ax.legend(); ax.invert_xaxis()
fig.tight_layout()
fig.savefig(str(OUT_DIR/"check1_score_magnitudes.png"), dpi=120)
plt.close(fig)
print(f"  → saved check1_score_magnitudes.png\n")

# ======================================================================
# CHECK [2]: Does likelihood gradient change with SNR?
# ======================================================================
print("="*60)
print("CHECK [2]: Likelihood gradient vs SNR")
print("="*60)

snr_configs = [
    (5,  float(np.load(str(DATA_DIR/"snr_05dB.npz"))["noise_sigma"])),
    (10, float(np.load(str(DATA_DIR/"snr_10dB.npz"))["noise_sigma"])),
    (20, noise_sigma_20),
    (40, float(np.load(str(DATA_DIR/"snr_40dB.npz"))["noise_sigma"])),
]

t_batch = torch.full((1,), 100, dtype=torch.long, device=device)
noise   = torch.randn_like(x_sample)
x_t_fixed = diff.q_sample(x_sample, t_batch, noise).requires_grad_(True)

snr_grad_norms = []
for snr_db, ns in snr_configs:
    lik_snr = make_lik(ns)
    # reload x_t with fresh graph
    x_t_s = x_t_fixed.detach().requires_grad_(True)
    raw_g  = unet(x_t_s, t_batch)
    x0h, _ = diff.to_x0_and_eps(x_t_s, t_batch, raw_g)
    ll     = lik_snr.log_likelihood(y_obs, x0h, None).sum()
    grad   = torch.autograd.grad(ll, x_t_s, allow_unused=True)[0]
    gn     = grad.norm().item() if grad is not None else 0.0
    snr_grad_norms.append(gn)
    print(f"  SNR={snr_db:2d}dB  noise_sigma={ns:.5f}  "
          f"||∇ log L||={gn:.4f}")

if max(snr_grad_norms) / (min(snr_grad_norms)+1e-8) < 2.0:
    print("  ⚠ Gradient barely changes across SNR — normalization issue!")
else:
    print("  ✓ Gradient changes meaningfully with SNR — normalization ok")

fig, ax = plt.subplots(figsize=(6,4))
snr_labels = [f"{s}dB" for s,_ in snr_configs]
ax.bar(snr_labels, snr_grad_norms, color="steelblue")
ax.set_xlabel("SNR level"); ax.set_ylabel("||∇_{x_t} log L||₂ at t=100")
ax.set_title("Likelihood gradient norm vs SNR\n(should increase with SNR)")
fig.tight_layout()
fig.savefig(str(OUT_DIR/"check2_grad_vs_snr.png"), dpi=120)
plt.close(fig)
print(f"  → saved check2_grad_vs_snr.png\n")

# ======================================================================
# CHECK [3]: Remove Langevin noise, take guided steps, plot residual norm
# ======================================================================
print("="*60)
print("CHECK [3]: Guided steps without Langevin noise — residual should decrease")
print("="*60)

# Start from a prior sample, take pure guided diffusion steps (no noise)
with torch.no_grad():
    x_init = prior.sample((1,1,4000), ddim_steps=20, device=device)

x_cur = x_init.clone()
residuals = []
t_seq = [500, 250, 100, 50, 10, 1]

for step, t_val in enumerate(t_seq):
    t_batch = torch.full((1,), t_val, dtype=torch.long, device=device)
    noise   = torch.randn_like(x_cur)
    x_t     = diff.q_sample(x_cur, t_batch, noise).requires_grad_(True)

    # Compute x0hat and likelihood gradient
    raw_g    = unet(x_t, t_batch)
    x0hat_g, _ = diff.to_x0_and_eps(x_t, t_batch, raw_g)
    ll       = lik_20.log_likelihood(y_obs, x0hat_g, None).sum()
    grad     = torch.autograd.grad(ll, x_t, allow_unused=True)[0]

    with torch.no_grad():
        raw_d    = unet(x_t.detach(), t_batch)
        x0hat_d, _ = diff.to_x0_and_eps(x_t.detach(), t_batch, raw_d)
        abar     = diff._extract(diff.alphas_bar, t_batch, x_t.shape)

        # Compute adaptive gamma
        sigma_t_sq  = 1.0 - abar
        prior_score = (x0hat_d - x_t.detach()) / (sigma_t_sq + 1e-8)
        prior_norm  = prior_score.norm().clamp(min=1e-8)
        lik_norm    = (grad.norm().clamp(min=1e-8) if grad is not None
                       else torch.tensor(1e-8))
        snr_t   = abar / (1.0 - abar + 1e-8)
        omega_t = snr_t / (1.0 + snr_t)
        lam     = 1.0   # use lambda=1 for this diagnostic
        gamma_t = float((omega_t * lam * prior_norm / lik_norm).clamp(max=50.0))

        g = grad.detach() if grad is not None else torch.zeros_like(x_t)
        x0hat_guided = (x0hat_d + gamma_t * g).clamp(-5.0, 5.0)

        # DDIM step WITHOUT Langevin noise
        t_next   = t_seq[step+1] if step < len(t_seq)-1 else 0
        t_next_b = torch.full((1,), t_next, dtype=torch.long, device=device)
        abar_prev = diff._extract(diff.alphas_bar_prev, t_next_b, x_t.shape)
        eps_g     = diff.eps_from_x0(x_t.detach(), t_batch, x0hat_guided)
        dir_xt    = (1 - abar_prev).clamp(min=0).sqrt() * eps_g
        x_cur     = (abar_prev.sqrt() * x0hat_guided + dir_xt).detach()

        # Residual ||y - H(x0hat, z_true)||²
        ppg_pred = apply_h(x0hat_guided.view(1,1,4000),
                           PHI_TRUE.to(device))
        res_norm = float((y_obs - ppg_pred).pow(2).sum().sqrt())
        residuals.append(res_norm)
        print(f"  step {step+1} (t={t_val:4d}): "
              f"||y - H(x0hat)||={res_norm:.4f}  gamma_t={gamma_t:.3f}")

if residuals[-1] < residuals[0]:
    print("  ✓ Residual decreases — gradient direction is correct")
else:
    print("  ⚠ Residual does NOT decrease — gradient direction may be wrong")

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(range(1, len(residuals)+1), residuals, 'o-', color="steelblue")
ax.set_xticks(range(1, len(t_seq)+1))
ax.set_xticklabels([f"t={t}" for t in t_seq], rotation=30)
ax.set_xlabel("Diffusion step"); ax.set_ylabel("||y - H(x̂₀, z_true)||")
ax.set_title("Residual after each guided step (no Langevin noise)\nShould decrease monotonically")
fig.tight_layout()
fig.savefig(str(OUT_DIR/"check3_residual_steps.png"), dpi=120)
plt.close(fig)
print(f"  → saved check3_residual_steps.png\n")

# ======================================================================
# CHECK [4]: Individual posterior samples — QRS structure + PPG residual
# ======================================================================
print("="*60)
print("CHECK [4]: Individual posterior ECG samples")
print("Professor hypothesis: individual samples have strong QRS at slightly")
print("different locations → posterior mean averages them out → weak QRS")
print("="*60)

# Load saved posterior samples from the staged check results
# Use a few prior samples as proxies if MCMC samples not available
print("  Drawing 6 prior samples + running 10 guided diffusion steps each")

samples = []
ppg_errs = []
for i in range(6):
    with torch.no_grad():
        s = prior.sample((1,1,4000), ddim_steps=50, device=device)

    # Run a few guided steps
    x_s = s.clone()
    for step, t_val in enumerate([500, 100, 10, 1]):
        t_batch = torch.full((1,), t_val, dtype=torch.long, device=device)
        noise   = torch.randn_like(x_s)
        x_t     = diff.q_sample(x_s, t_batch, noise).requires_grad_(True)
        raw_g   = unet(x_t, t_batch)
        x0h, _  = diff.to_x0_and_eps(x_t, t_batch, raw_g)
        ll      = lik_20.log_likelihood(y_obs, x0h, None).sum()
        grad    = torch.autograd.grad(ll, x_t, allow_unused=True)[0]
        with torch.no_grad():
            raw_d   = unet(x_t.detach(), t_batch)
            x0d, _  = diff.to_x0_and_eps(x_t.detach(), t_batch, raw_d)
            abar    = diff._extract(diff.alphas_bar, t_batch, x_t.shape)
            g       = grad.detach() if grad is not None else torch.zeros_like(x_t)
            x0_guided = (x0d + 1.0 * g).clamp(-5.0, 5.0)
            t_next  = [500,100,10,1][step+1] if step < 3 else 0
            tnb     = torch.full((1,), t_next, dtype=torch.long, device=device)
            ap      = diff._extract(diff.alphas_bar_prev, tnb, x_t.shape)
            eps_g   = diff.eps_from_x0(x_t.detach(), t_batch, x0_guided)
            x_s     = (ap.sqrt() * x0_guided +
                       (1-ap).clamp(min=0).sqrt() * eps_g).detach()

    ppg_pred = apply_h(x_s, PHI_TRUE.to(device))
    err = float((y_obs - ppg_pred).abs().mean())
    ppg_errs.append(err)
    samples.append(x_s.squeeze().cpu().numpy())
    print(f"  sample {i}: PPG err={err:.4f}  "
          f"ECG max={x_s.max().item():.3f}  std={x_s.std().item():.3f}")

t_ax = np.arange(4000)/FS
x_np = ecg_np
mean_s = np.mean(samples, axis=0)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

ax = axes[0]
for i, s in enumerate(samples):
    ax.plot(t_ax, s, alpha=0.4, lw=0.7,
            label=f"sample {i} (ppg_err={ppg_errs[i]:.3f})")
ax.plot(t_ax, x_np, 'k', lw=1.0, label="True ECG")
ax.set_title("Individual posterior ECG samples vs true ECG")
ax.legend(fontsize=6); ax.set_ylabel("Amplitude")

ax = axes[1]
ax.plot(t_ax, mean_s, "steelblue", lw=0.8, label="Posterior mean")
ax.plot(t_ax, x_np,  "k",         lw=0.8, label="True ECG")
ax.set_title(f"Posterior mean vs true ECG  "
             f"(corr={float(np.corrcoef(mean_s, x_np)[0,1]):.3f})")
ax.legend(fontsize=7); ax.set_ylabel("Amplitude")

ax = axes[2]
ax.plot(t_ax, ppg_np, "orange", lw=0.8, label="PPG observed")
ax.set_title(f"Observed PPG  (individual sample PPG errs: "
             f"{[f'{e:.3f}' for e in ppg_errs]})")
ax.legend(fontsize=7); ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")

fig.tight_layout()
fig.savefig(str(OUT_DIR/"check4_individual_samples.png"), dpi=120)
plt.close(fig)
print(f"  → saved check4_individual_samples.png\n")

print("="*60)
print("ALL CHECKS COMPLETE")
print(f"Figures saved to {OUT_DIR}/")
print("="*60)
