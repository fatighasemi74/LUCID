"""
experiments/part1/check_x_update.py
=====================================
Professor's Sep 2 diagnostic requests for Stage 1 X update stability.

[1] Zero guidance — verify diffusion alone is stable
[2] Per-timestep: gamma_t, ||s_theta||, ||g_t||, ||gamma_t*g_t||, ratio
[3] Displacement magnitudes: eta*||s_theta|| and eta*||gamma_t*g_t||
[4] Direction check: R(eta) = ||y - H(x0hat(x_t + eta*g_t), z_true)||^2
    for small eta — should decrease. Plus finite-difference gradient check.

All with Z=z_true, phi=phi_true fixed (Stage 1 oracle).

Usage
-----
python experiments/part1/check_x_update.py
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
OUT_DIR  = ROOT / "experiments/part1/check_x_update"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
PHI_TRUE = torch.tensor([0.2, 0.8, 0.05, -3.0])

def phi_to_params(phi):
    return {"log_a": phi[1].abs().log(), "b": phi[2],
            "tau": phi[0], "log_diag": phi[3].expand(1)}

def apply_h(x, phi=None, fs=FS):
    if phi is None: phi = PHI_TRUE
    phi = phi.to(x.device)
    a = phi[1].abs(); b = phi[2]
    half = 30
    t = torch.arange(-half, half+1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t**2/(2*10.0**2)); k = k/k.sum()
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
diff  = GaussianDiffusion(T=1000, schedule="cosine", pred_type="x0").to(device)
prior = ECGDiffusionPrior(denoiser=unet, T=1000, pred_type="x0",
                          device=device).to(device)

# ── Load test window ───────────────────────────────────────────────────
data   = np.load(str(DATA_DIR/"snr_20dB.npz"))
ecg_np = data["ecg"][0]
ppg_np = data["ppg"][0]
x_true = torch.tensor(ecg_np).view(1,1,4000).to(device)
y_obs  = torch.tensor(ppg_np).view(1,1,4000).to(device)
ns     = float(data["noise_sigma"])

lik = GaussianLikelihood(obs_dim=4000, fs=500.0, cov_mode=CovMode.ISOTROPIC,
                         kernel_sigma=10.0, kernel_size=61).to(device)
params = phi_to_params(PHI_TRUE.to(device))
params["log_diag"] = torch.tensor([math.log(ns**2)], device=device)
_set_likelihood_params(lik, params)

print(f"SNR=20dB  noise_sigma={ns:.5f}")
print(f"ECG std={x_true.std():.3f}  PPG std={y_obs.std():.3f}\n")

t_anneal = [500, 250, 100, 50, 10, 1]

# DDIM step size η — from diffusion schedule
# In DDIM: x_{t-1} = sqrt(abar_{t-1})*x0hat + sqrt(1-abar_{t-1})*eps
# The "step size" for x0hat update is effectively sqrt(abar_{t-1})
def get_eta(diff, t_val, t_next_val, x_shape, device):
    t_b  = torch.full((1,), t_val,      dtype=torch.long, device=device)
    tn_b = torch.full((1,), t_next_val, dtype=torch.long, device=device)
    abar      = diff._extract(diff.alphas_bar,      t_b,  x_shape).item()
    abar_prev = diff._extract(diff.alphas_bar_prev, tn_b, x_shape).item()
    return math.sqrt(abar_prev)   # coefficient on x0hat in DDIM step

# ======================================================================
# CHECK [1]: Zero guidance — diffusion alone stability
# ======================================================================
print("="*60)
print("CHECK [1]: Unconditional diffusion (zero guidance)")
print("="*60)

with torch.no_grad():
    x_ug = prior.sample((1,1,4000), ddim_steps=50, device=device)

print(f"  Unconditional sample: std={x_ug.std():.3f}  "
      f"max={x_ug.abs().max():.3f}  min={x_ug.min():.3f}")
print(f"  True ECG:             std={x_true.std():.3f}  "
      f"max={x_true.abs().max():.3f}")

# Run annealing with ZERO guidance to confirm stability
x_zero = prior.sample((1,1,4000), ddim_steps=10, device=device).detach()
for i, t_val in enumerate(t_anneal):
    t_b  = torch.full((1,), t_val, dtype=torch.long, device=device)
    t_nv = t_anneal[i+1] if i < len(t_anneal)-1 else 0
    tn_b = torch.full((1,), t_nv,  dtype=torch.long, device=device)
    noise = torch.randn_like(x_zero)
    x_t   = diff.q_sample(x_zero, t_b, noise)
    with torch.no_grad():
        raw      = unet(x_t, t_b)
        x0hat, _ = diff.to_x0_and_eps(x_t, t_b, raw)
        abar_p   = diff._extract(diff.alphas_bar_prev, tn_b, x_t.shape)
        eps0     = diff.eps_from_x0(x_t, t_b, x0hat)
        x_zero   = (abar_p.sqrt()*x0hat +
                    (1-abar_p).clamp(min=0).sqrt()*eps0).detach()
    print(f"  t={t_val:4d}: std={x_zero.std():.3f}  "
          f"max={x_zero.abs().max():.3f}  "
          f"clipped={'YES' if x_zero.abs().max()>4.9 else 'no'}")

fig, axes = plt.subplots(2,1, figsize=(14,6), sharex=True)
t_ax = np.arange(4000)/FS
axes[0].plot(t_ax, ecg_np, 'k', lw=0.8, label="True ECG")
axes[0].set_title(f"True ECG  std={x_true.std():.3f}")
axes[0].legend(fontsize=7)
axes[1].plot(t_ax, x_zero.squeeze().cpu().numpy(), 'steelblue', lw=0.8,
             label=f"Zero-guidance sample  std={x_zero.std():.3f}")
axes[1].set_title("Diffusion sample (zero guidance) — should look like ECG")
axes[1].legend(fontsize=7); axes[1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig(str(OUT_DIR/"check1_zero_guidance.png"), dpi=120)
plt.close(fig)
print(f"  → saved check1_zero_guidance.png\n")

# ======================================================================
# CHECK [2] and [3]: Per-timestep magnitudes and displacements
# ======================================================================
print("="*60)
print("CHECK [2+3]: Per-timestep magnitudes and displacements")
print("="*60)
print(f"{'t':>6} {'gamma_t':>9} {'||s_th||':>10} {'||g_t||':>10} "
      f"{'||gam*g||':>11} {'ratio':>8} {'eta*||s||':>11} {'eta*||gam*g||':>14}")
print("-"*85)

with torch.no_grad():
    x_start = prior.sample((1,1,4000), ddim_steps=20, device=device)

rows = []
for i, t_val in enumerate(t_anneal):
    t_b  = torch.full((1,), t_val, dtype=torch.long, device=device)
    t_nv = t_anneal[i+1] if i < len(t_anneal)-1 else 0

    noise = torch.randn_like(x_start)
    x_t   = diff.q_sample(x_start, t_b, noise).requires_grad_(True)

    # Likelihood gradient g_t = ∇_{x_t} log L
    raw_g    = unet(x_t, t_b)
    x0hat_g, _ = diff.to_x0_and_eps(x_t, t_b, raw_g)
    ll       = lik.log_likelihood(y_obs, x0hat_g, None).sum()
    g_t      = torch.autograd.grad(ll, x_t, allow_unused=True)[0]
    g_norm   = float(g_t.norm()) if g_t is not None else 0.0
    g_t_d    = g_t.detach() if g_t is not None else torch.zeros_like(x_t)

    # Prior score ||s_theta||
    with torch.no_grad():
        raw_d      = unet(x_t.detach(), t_b)
        x0hat_d, _ = diff.to_x0_and_eps(x_t.detach(), t_b, raw_d)
        abar       = diff._extract(diff.alphas_bar, t_b, x_t.shape)
        sigma_t_sq = 1.0 - abar
        prior_score = (x0hat_d - x_t.detach()) / (sigma_t_sq + 1e-8)
        s_norm      = float(prior_score.norm())

        # Adaptive gamma_t
        snr_t   = abar / (sigma_t_sq + 1e-8)
        omega_t = snr_t / (1.0 + snr_t)
        lam     = 1.0   # lambda=1 for this diagnostic
        gamma_t = float((omega_t * lam * s_norm /
                         (g_norm + 1e-8)).clamp(max=50.0))

        gamma_g_norm = gamma_t * g_norm
        ratio        = gamma_g_norm / (s_norm + 1e-8)

        # Step size eta — coefficient on x0hat in DDIM
        eta = get_eta(diff, t_val, t_nv, x_t.shape, device)

        eta_s   = eta * s_norm
        eta_gg  = eta * gamma_g_norm

    rows.append((t_val, gamma_t, s_norm, g_norm, gamma_g_norm,
                 ratio, eta_s, eta_gg))
    print(f"{t_val:>6} {gamma_t:>9.4f} {s_norm:>10.2f} {g_norm:>10.2f} "
          f"{gamma_g_norm:>11.2f} {ratio:>8.3f} {eta_s:>11.2f} {eta_gg:>14.2f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(9,7))
ts = [r[0] for r in rows]
ax = axes[0]
ax.semilogy(ts, [r[2] for r in rows], 'o-', label="||s_theta|| (prior score)")
ax.semilogy(ts, [r[4] for r in rows], 's-', label="||gamma_t * g_t|| (guidance)")
ax.invert_xaxis()
ax.set_xlabel("t"); ax.set_ylabel("L2 norm (log scale)")
ax.set_title("Prior score vs guidance magnitude")
ax.legend()

ax = axes[1]
ax.semilogy(ts, [r[6] for r in rows], 'o-', label="eta * ||s_theta||")
ax.semilogy(ts, [r[7] for r in rows], 's-', label="eta * ||gamma_t * g_t||")
ax.invert_xaxis()
ax.set_xlabel("t"); ax.set_ylabel("Displacement magnitude")
ax.set_title("Displacement magnitudes in DDIM step")
ax.legend()
fig.tight_layout()
fig.savefig(str(OUT_DIR/"check23_magnitudes.png"), dpi=120)
plt.close(fig)
print(f"  → saved check23_magnitudes.png\n")

# ======================================================================
# CHECK [4]: Direction check — R(eta) should decrease for small eta
#            Plus finite-difference gradient verification
# ======================================================================
print("="*60)
print("CHECK [4]: Guidance direction check at fixed t=100, fixed x_t")
print("="*60)

t_fixed = 100
t_b_f   = torch.full((1,), t_fixed, dtype=torch.long, device=device)
noise   = torch.randn_like(x_start)
x_t_f   = diff.q_sample(x_start, t_b_f, noise).requires_grad_(True)

# Compute g_t at this fixed point
raw_g    = unet(x_t_f, t_b_f)
x0hat_f, _ = diff.to_x0_and_eps(x_t_f, t_b_f, raw_g)
ll_f     = lik.log_likelihood(y_obs, x0hat_f, None).sum()
g_t_f    = torch.autograd.grad(ll_f, x_t_f, allow_unused=True)[0].detach()

print(f"  Fixed t={t_fixed}  ||g_t||={g_t_f.norm():.4f}")

def R_eta(x_t_base, g, eta_val):
    """R(eta) = ||y - H(x0hat(x_t + eta*g), z_true)||^2"""
    x_shifted = (x_t_base + eta_val * g).requires_grad_(False)
    with torch.no_grad():
        raw_s    = unet(x_shifted, t_b_f)
        x0h_s, _ = diff.to_x0_and_eps(x_shifted, t_b_f, raw_s)
        ppg_s    = apply_h(x0h_s)
        return float((y_obs - ppg_s).pow(2).sum())

etas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
print(f"\n  R(eta) = ||y - H(x0hat(x_t + eta*g_t), z_true)||^2")
print(f"  {'eta':>10} {'R(eta)':>15} {'decreasing?':>12}")
R_vals = []
x_t_base = x_t_f.detach()
for eta in etas:
    R = R_eta(x_t_base, g_t_f, eta)
    R_vals.append(R)
    dec = "↓" if eta > 0 and R < R_vals[0] else ("↑" if eta > 0 else "—")
    print(f"  {eta:>10.1e} {R:>15.4f} {dec:>12}")

if R_vals[-2] < R_vals[0]:  # small eta should decrease
    print("  ✓ R(eta) decreases for small eta — gradient direction CORRECT")
else:
    print("  ⚠ R(eta) does NOT decrease — gradient direction may be wrong")

# Finite-difference gradient check
print(f"\n  Finite-difference gradient check (directional derivative):")
delta   = 1e-4
x_plus  = x_t_base + delta * g_t_f
x_minus = x_t_base - delta * g_t_f

with torch.no_grad():
    # log L at x_plus and x_minus
    r_plus   = unet(x_plus,  t_b_f)
    x0p, _   = diff.to_x0_and_eps(x_plus,  t_b_f, r_plus)
    r_minus  = unet(x_minus, t_b_f)
    x0m, _   = diff.to_x0_and_eps(x_minus, t_b_f, r_minus)
    ll_plus  = float(lik.log_likelihood(y_obs, x0p, None).sum())
    ll_minus = float(lik.log_likelihood(y_obs, x0m, None).sum())

fd_deriv   = (ll_plus - ll_minus) / (2 * delta)
auto_deriv = float((g_t_f * g_t_f).sum())  # g_t · g_t = ||g_t||^2

# Normalized agreement
fd_norm   = fd_deriv / (auto_deriv + 1e-8)
print(f"  Finite-diff directional deriv: {fd_deriv:.4f}")
print(f"  Autograd ||g_t||^2:            {auto_deriv:.4f}")
print(f"  Normalized ratio (should ≈ 1): {fd_norm:.4f}")
if 0.5 < fd_norm < 2.0:
    print("  ✓ Finite-difference agrees with autograd — gradient is correct")
else:
    print("  ⚠ Large discrepancy — gradient may be incorrect")

# Plot R(eta)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(etas[1:], R_vals[1:], 'o-', color="steelblue")
ax.axhline(R_vals[0], color='k', ls='--', label=f"R(0) = {R_vals[0]:.2f}")
ax.set_xlabel("η"); ax.set_ylabel("R(η) = ||y - H(x̂₀(x_t + η·g_t))||²")
ax.set_title(f"Guidance direction check at t={t_fixed}\n"
             f"R(η) should decrease for small η")
ax.legend(); ax.set_xscale("log")
fig.tight_layout()
fig.savefig(str(OUT_DIR/"check4_direction.png"), dpi=120)
plt.close(fig)
print(f"  → saved check4_direction.png\n")

# ======================================================================
# SUMMARY TABLE
# ======================================================================
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Check 1 — Zero guidance stable: "
      f"{'YES' if x_zero.abs().max() < 4.9 else 'NO — clipping!'}")
print(f"Check 2+3 — ratio ||gamma*g||/||s|| at t=500: {rows[0][5]:.3f}  "
      f"at t=1: {rows[-1][5]:.3f}")
print(f"Check 4 — R(eta) decreases: "
      f"{'YES' if R_vals[2] < R_vals[0] else 'NO'}")
print(f"Check 4 — FD gradient agrees: "
      f"{'YES' if 0.5 < fd_norm < 2.0 else 'NO'}")
print(f"\nFigures in {OUT_DIR}/")
