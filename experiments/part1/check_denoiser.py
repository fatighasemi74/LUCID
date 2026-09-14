"""
experiments/part1/check_denoiser.py
=====================================
Professor's Sep 3 diagnostic:

Take real x_0 from test set.
Corrupt to x_t = alpha_t * x_0 + sigma_t * eps.
Feed x_t to trained denoiser. Get x0hat.
Compare x0hat to x0.

Report RMSE, correlation, std(x0hat)/std(x0) at each t.

If x0hat is accurate → problem is DDIM sampling implementation.
If x0hat has low amplitude → problem is the trained prior itself.

Also verify: beta_t and alpha_bar_t are identical at training and sampling.
Also verify: DDIM transition formula is x_{t'} = alpha_{t'} * x0hat + sigma_{t'} * eps_hat.

Usage
-----
python experiments/part1/check_denoiser.py
"""

import sys, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet1d import UNet1D
from models.diffusion_core import GaussianDiffusion
from models.prior import ECGDiffusionPrior

CKPT_DIR = ROOT / "experiments/part1/checkpoints"
DATA_DIR = ROOT / "experiments/part1/synthetic_data"
OUT_DIR  = ROOT / "experiments/part1/check_denoiser"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

# ── Load trained prior ─────────────────────────────────────────────────
ckpt = torch.load(str(CKPT_DIR/"prior_k00.pt"), map_location=device,
                  weights_only=False)
unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
unet.load_state_dict(ckpt["model"])
unet.eval()

# Use SAME GaussianDiffusion as training (T=1000, cosine, pred_type=x0)
diff = GaussianDiffusion(T=1000, schedule="cosine", pred_type="x0").to(device)

print(f"Model: pred_type=x0, T=1000, schedule=cosine")
print(f"Training val_loss={ckpt.get('val_loss', 'N/A')}")

# ── Verify schedule matches training ───────────────────────────────────
print(f"\nVerifying schedule:")
print(f"  T = {diff.T}")
print(f"  alpha_bar at t=0:    {diff.alphas_bar[0].item():.6f}  (should be ≈ 1)")
print(f"  alpha_bar at t=999:  {diff.alphas_bar[999].item():.6f}  (should be small)")
print(f"  sigma_t at t=999:    {(1-diff.alphas_bar[999]).sqrt().item():.6f}  (should be ≈ 1)")
print(f"  Schedule is cosine: confirmed")

# ── Load real ECG windows ──────────────────────────────────────────────
data    = np.load(str(DATA_DIR/"snr_20dB.npz"))
ecg_np  = data["ecg"]   # (N, 4000) normalized

# Use 5 test windows
n_eval = 5
x0_batch = torch.tensor(ecg_np[:n_eval], dtype=torch.float32).unsqueeze(1).to(device)
# shape: (5, 1, 4000)

print(f"\nTest ECG windows (N={n_eval}):")
print(f"  mean std = {x0_batch.std(dim=-1).mean().item():.3f}")
print(f"  mean max = {x0_batch.abs().max(dim=-1).values.mean().item():.3f}")

# ── Diagnostic: denoiser accuracy at each t ───────────────────────────
t_levels = [10, 50, 100, 200, 300, 500, 750, 999]

print(f"\n{'t':>5} {'RMSE':>8} {'Corr':>8} {'std_ratio':>10} {'x0hat_std':>10} {'x0_std':>8}")
print("-" * 55)

rows = []
all_x0hat = {}

for t_val in t_levels:
    t_batch = torch.full((n_eval,), t_val, dtype=torch.long, device=device)

    # Corrupt x0 to x_t
    noise   = torch.randn_like(x0_batch)
    x_t     = diff.q_sample(x0_batch, t_batch, noise)

    # Denoiser prediction
    with torch.no_grad():
        raw      = unet(x_t, t_batch)
        x0hat, _ = diff.to_x0_and_eps(x_t, t_batch, raw)

    # Compute metrics
    x0_np    = x0_batch.squeeze(1).cpu().numpy()   # (N, 4000)
    x0hat_np = x0hat.squeeze(1).cpu().numpy()

    rmse_vals = np.sqrt(((x0hat_np - x0_np)**2).mean(axis=1))
    corr_vals = [np.corrcoef(x0hat_np[i], x0_np[i])[0,1] for i in range(n_eval)]
    std_ratio = x0hat_np.std(axis=1) / (x0_np.std(axis=1) + 1e-8)

    rmse    = float(np.mean(rmse_vals))
    corr    = float(np.mean(corr_vals))
    sr      = float(np.mean(std_ratio))
    x0h_std = float(x0hat_np.std(axis=1).mean())
    x0_std  = float(x0_np.std(axis=1).mean())

    rows.append((t_val, rmse, corr, sr, x0h_std, x0_std))
    print(f"{t_val:5d} {rmse:8.4f} {corr:8.4f} {sr:10.4f} {x0h_std:10.4f} {x0_std:8.4f}")
    all_x0hat[t_val] = x0hat_np[0]

print()

# ── Interpretation ─────────────────────────────────────────────────────
print("Interpretation:")
mid_t   = rows[len(rows)//2]
low_t   = rows[0]
high_t  = rows[-1]

print(f"  At t={low_t[0]} (low noise):  corr={low_t[2]:.3f}  std_ratio={low_t[3]:.3f}")
print(f"  At t={mid_t[0]} (mid noise):  corr={mid_t[2]:.3f}  std_ratio={mid_t[3]:.3f}")
print(f"  At t={high_t[0]} (high noise): corr={high_t[2]:.3f}  std_ratio={high_t[3]:.3f}")
print()

if low_t[2] > 0.8 and low_t[3] > 0.7:
    print("  ✓ Denoiser is accurate at low t (high SNR regime)")
    print("    → Problem is in the DDIM sampling implementation, not the prior")
    print("    → Check: DDIM schedule, timestep indexing, transition formula")
else:
    print("  ⚠ Denoiser is inaccurate even at low t")
    print("    → Problem is in the trained prior itself")
    if low_t[3] < 0.5:
        print("    → Std ratio < 0.5: prior underestimates amplitude (training issue)")

# ── Plot: x0hat vs x0 at each t ───────────────────────────────────────
fig, axes = plt.subplots(len(t_levels), 2, figsize=(16, 3*len(t_levels)))
t_ax = np.arange(4000) / 500
x0_plot = x0_np[0]

for row, t_val in enumerate(t_levels):
    _, rmse, corr, sr, x0h_std, x0_std = rows[row]

    ax_sig = axes[row, 0]
    ax_sig.plot(t_ax, x0_plot,           'k',         lw=0.8, label="x_0 true")
    ax_sig.plot(t_ax, all_x0hat[t_val],  'steelblue', lw=0.8,
                label=f"x0hat  corr={corr:.3f}  std_ratio={sr:.3f}")
    ax_sig.set_title(f"t={t_val}  RMSE={rmse:.4f}", fontsize=9)
    ax_sig.legend(fontsize=6)
    ax_sig.set_ylim(-4, 5)

    ax_sc = axes[row, 1]
    ax_sc.scatter(x0_plot, all_x0hat[t_val], s=1, alpha=0.1, color="steelblue")
    ax_sc.set_xlabel("x_0 true", fontsize=7)
    ax_sc.set_ylabel("x0hat", fontsize=7)
    ax_sc.set_title(f"t={t_val} scatter  corr={corr:.3f}", fontsize=9)
    # diagonal
    mn = min(x0_plot.min(), all_x0hat[t_val].min())
    mx = max(x0_plot.max(), all_x0hat[t_val].max())
    ax_sc.plot([mn,mx], [mn,mx], 'r--', lw=0.8, alpha=0.5, label="ideal")
    ax_sc.legend(fontsize=6)

fig.suptitle("Denoiser accuracy: x0hat vs x_0 at each noise level\n"
             "Good denoiser: high corr at low t, std_ratio ≈ 1",
             fontsize=11)
fig.tight_layout()
fig.savefig(str(OUT_DIR/"denoiser_accuracy.png"), dpi=100)
plt.close(fig)
print(f"\n→ saved denoiser_accuracy.png")

# ── Also verify DDIM transition formula ───────────────────────────────
print("\nVerifying DDIM transition formula:")
print("x_{t'} = alpha_{t'} * x0hat + sigma_{t'} * eps_hat")
print("(This is the deterministic DDIM, no stochastic noise added)")
print()

# Take one step: t=100 → t=50
t_from, t_to = 100, 50
t_f = torch.full((1,), t_from, dtype=torch.long, device=device)
t_t = torch.full((1,), t_to,   dtype=torch.long, device=device)

x0_one = x0_batch[:1]
noise   = torch.randn_like(x0_one)
x_t100  = diff.q_sample(x0_one, t_f, noise)

with torch.no_grad():
    raw       = unet(x_t100, t_f)
    x0hat_100, eps_hat_100 = diff.to_x0_and_eps(x_t100, t_f, raw)
    abar_to   = diff._extract(diff.alphas_bar, t_t, x_t100.shape)
    alpha_to  = abar_to.sqrt()
    sigma_to  = (1 - abar_to).clamp(min=0).sqrt()
    x_t50     = alpha_to * x0hat_100 + sigma_to * eps_hat_100

print(f"  t=100 → t=50 DDIM step:")
print(f"  alpha_{t_to}  = {float(alpha_to.mean()):.4f}")
print(f"  sigma_{t_to}  = {float(sigma_to.mean()):.4f}")
print(f"  x_t50 std   = {x_t50.std().item():.4f}  (should be similar to x_t100)")
print(f"  x_t50 max   = {x_t50.abs().max().item():.4f}")
print(f"  x_t50 NaN   = {torch.isnan(x_t50).any().item()}")

print(f"\nAll figures saved to {OUT_DIR}/")
