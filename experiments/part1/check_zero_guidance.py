"""
experiments/part1/check_zero_guidance.py
==========================================
Professor's request: validate zero-guidance sampler with 50-100 DDIM steps.
The 6-step annealing check was not sufficient — samples did not have clear ECG morphology.
This script runs proper unconditional DDIM sampling and confirms samples look like ECG.

Usage
-----
python experiments/part1/check_zero_guidance.py
"""

import sys
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
OUT_DIR  = ROOT / "experiments/part1/check_zero_guidance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

# Load prior k=0
ckpt = torch.load(str(CKPT_DIR/"prior_k00.pt"), map_location=device,
                  weights_only=False)
unet = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
unet.load_state_dict(ckpt["model"]); unet.eval()
diff  = GaussianDiffusion(T=1000, schedule="cosine", pred_type="x0").to(device)
prior = ECGDiffusionPrior(denoiser=unet, T=1000, pred_type="x0",
                          device=device).to(device)

# Load a few real ECGs for comparison
data   = np.load(str(DATA_DIR/"snr_20dB.npz"))
real_ecg = data["ecg"][:4]   # 4 real ECG windows

print("Generating unconditional samples with various DDIM step counts...")
print("(No guidance — pure prior sampling)\n")

step_counts = [10, 20, 50, 100]
results = {}

for n_steps in step_counts:
    samples = []
    with torch.no_grad():
        for _ in range(5):
            s = prior.sample((1,1,4000), ddim_steps=n_steps, device=device)
            samples.append(s.squeeze().cpu().numpy())
    samples = np.stack(samples)
    results[n_steps] = samples
    print(f"ddim_steps={n_steps:4d}: "
          f"mean_std={samples.std(axis=1).mean():.3f}  "
          f"mean_max={samples.max(axis=1).mean():.3f}  "
          f"max_of_max={samples.max(axis=1).max():.3f}  "
          f"clipped={int((samples.max(axis=1) > 4.9).sum())}/5")

print(f"\nReal ECG windows:")
print(f"  mean_std={real_ecg.std(axis=1).mean():.3f}  "
      f"mean_max={real_ecg.max(axis=1).mean():.3f}")

# ── Plot comparison ────────────────────────────────────────────────────
t_ax = np.arange(4000) / 500   # time axis in seconds

fig, axes = plt.subplots(len(step_counts) + 1, 3, figsize=(18, 4*(len(step_counts)+1)))

# Row 0: real ECG
for col in range(3):
    ax = axes[0, col]
    ax.plot(t_ax, real_ecg[col], 'k', lw=0.7)
    ax.set_title(f"Real ECG (window {col})  std={real_ecg[col].std():.3f}", fontsize=9)
    ax.set_ylim(-3, 5)
axes[0, 0].set_ylabel("Real ECG", fontsize=9)

# Rows 1+: prior samples at each step count
for row, n_steps in enumerate(step_counts):
    samples = results[n_steps]
    for col in range(3):
        ax = axes[row+1, col]
        ax.plot(t_ax, samples[col], 'steelblue', lw=0.7)
        ax.set_title(f"Prior sample (steps={n_steps})  "
                     f"std={samples[col].std():.3f}", fontsize=9)
        ax.set_ylim(-5, 5)
    axes[row+1, 0].set_ylabel(f"steps={n_steps}", fontsize=9)

for ax in axes.flat:
    ax.set_xlabel("Time (s)", fontsize=7)

fig.suptitle("Zero-guidance unconditional ECG samples vs real ECG\n"
             "Samples should show clear P-QRS-T morphology at correct scale",
             fontsize=11)
fig.tight_layout()
fig.savefig(str(OUT_DIR/"zero_guidance_samples.png"), dpi=120)
plt.close(fig)
print(f"\n→ saved zero_guidance_samples.png")

# ── Statistical comparison ─────────────────────────────────────────────
print("\nStatistical comparison:")
print(f"{'Source':<20} {'mean std':>10} {'mean max':>10} {'min std':>10} {'max std':>10}")
print("-"*55)
print(f"{'Real ECG':<20} "
      f"{real_ecg.std(axis=1).mean():>10.3f} "
      f"{real_ecg.max(axis=1).mean():>10.3f} "
      f"{real_ecg.std(axis=1).min():>10.3f} "
      f"{real_ecg.std(axis=1).max():>10.3f}")
for n_steps in step_counts:
    s = results[n_steps]
    print(f"{'Prior ('+str(n_steps)+' steps)':<20} "
          f"{s.std(axis=1).mean():>10.3f} "
          f"{s.max(axis=1).mean():>10.3f} "
          f"{s.std(axis=1).min():>10.3f} "
          f"{s.std(axis=1).max():>10.3f}")

print(f"\nConclusion:")
best_steps = max(step_counts,
                 key=lambda n: -abs(results[n].std(axis=1).mean() -
                                    real_ecg.std(axis=1).mean()))
print(f"  Best amplitude match: {best_steps} DDIM steps")
print(f"  Real ECG std: {real_ecg.std(axis=1).mean():.3f}")
print(f"  Prior at {best_steps} steps: std={results[best_steps].std(axis=1).mean():.3f}")
print(f"\nFigures saved to {OUT_DIR}/")
