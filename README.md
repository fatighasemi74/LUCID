# LUCID — Bayesian Inverse Diffusion for PPG-to-ECG Reconstruction

<p align="center">
  <img src="assets/lucid_overview.png" width="700" alt="LUCID overview"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg"/></a>
  <a href="https://github.com/kaneko29/Pulse2Posterior/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-brightgreen"/>
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange"/>
</p>

> **LUCID: Calibrated Bayesian ECG Reconstruction from PPG with Source-Attributed Uncertainty**  
> Fatemeh Ghassemi, Alex Kaneko, Mahesh Banavar, Bahman Moraffah  
> *ICASSP 2027*

---

## Overview

Reconstructing electrocardiograms (ECGs) from photoplethysmography (PPG) is a fundamentally ill-posed inverse problem. LUCID is a Bayesian framework that:

- **Reconstructs** ECG waveforms from PPG signals using a diffusion prior
- **Quantifies** uncertainty from four distinct sources: ECG prior ($U_X$), physiological parameters ($U_Z$), forward model ($U_\Phi$), and prior ensemble ($U_\Theta$)
- **Calibrates** posterior credible intervals to nominal coverage levels
- **Identifies** unreliable reconstructions via posterior uncertainty

### Key results

| Metric | Value |
|--------|-------|
| PPG consistency at SNR=40 dB | 0.049 |
| PPG consistency at SNR=5 dB | 0.296 |
| 90% calibrated coverage | 87.5%–93.3% |
| Staged ablation correlation | 0.938 |
| VitalDB Spearman (unc vs PPG err) | 0.776 (p=0.003) |

---

## Method

<p align="center">
  <img src="assets/method_diagram.png" width="650" alt="Method overview"/>
</p>

LUCID combines:

1. **Bootstrap ensemble of ECG diffusion priors** — $K=10$ priors trained on PTB-XL ECG, providing prior model uncertainty $U_\Theta$
2. **Physiological forward operator** — $H_\Phi(X, Z) = a \cdot (h_\eta \ast X)(\cdot - \tau) + b$ with record-specific $Z = (\tau, a, b)$
3. **Nested MCMC sampler** — three update steps per iteration:
   - $\tau$: cross-correlation MH proposal
   - $(a, b)$: MALA
   - $X$: annealed guided DDIM
4. **Variance decomposition** — exact attribution of posterior variance to each source

---

## Installation

```bash
git clone https://github.com/kaneko29/Pulse2Posterior.git
cd Pulse2Posterior
conda env create -f environment.yml
conda activate pulse2posterior
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 11.8+ (recommended)

---

## Quick start

### 1. Download PTB-XL and generate synthetic data

```bash
# Download PTB-XL (requires kaggle API)
python data/ptbxl.py --download --out-dir data/ptbxl

# Generate semi-synthetic PPG at four SNR levels
python experiments/part1/generate.py --snr-db 5 10 20 40 --out-dir experiments/part1/synthetic_data
```

### 2. Train the bootstrap ensemble of ECG priors

```bash
# Train K=10 priors (one per bootstrap resample of PTB-XL training set)
for k in $(seq 0 9); do
    python experiments/part1/train_prior.py --k $k --epochs 100
done
```

Checkpoints saved to `experiments/part1/checkpoints/prior_k{00..09}.pt`

### 3. Run controlled SNR sweep

```bash
python experiments/part1/run.py \
    --snr-db 5 10 20 40 \
    --n-windows 20 \
    --n-inner 100 \
    --burn-in 20
```

Results saved to `experiments/part1/results/results_snr{05,10,20,40}dB.npz`

### 4. Generate paper figures

```bash
python experiments/part1/make_figures.py
python experiments/part1/calibrate.py --snr-db 5 10 20 40
```

### 5. VitalDB real-data evaluation

```bash
# Estimate forward model parameters from VitalDB paired data
python experiments/part1/train_vi_vitaldb.py --max-cases 200 --n-estimate 500

# Run inference on held-out patients
python experiments/part1/run_vitaldb.py --n-windows 20 --n-inner 100 --burn-in 20
```

---

## Repository structure

```
Pulse2Posterior/
├── data/
│   ├── ptbxl.py              # PTB-XL data loader and preprocessing
│   └── vitaldb.py            # VitalDB streaming loader (no download needed)
├── models/
│   ├── unet1d.py             # 1-D U-Net denoiser backbone
│   ├── diffusion_core.py     # Gaussian diffusion (DDIM, cosine schedule)
│   ├── prior.py              # ECG diffusion prior wrapper
│   ├── likelihood.py         # Gaussian likelihood L_Φ(y|x,z)
│   └── noise_cov.py          # Isotropic/diagonal noise covariance
├── inference/
│   ├── mala.py               # MCMC kernels: τ xcorr-MH, (a,b) MALA, X DDIM
│   ├── sampler.py            # Bootstrap ensemble sampler + SMC evidence
│   └── vi.py                 # Mean-field Gaussian q_λ(Φ) and ELBO
└── experiments/part1/
    ├── generate.py           # Semi-synthetic PPG generation
    ├── train_prior.py        # Single prior training (one bootstrap resample)
    ├── train_bootstrap.py    # Full K=10 ensemble training
    ├── run.py                # Main SNR sweep experiment
    ├── calibrate.py          # Posterior-scale inflation on D_cal
    ├── make_figures.py       # Paper figures from saved results
    ├── staged_check.py       # Staged ablation (Stages 1–4)
    ├── phi_ablation.py       # φ ablation: which component drives PPG error
    ├── train_vi_vitaldb.py   # Empirical Φ estimation from VitalDB
    └── run_vitaldb.py        # Held-out VitalDB patient evaluation
```

---

## Reproducing paper results

All results in the paper can be reproduced with the following commands.

### Table I — Staged ablation (SNR=20 dB, N=30, K=1)

```bash
python experiments/part1/staged_check.py \
    --n-inner 30 --burn-in 5 --gamma 1.0 --snr-db 20
```

### Table II — SNR sweep (K=10, N=100, 20 windows)

```bash
python experiments/part1/run.py \
    --snr-db 5 10 20 40 --n-windows 20 --n-inner 100 --burn-in 20
python experiments/part1/calibrate.py --snr-db 5 10 20 40
```

### Figure 2 — PPG consistency

```bash
python experiments/part1/make_figures.py
```

### Figure 3 — Calibration curves

```bash
python experiments/part1/calibrate.py --snr-db 5 10 20 40
```

### Table IV — VitalDB evaluation

```bash
python experiments/part1/train_vi_vitaldb.py --max-cases 200
python experiments/part1/run_vitaldb.py --n-windows 20 --n-inner 100 --burn-in 20
```

---

## Pre-trained checkpoints

| Checkpoint | Description | Download |
|-----------|-------------|----------|
| `prior_k{00..09}.pt` | K=10 bootstrap ECG priors (PTB-XL) | [Google Drive] |
| `lambda_star.pt` | q_λ*(Φ) for semi-synthetic experiments | [Google Drive] |
| `lambda_star_vitaldb.pt` | q_λ*(Φ) estimated from VitalDB | [Google Drive] |

---

## Citation

```bibtex
@inproceedings{ghassemi2027lucid,
  title     = {{LUCID}: Calibrated {B}ayesian {ECG} Reconstruction
               from {PPG} with Source-Attributed Uncertainty},
  author    = {Ghassemi, Fatemeh and Kaneko, Alex and
               Banavar, Mahesh and Moraffah, Bahman},
  booktitle = {IEEE International Conference on Acoustics,
               Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

ECG prior trained on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (Wagner et al., 2020).  
Real-data evaluation on [VitalDB](https://vitaldb.net) (Lee et al., 2022).  
Baselines: [CardioGAN](https://github.com/pritamqu/ppg2ecg) (Sarkar & Etemad, 2021), [RDDM](https://github.com/shome-g/rddm) (Shome et al., 2024).
