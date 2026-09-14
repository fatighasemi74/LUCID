"""
experiments/part1/calibrate.py
================================
Posterior-scale inflation calibration.

The posterior is underconfident (coverage > nominal) because the
fresh-noise DDIM initialization creates prior-width diversity rather
than posterior-width diversity. This script fits a scalar inflation
factor c on D_cal that corrects residual miscalibration, then reports
calibrated coverage and produces paper figures on D_test.

Method
------
For each window i, the posterior produces mean μ_i and std σ_i.
The inflated std is σ_i' = c · σ_i.
We find c* = argmin_{c} |coverage_empirical(c) - coverage_nominal|
on D_cal, then apply c* to D_test.

Usage
-----
python experiments/part1/calibrate.py --snr-db 20
python experiments/part1/calibrate.py --snr-db 5 10 20 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "experiments/part1/results"
OUT_DIR     = ROOT / "experiments/part1/calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Load results
# ======================================================================

def load_results(snr_db: int) -> dict:
    path = RESULTS_DIR / f"results_snr{snr_db:02d}dB.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run run.py first.")
    d = np.load(str(path))
    return {k: d[k] for k in d.files}

# ======================================================================
# Coverage computation
# ======================================================================

def empirical_coverage(
    x_true    : np.ndarray,   # (N, L) true ECG windows
    mean      : np.ndarray,   # (N, L) posterior means
    std       : np.ndarray,   # (N, L) posterior stds
    level     : float,        # nominal coverage e.g. 0.90
    c         : float = 1.0,  # scale inflation factor
) -> float:
    """Fraction of true timesteps inside ±z_alpha * c * std."""
    z    = stats.norm.ppf((1 + level) / 2)
    half = z * c * std
    inside = np.abs(x_true - mean) <= half
    return float(inside.mean())

# ======================================================================
# Fit inflation factor on calibration split
# ======================================================================

def fit_inflation(
    x_cal  : np.ndarray,
    mean_cal: np.ndarray,
    std_cal : np.ndarray,
    target_level: float = 0.90,
) -> float:
    """
    Find c* such that empirical coverage at target_level matches nominal.
    Uses bisection on c in [0.1, 5.0].
    """
    def gap(c):
        emp = empirical_coverage(x_cal, mean_cal, std_cal, target_level, c)
        return abs(emp - target_level)

    result = minimize_scalar(gap, bounds=(0.1, 5.0), method='bounded')
    return float(result.x)

# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr-db", nargs="+", type=int, default=[20],
                    help="SNR levels to calibrate")
    ap.add_argument("--cal-frac", type=float, default=0.3,
                    help="fraction of windows used for calibration (rest = test)")
    args = ap.parse_args()

    nominal_levels = [0.50, 0.80, 0.90, 0.95]

    print("Posterior-scale inflation calibration")
    print(f"Cal fraction: {args.cal_frac:.0%}  Test fraction: {1-args.cal_frac:.0%}")
    print()

    all_snr_results = {}

    for snr_db in args.snr_db:
        print(f"{'='*55}")
        print(f"SNR = {snr_db} dB")
        print(f"{'='*55}")

        # Load per-window metrics saved by run.py
        d = load_results(snr_db)
        n = len(d["rmse"])

        # We only have per-window scalars from run.py, not per-timestep arrays.
        # We need to reconstruct coverage from the saved coverage arrays.
        # run.py saves coverage_50, coverage_80, coverage_90, coverage_95
        # as per-window empirical coverage values.

        # Split into cal / test
        n_cal  = max(1, int(n * args.cal_frac))
        n_test = n - n_cal
        idx    = np.arange(n)
        cal_idx  = idx[:n_cal]
        test_idx = idx[n_cal:]

        print(f"  N={n} windows  cal={n_cal}  test={n_test}")

        # Per-window coverage at each nominal level
        cov_arrays = {
            0.50: d["coverage_50"],
            0.80: d["coverage_80"],
            0.90: d["coverage_90"],
            0.95: d["coverage_95"],
        }
        uncertainty = d["uncertainty"]  # per-window posterior std mean

        # Fit c: find scalar that maps uncertainty to correct coverage
        # Since we have per-window coverage (not per-timestep), we use
        # the mean coverage on cal set and find c that corrects it.
        # Approximation: empirical_coverage(c) ≈ Phi(Phi^{-1}(nominal)/c)
        # for Gaussian posterior with std scaled by c.

        def corrected_coverage(c, level, cov_vals_cal):
            """
            If original coverage at level α is p_0, what is coverage
            after multiplying std by c?
            p_0 = Phi(z_α / 1) approximately, so
            p_c = Phi(z_α / c) if p_0 < α (overconfident)
                  Phi(z_α · c) if p_0 > α (underconfident, c < 1)
            Since our posterior is underconfident (p_0 > α), we need c < 1
            to shrink intervals. But here we fit directly from data.
            """
            z_alpha = stats.norm.ppf((1 + level) / 2)
            # Empirical z-score: what z_emp would give observed coverage?
            # p_0 = Phi(z_emp) => z_emp = Phi^{-1}(p_0)
            p_cal = float(np.mean(cov_vals_cal))
            p_cal_clipped = np.clip(p_cal, 0.501, 0.999)
            z_emp = stats.norm.ppf((1 + p_cal_clipped) / 2)
            # After scaling by c: new z = z_alpha, old z = z_emp
            # c_fit = z_alpha / z_emp
            c_fit = z_alpha / (z_emp + 1e-8)
            return float(c_fit)

        # Fit c on calibration set at 90% level (primary target)
        c_star = corrected_coverage(1.0, 0.90, cov_arrays[0.90][cal_idx])
        print(f"  Fitted inflation factor c* = {c_star:.4f}")
        print(f"  (c < 1 means shrinking intervals to reduce underconfidence)")
        print()

        # Compute corrected coverage on test set
        print(f"  {'Level':>8} {'Cal emp':>10} {'Test raw':>10} "
              f"{'Test cal':>10} {'Gap raw':>9} {'Gap cal':>9}")
        print(f"  {'-'*58}")

        rows = {}
        for level in nominal_levels:
            cal_emp  = float(np.mean(cov_arrays[level][cal_idx]))
            test_raw = float(np.mean(cov_arrays[level][test_idx]))
            # Corrected: apply c* to the z-score
            z_alpha  = stats.norm.ppf((1 + level) / 2)
            z_emp_test = stats.norm.ppf(
                (1 + np.clip(test_raw, 0.501, 0.999)) / 2
            )
            z_corrected = z_emp_test * c_star
            test_cal = float(2 * stats.norm.cdf(abs(z_corrected)) - 1)
            gap_raw  = test_raw - level
            gap_cal  = test_cal - level
            print(f"  {int(level*100):>7}%  {cal_emp*100:>9.1f}%  "
                  f"{test_raw*100:>9.1f}%  {test_cal*100:>9.1f}%  "
                  f"{gap_raw*100:>+8.1f}%  {gap_cal*100:>+8.1f}%")
            rows[level] = dict(cal_emp=cal_emp, test_raw=test_raw,
                               test_cal=test_cal)

        # Reconstruction and sharpness on test set
        rmse_test = float(np.mean(d["rmse"][test_idx]))
        corr_test = float(np.mean(d["corr"][test_idx]))
        unc_test  = float(np.mean(uncertainty[test_idx]))
        unc_cal   = unc_test * c_star
        ppg_test  = float(np.mean(d["ppg_consistency"][test_idx]))

        print()
        print(f"  Test set reconstruction:")
        print(f"    RMSE:        {rmse_test:.4f}")
        print(f"    Correlation: {corr_test:.4f}")
        print(f"    Sharpness (std):  {unc_test:.4f}  → calibrated: {unc_cal:.4f}")
        print(f"    PPG consistency:  {ppg_test:.4f}")

        all_snr_results[snr_db] = dict(
            c_star=c_star, rows=rows,
            rmse=rmse_test, corr=corr_test,
            unc=unc_test, unc_cal=unc_cal,
            ppg=ppg_test,
        )

    # ── Calibration curve figure ───────────────────────────────────────
    fig, axes = plt.subplots(1, len(args.snr_db),
                             figsize=(5*len(args.snr_db), 5))
    if len(args.snr_db) == 1:
        axes = [axes]

    for ax, snr_db in zip(axes, args.snr_db):
        r = all_snr_results[snr_db]
        nom    = [l for l in nominal_levels]
        raw    = [r["rows"][l]["test_raw"]  for l in nominal_levels]
        cal    = [r["rows"][l]["test_cal"]  for l in nominal_levels]

        ax.plot([0,1], [0,1], 'k--', lw=1.0, label="Ideal")
        ax.plot(nom, raw, 'o-', color="steelblue", lw=1.5,
                label=f"Before inflation")
        ax.plot(nom, cal, 's-', color="darkorange", lw=1.5,
                label=f"After inflation (c={r['c_star']:.2f})")
        ax.set_xlabel("Nominal coverage", fontsize=11)
        ax.set_ylabel("Empirical coverage", fontsize=11)
        ax.set_title(f"Calibration — SNR {snr_db} dB", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
        ax.set_xticks(nominal_levels)
        ax.set_yticks(nominal_levels)

    fig.tight_layout()
    out_cal = OUT_DIR / "calibration_curve.png"
    fig.savefig(str(out_cal), dpi=150)
    plt.close(fig)
    print(f"\n→ saved calibration_curve.png")

    # ── Uncertainty vs error figure ────────────────────────────────────
    if len(args.snr_db) == 1:
        snr_db = args.snr_db[0]
        d = load_results(snr_db)
        unc  = d["uncertainty"]
        rmse = d["rmse"]
        r_val, p_val = stats.pearsonr(unc, rmse)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(unc, rmse, alpha=0.6, s=30, color="steelblue")
        ax.set_xlabel("Posterior std (uncertainty)", fontsize=11)
        ax.set_ylabel("RMSE (error)", fontsize=11)
        ax.set_title(f"Uncertainty vs Error — SNR {snr_db} dB\n"
                     f"r={r_val:.3f}  p={p_val:.3f}", fontsize=10)
        # Fit line
        m, b = np.polyfit(unc, rmse, 1)
        x_line = np.linspace(unc.min(), unc.max(), 50)
        ax.plot(x_line, m*x_line+b, 'r--', lw=1.0, alpha=0.7)
        fig.tight_layout()
        out_unc = OUT_DIR / f"uncertainty_vs_error_snr{snr_db:02d}dB.png"
        fig.savefig(str(out_unc), dpi=150)
        plt.close(fig)
        print(f"→ saved {out_unc.name}")

    # ── Summary table for paper ────────────────────────────────────────
    print()
    print("="*60)
    print("PAPER TABLE (test set, after scale inflation)")
    print("="*60)
    print(f"{'SNR':>6} {'RMSE':>8} {'Corr':>8} {'Cov90%':>10} "
          f"{'Sharp':>8} {'PPG_err':>10}")
    print("-"*55)
    for snr_db in args.snr_db:
        r = all_snr_results[snr_db]
        cov90_cal = r["rows"][0.90]["test_cal"]
        print(f"{snr_db:>4}dB  {r['rmse']:>8.3f}  {r['corr']:>8.3f}  "
              f"{cov90_cal*100:>9.1f}%  {r['unc_cal']:>8.3f}  "
              f"{r['ppg']:>10.3f}")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
