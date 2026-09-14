"""
data/preprocess.py
==================
Shared windowing, normalization, and QC for ECG signals.

Used by both Fatemeh (Part 1 / prior training) and Alex (Part 2 / evaluation).
Import this module; do not duplicate its logic in experiment scripts.

Design
------
*   Non-overlapping windows: one 4000-sample (8s) window per 10s PTB-XL record.
    Samples 0–3999 are used; the last 1000 samples are discarded.
*   Normalization is GLOBAL (training set mean/std), not per-window.
    This ensures the model sees consistent scale and the normalization
    stats can be saved once and reused at inference time.
*   QC gates are applied before normalization so decisions are made on
    physical (mV) values, not normalized ones.

Ownership
---------
Shared — both Fatemeh and Alex import from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Constants ──────────────────────────────────────────────────────────
WINDOW_SAMPLES = 4000      # 8s × 500 Hz
FS             = 500       # Hz


# ======================================================================
# QC
# ======================================================================

def qc_signal(window: np.ndarray,
              min_std   : float = 0.01,
              max_abs   : float = 5.0,
              max_nan   : int   = 0) -> tuple[bool, str]:
    """
    Quality check for one Lead II window in physical units (mV).

    Parameters
    ----------
    window : np.ndarray, shape (WINDOW_SAMPLES,)
        Raw (unnormalized) Lead II window.
    min_std : float
        Minimum acceptable standard deviation (mV).
        Rejects flatlines and near-constant signals.
    max_abs : float
        Maximum acceptable absolute amplitude (mV).
        PTB-XL physical range is ±5 mV; values beyond indicate
        clipping or ADC error.
    max_nan : int
        Maximum number of NaN/Inf values tolerated (default 0).

    Returns
    -------
    (passed, reason) : tuple[bool, str]
        passed=True if window passes all checks.
        reason is empty string if passed, otherwise names the failing check.
    """
    n_bad = int(np.sum(~np.isfinite(window)))
    if n_bad > max_nan:
        return False, f"nan_inf ({n_bad} bad samples)"

    if window.std() < min_std:
        return False, f"flatline (std={window.std():.4f} < {min_std})"

    if np.abs(window).max() > max_abs:
        return False, f"amplitude ({np.abs(window).max():.3f} > {max_abs} mV)"

    return True, ""


# ======================================================================
# Normalization stats
# ======================================================================

class NormStats:
    """
    Global normalization statistics (mean and std) computed from the
    training split and applied to all splits.

    Saved to / loaded from a JSON file so stats are computed once and
    reused across runs without reloading the full training set.

    Usage
    -----
    # Compute once from training data:
    stats = NormStats.from_windows(list_of_windows)
    stats.save("configs/norm_stats.json")

    # Load in all subsequent runs:
    stats = NormStats.load("configs/norm_stats.json")
    x_norm = stats.normalize(x_raw)
    x_raw  = stats.denormalize(x_norm)
    """

    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std  = float(max(std, 1e-8))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply (x - mean) / std. Works on any shape."""
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Invert normalization: x * std + mean."""
        return x * self.std + self.mean

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f, indent=2)
        print(f"[NormStats] saved to {path}  (mean={self.mean:.6f}, std={self.std:.6f})")

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        with open(str(path)) as f:
            d = json.load(f)
        print(f"[NormStats] loaded from {path}  (mean={d['mean']:.6f}, std={d['std']:.6f})")
        return cls(d["mean"], d["std"])

    @classmethod
    def from_windows(cls, windows: list[np.ndarray]) -> "NormStats":
        """
        Compute global mean and std from a list of 1-D windows (Welford).

        Parameters
        ----------
        windows : list[np.ndarray]
            Each element is a 1-D array of physical-unit samples.

        Returns
        -------
        NormStats
        """
        n, mean, M2 = 0, 0.0, 0.0
        for w in windows:
            for v in w.ravel():
                n    += 1
                delta = float(v) - mean
                mean += delta / n
                M2   += delta * (float(v) - mean)
        std = float(np.sqrt(M2 / max(1, n - 1)))
        print(f"[NormStats] computed from {len(windows)} windows: "
              f"mean={mean:.6f}, std={std:.6f}")
        return cls(mean, std)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NormStats":
        """Compute from a stacked array, shape (N, L) or (N*L,)."""
        return cls(float(arr.mean()), float(arr.std()))


# ======================================================================
# Window extraction
# ======================================================================

def extract_window(
    signal     : np.ndarray,
    window_len : int = WINDOW_SAMPLES,
    start      : int = 0,
) -> np.ndarray:
    """
    Extract one window from a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Full-length ECG signal (Lead II, physical units).
    window_len : int
        Window length in samples.
    start : int
        Start sample index.

    Returns
    -------
    np.ndarray, shape (window_len,)
        Extracted window. Returns zeros if the signal is too short.
    """
    if len(signal) < start + window_len:
        return np.zeros(window_len, dtype=np.float32)
    return signal[start : start + window_len].astype(np.float32)


def extract_nonoverlapping_windows(
    signal     : np.ndarray,
    window_len : int   = WINDOW_SAMPLES,
    qc_kwargs  : dict  | None = None,
) -> list[np.ndarray]:
    """
    Extract non-overlapping windows from a 1-D signal.

    For PTB-XL (5000 samples, 10s): one 4000-sample window starting at
    sample 0. The last 1000 samples are discarded.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Full-length Lead II signal in physical units (mV).
    window_len : int
        Window length in samples (default 4000 = 8s at 500 Hz).
    qc_kwargs : dict or None
        Keyword arguments passed to qc_signal(). None uses defaults.

    Returns
    -------
    list[np.ndarray]
        List of windows that passed QC. Each shape (window_len,).
        Empty list if signal is too short or all windows fail QC.
    """
    qc_kw = qc_kwargs or {}
    windows = []
    for start in range(0, len(signal) - window_len + 1, window_len):
        w = extract_window(signal, window_len, start)
        passed, _ = qc_signal(w, **qc_kw)
        if passed:
            windows.append(w)
    return windows


# ======================================================================
# Dataset
# ======================================================================

class ECGWindowDataset(Dataset):
    """
    In-memory PyTorch Dataset of normalized ECG windows.

    Built from a list of raw windows and a NormStats object.
    Used by both the prior training DataLoader and the Part 1
    evaluation pipeline.

    Parameters
    ----------
    windows : list[np.ndarray] or np.ndarray
        Raw (physical unit, mV) ECG windows, each shape (WINDOW_SAMPLES,).
    stats : NormStats
        Normalization statistics. Applied at __getitem__ time.
    meta : list[dict] or None
        Optional per-window metadata (ecg_id, strat_fold, etc.).
        If None, metadata returns empty dicts.

    Returns (per item)
    ------------------
    x_norm : torch.Tensor, shape (1, WINDOW_SAMPLES)
        Normalized Lead II window, channel-first for the U-Net.
    meta : dict
        Per-window metadata including normalization stats for
        denormalization at evaluation time.
    """

    def __init__(
        self,
        windows : list[np.ndarray] | np.ndarray,
        stats   : NormStats,
        meta    : Optional[list[dict]] = None,
    ) -> None:
        if isinstance(windows, np.ndarray):
            windows = [windows[i] for i in range(len(windows))]
        self.windows = windows
        self.stats   = stats
        self.meta    = meta or [{} for _ in windows]
        assert len(self.windows) == len(self.meta), \
            "windows and meta must have the same length"

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        w      = self.windows[idx]
        x_norm = self.stats.normalize(w)
        x_t    = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # (1, L)
        m      = {**self.meta[idx],
                  "mean": self.stats.mean,
                  "std" : self.stats.std}
        return x_t, m


# ======================================================================
# Build dataset from PTB-XL metadata
# ======================================================================

def build_dataset_from_ptbxl(
    metadata    : "pd.DataFrame",
    stats       : Optional[NormStats] = None,
    max_records : Optional[int] = None,
    qc_kwargs   : Optional[dict] = None,
    verbose     : bool = True,
) -> tuple[ECGWindowDataset, NormStats]:
    """
    Build an ECGWindowDataset from PTB-XL metadata by streaming records.

    Parameters
    ----------
    metadata : pd.DataFrame
        PTB-XL metadata for one split (from data.ptbxl.split_metadata).
    stats : NormStats or None
        If None, computes stats from the loaded windows (use for training
        split only). If provided, applies those stats (use for val/test).
    max_records : int or None
        Cap on records loaded. None = all. Use small values for debugging.
    qc_kwargs : dict or None
        Passed to qc_signal().
    verbose : bool

    Returns
    -------
    (dataset, stats) : tuple[ECGWindowDataset, NormStats]
        dataset : ECGWindowDataset ready for DataLoader.
        stats   : NormStats (computed if not provided, else the input stats).

    Example
    -------
    from data.ptbxl import load_metadata, split_metadata
    from data.preprocess import build_dataset_from_ptbxl, NormStats

    meta_all   = load_metadata()
    meta_train = split_metadata(meta_all, "train")
    meta_val   = split_metadata(meta_all, "val")

    train_ds, norm_stats = build_dataset_from_ptbxl(meta_train)
    norm_stats.save("configs/norm_stats.json")

    val_ds, _ = build_dataset_from_ptbxl(meta_val, stats=norm_stats)
    """
    from data.ptbxl import load_record, extract_lead_ii

    records = metadata.reset_index()
    if max_records is not None:
        records = records.iloc[:max_records]

    raw_windows, meta_list = [], []
    n_skipped = 0

    if verbose:
        print(f"[preprocess] loading {len(records)} records …")

    for i, (_, row) in enumerate(records.iterrows()):
        try:
            sig  = load_record(row["filename_hr"])
            lead = extract_lead_ii(sig)
        except Exception as e:
            if verbose:
                print(f"  [skip] {row.get('filename_hr', '?')}: {e}")
            n_skipped += 1
            continue

        windows = extract_nonoverlapping_windows(lead, qc_kwargs=qc_kwargs)
        for w in windows:
            raw_windows.append(w)
            meta_list.append({
                "ecg_id"    : int(row.get("ecg_id", i)),
                "strat_fold": int(row.get("strat_fold", 0)),
                "filename"  : str(row.get("filename_hr", "")),
            })

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(records)} records processed, "
                  f"{len(raw_windows)} windows so far")

    if verbose:
        print(f"[preprocess] done: {len(raw_windows)} windows "
              f"({n_skipped} records skipped)")

    # Compute or use provided normalization stats
    if stats is None:
        stats = NormStats.from_windows(raw_windows)

    dataset = ECGWindowDataset(raw_windows, stats, meta_list)
    return dataset, stats
