"""
data/vitaldb.py
===============
VitalDB paired ECG + PPG loader for Phase 1 VI training.

VitalDB facts
-------------
*   6,388 surgical cases, 500 Hz waveforms
*   ECG track  : "SNUADC/ECG_II"   (Lead II, 500 Hz)
*   PPG track  : "SNUADC/PLETH"    (plethysmogram, 500 Hz)
*   Access via the vitaldb Python package (already installed)
*   No download needed — streams from api.vitaldb.net

Usage in the algorithm
----------------------
D_pair = VitalDB paired (ECG, PPG) windows
→ used ONLY in Phase 1 VI to learn q_{λ*}(φ)
→ NOT used in bootstrap prior training (that uses PTB-XL ECG only)
→ test split is patient-disjoint from VI training split

Ownership: Shared (Fatemeh uses for VI, Alex uses for Part 2)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import vitaldb
except ImportError:
    raise ImportError(
        "vitaldb package not found. Install with: pip install vitaldb"
    )

# ── Track names ────────────────────────────────────────────────────────
ECG_TRACK  = "SNUADC/ECG_II"
PPG_TRACK  = "SNUADC/PLETH"
FS         = 500          # Hz — both tracks are 500 Hz in VitalDB
INTERVAL   = 1.0 / FS    # seconds per sample

# ── Window config (matches PTB-XL preprocessing) ──────────────────────
WINDOW_SAMPLES = 4000     # 8s × 500 Hz
TOTAL_CASES    = 6388     # approximate VitalDB size

# ── Default patient-disjoint split ────────────────────────────────────
# Use case IDs directly — VitalDB cases are numbered 1..N
# 80% VI training, 10% VI val, 10% held out for Part 2 evaluation
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC  = 0.10  (remainder, used by Alex in Part 2)


# ======================================================================
# Case discovery
# ======================================================================

def find_paired_cases(max_cases: Optional[int] = None) -> list[int]:
    """
    Find VitalDB case IDs that have BOTH ECG_II and PLETH tracks.
    """
    print(f"[vitaldb] finding cases with {ECG_TRACK} and {PPG_TRACK} …")
    try:
        # Use newer API: load_cases with track filter
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            case_ids = vitaldb.find_cases(f"{ECG_TRACK},{PPG_TRACK}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to query VitalDB: {e}\n"
            "Check internet connection and that vitaldb package is installed."
        )

    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    print(f"[vitaldb] {len(case_ids)} cases have both ECG and PPG tracks")
    return list(case_ids)


def split_case_ids(
    case_ids  : list[int],
    seed      : int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split case IDs into patient-disjoint train / val / test sets.

    Parameters
    ----------
    case_ids : list[int]
    seed : int

    Returns
    -------
    (train_ids, val_ids, test_ids)
    """
    rng = random.Random(seed)
    ids = case_ids.copy()
    rng.shuffle(ids)

    n       = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train = ids[:n_train]
    val   = ids[n_train : n_train + n_val]
    test  = ids[n_train + n_val :]

    print(f"[vitaldb] split: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ======================================================================
# Single case loader
# ======================================================================

def load_case_signals(case_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load ECG and PPG signals for one case.

    Returns (ecg, ppg) as (T,) float32 arrays, or None if load fails.
    VitalDB returns a (T, 2) ndarray: col 0 = ECG, col 1 = PPG.
    """
    try:
        data = vitaldb.load_case(
            case_id,
            f"{ECG_TRACK},{PPG_TRACK}",
            interval=INTERVAL,
        )
    except Exception:
        return None

    if data is None:
        return None

    # data is (T, 2) ndarray — col 0 = ECG_II, col 1 = PLETH
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 2:
        return None

    ecg = data[:, 0].astype(np.float32)
    ppg = data[:, 1].astype(np.float32)

    n = min(len(ecg), len(ppg))
    if n < WINDOW_SAMPLES:
        return None

    # Drop leading/trailing NaNs by finding valid range
    valid = np.isfinite(ecg) & np.isfinite(ppg)
    if valid.sum() < WINDOW_SAMPLES:
        return None

    # Trim to the longest contiguous valid segment
    first = int(np.argmax(valid))
    last  = int(len(valid) - np.argmax(valid[::-1]))
    ecg = ecg[first:last]
    ppg = ppg[first:last]

    if len(ecg) < WINDOW_SAMPLES:
        return None

    return ecg, ppg


# ======================================================================
# QC for paired windows
# ======================================================================

def is_physical_units(ecg: np.ndarray, ppg: np.ndarray) -> bool:
    """
    Check if a case is in physical units (mV), not raw ADC counts.
    Called once per case before windowing to reject bad cases entirely.

    Physical ECG: std ~ 0.1–1.0 mV, abs max < 5 mV.
    ADC-scale ECG: std >> 1 (e.g. std=37 as seen in VitalDB case 3).
    """
    ecg_std = float(np.nanstd(ecg))
    ppg_std = float(np.nanstd(ppg))
    if ecg_std > 5.0 or ecg_std < 0.001:
        return False
    if ppg_std > 100.0 or ppg_std < 0.001:
        return False
    return True


def qc_paired_window(
    ecg    : np.ndarray,
    ppg    : np.ndarray,
    min_std: float = 0.01,
    max_abs: float = 5.0,
) -> bool:
    """
    Quality check for a paired (ECG, PPG) window.

    Both signals must pass: no NaN/Inf, not flat, not clipped.
    ECG threshold: ±5 mV rejects ADC-scale signals.
    """
    for sig in [ecg, ppg]:
        if not np.isfinite(sig).all():
            return False
        if sig.std() < min_std:
            return False
    if np.abs(ecg).max() > max_abs:
        return False
    if np.abs(ppg).max() > 100.0:
        return False
    return True


# ======================================================================
# Dataset
# ======================================================================

class VitalDBPairedDataset(Dataset):
    """
    PyTorch Dataset of paired (ECG, PPG) windows from VitalDB.

    Used exclusively for Phase 1 VI training — NOT for MCMC inference.

    Each item returns (ecg_norm, ppg_norm) where both signals are
    normalized using their respective global statistics so they live
    on compatible scales.

    Parameters
    ----------
    case_ids : list[int]
        VitalDB case IDs for this split.
    ecg_stats : tuple[float, float]
        (mean, std) for ECG normalization.
    ppg_stats : tuple[float, float]
        (mean, std) for PPG normalization.
    max_cases : int or None
        Cap on cases to load (for debugging).
    verbose : bool
    """

    def __init__(
        self,
        case_ids   : list[int],
        ecg_stats  : tuple[float, float],
        ppg_stats  : tuple[float, float],
        max_cases  : Optional[int] = None,
        verbose    : bool = True,
    ) -> None:
        self.ecg_mean, self.ecg_std = ecg_stats
        self.ppg_mean, self.ppg_std = ppg_stats
        self.ecg_std = max(self.ecg_std, 1e-8)
        self.ppg_std = max(self.ppg_std, 1e-8)

        self.ecg_windows : list[np.ndarray] = []
        self.ppg_windows : list[np.ndarray] = []

        cases = case_ids[:max_cases] if max_cases else case_ids
        n_skip = 0

        if verbose:
            print(f"[vitaldb] loading {len(cases)} cases …")

        for i, cid in enumerate(cases):
            result = load_case_signals(cid)
            if result is None:
                n_skip += 1
                continue

            ecg_sig, ppg_sig = result

            # Reject ADC-scale cases before windowing
            if not is_physical_units(ecg_sig, ppg_sig):
                n_skip += 1
                continue

            # Extract non-overlapping windows
            for start in range(0, len(ecg_sig) - WINDOW_SAMPLES + 1,
                               WINDOW_SAMPLES):
                ecg_w = ecg_sig[start : start + WINDOW_SAMPLES]
                ppg_w = ppg_sig[start : start + WINDOW_SAMPLES]
                if qc_paired_window(ecg_w, ppg_w):
                    self.ecg_windows.append(ecg_w)
                    self.ppg_windows.append(ppg_w)

            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(cases)} cases  "
                      f"{len(self.ecg_windows)} windows")

        if verbose:
            print(f"[vitaldb] {len(self.ecg_windows)} paired windows "
                  f"({n_skip} cases skipped)")

        if not self.ecg_windows:
            raise RuntimeError(
                "No paired windows loaded from VitalDB. "
                "Check internet connection and track names."
            )

    def __len__(self) -> int:
        return len(self.ecg_windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ecg = (self.ecg_windows[idx] - self.ecg_mean) / self.ecg_std
        ppg = (self.ppg_windows[idx] - self.ppg_mean) / self.ppg_std
        return (
            torch.tensor(ecg, dtype=torch.float32).unsqueeze(0),  # (1, L)
            torch.tensor(ppg, dtype=torch.float32).unsqueeze(0),  # (1, L)
        )


# ======================================================================
# Global stats computation
# ======================================================================

def compute_paired_stats(
    case_ids   : list[int],
    max_cases  : Optional[int] = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute global (mean, std) for ECG and PPG from the training cases.

    Parameters
    ----------
    case_ids : list[int]   training case IDs
    max_cases : int or None

    Returns
    -------
    ecg_stats : (mean, std)
    ppg_stats : (mean, std)
    """
    cases = case_ids[:max_cases] if max_cases else case_ids
    ecg_vals, ppg_vals = [], []

    print(f"[vitaldb] computing normalization stats from {len(cases)} cases …")
    for cid in cases:
        result = load_case_signals(cid)
        if result is None:
            continue
        ecg_sig, ppg_sig = result
        if not is_physical_units(ecg_sig, ppg_sig):
            continue
        ecg_vals.append(ecg_sig)
        ppg_vals.append(ppg_sig)

    ecg_all = np.concatenate(ecg_vals)
    ppg_all = np.concatenate(ppg_vals)

    ecg_stats = (float(ecg_all.mean()), float(ecg_all.std()))
    ppg_stats = (float(ppg_all.mean()), float(ppg_all.std()))

    print(f"  ECG: mean={ecg_stats[0]:.4f}  std={ecg_stats[1]:.4f}")
    print(f"  PPG: mean={ppg_stats[0]:.4f}  std={ppg_stats[1]:.4f}")
    return ecg_stats, ppg_stats


# ======================================================================
# Build loaders
# ======================================================================

def build_paired_loaders(
    batch_size  : int = 16,
    max_cases   : Optional[int] = None,
    seed        : int = 42,
    num_workers : int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Full pipeline: discover cases → split → compute stats → build loaders.

    Parameters
    ----------
    batch_size : int
    max_cases : int or None   cap for debugging
    seed : int
    num_workers : int

    Returns
    -------
    (train_loader, val_loader, info_dict)
        info_dict contains ecg_stats, ppg_stats, case splits.
    """
    case_ids = find_paired_cases(max_cases=max_cases)
    train_ids, val_ids, test_ids = split_case_ids(case_ids, seed=seed)

    # Compute stats from training cases only
    ecg_stats, ppg_stats = compute_paired_stats(train_ids, max_cases=max_cases)

    train_ds = VitalDBPairedDataset(train_ids, ecg_stats, ppg_stats,
                                    max_cases=max_cases)
    val_ds   = VitalDBPairedDataset(val_ids,   ecg_stats, ppg_stats,
                                    max_cases=max_cases)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    info = {
        "ecg_stats" : ecg_stats,
        "ppg_stats" : ppg_stats,
        "train_ids" : train_ids,
        "val_ids"   : val_ids,
        "test_ids"  : test_ids,
    }
    return train_loader, val_loader, info


# ======================================================================
# Verification
# ======================================================================

def verify_access(n_cases: int = 3) -> None:
    """
    Quick smoke test: load n_cases and print signal shapes.

        python -c "from data.vitaldb import verify_access; verify_access()"
    """
    print("Testing VitalDB access …")
    try:
        ids = vitaldb.find_cases(f"{ECG_TRACK},{PPG_TRACK}")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    print(f"  Found {len(ids)} cases with ECG+PPG")
    ok = 0
    for cid in ids[:n_cases * 3]:
        result = load_case_signals(cid)
        if result is None:
            continue
        ecg, ppg = result
        dur = len(ecg) / FS
        print(f"  case {cid}: ECG {ecg.shape}  PPG {ppg.shape}  "
              f"{dur:.0f}s  ECG_std={ecg.std():.3f}")
        ok += 1
        if ok >= n_cases:
            break

    if ok > 0:
        print("VitalDB ACCESS OK")
    else:
        print("WARNING: no cases loaded successfully")


if __name__ == "__main__":
    verify_access()
