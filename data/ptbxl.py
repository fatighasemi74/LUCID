"""
data/ptbxl.py
=============
PTB-XL ECG loader via the PhysioNet wfdb streaming API.

No bulk download required — records are fetched on demand.
Credentials are cached by wfdb after the first authenticated call.

One-time setup on your machine
--------------------------------
    pip install wfdb pandas
    python -c "
        import wfdb
        wfdb.dl_database('ptb-xl/1.0.3', dl_dir='./ptbxl_test',
                         records=['records500/00000/00001_hr'])
    "
    # enter your PhysioNet username and password when prompted
    # credentials are cached; all future calls are silent

Dataset facts
-------------
*   21,799 12-lead ECG recordings, 10 seconds each.
*   Two sampling rates: records100 (100 Hz) and records500 (500 Hz).
*   We use records500 exclusively — QRS/QTc demand it.
*   Lead layout (0-indexed):  0=I  1=II  2=III  3=aVR  4=aVL  5=aVF
                               6=V1 7=V2  8=V3   9=V4  10=V5  11=V6
*   Lead II = index 1. All models are trained on Lead II only.
*   Metadata is in ptbxl_database.csv (streaming) or local cache.
*   SCP diagnostic codes are in scp_statements.csv.
*   Patient-disjoint train/val/test split: use the strat_fold column
    (1–10). Standard split: folds 1–8 train, 9 val, 10 test.

Ownership
---------
Shared module — both Fatemeh and Alex use this.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wfdb
except ImportError:
    raise ImportError(
        "wfdb is required: pip install wfdb\n"
        "Then authenticate once:\n"
        "  python -c \"import wfdb; wfdb.dl_database('ptb-xl/1.0.3', "
        "dl_dir='./tmp', records=['records500/00000/00001_hr'])\""
    )

# ── Constants ──────────────────────────────────────────────────────────
PN_DIR       = "ptb-xl/1.0.3"          # wfdb pn_dir argument
FS_500       = 500                      # Hz — always use this
LEAD_II      = 1                        # index in the 12-lead array
WINDOW_LEN   = 4000                     # samples = 8 s × 500 Hz
N_LEADS      = 12

# Standard patient-disjoint split (strat_fold column)
TRAIN_FOLDS  = list(range(1, 9))        # folds 1–8
VAL_FOLDS    = [9]
TEST_FOLDS   = [10]


# ── Metadata loader ────────────────────────────────────────────────────

def load_metadata(local_csv: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load PTB-XL metadata from ptbxl_database.csv.

    Parameters
    ----------
    local_csv : str or Path or None
        Path to a locally cached ptbxl_database.csv.
        If None, streams from PhysioNet (requires credentials).

    Returns
    -------
    pd.DataFrame
        One row per recording. Key columns:
        ecg_id, patient_id, strat_fold, filename_hr, scp_codes,
        age, sex, heart_axis, ...
    """
    if local_csv is not None:
        return pd.read_csv(str(local_csv), index_col="ecg_id")

    # Stream the CSV from PhysioNet
    import urllib.request
    url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
    with urllib.request.urlopen(url) as f:
        df = pd.read_csv(io.BytesIO(f.read()), index_col="ecg_id")
    return df


def split_metadata(
    df: pd.DataFrame,
    split: str = "train",
) -> pd.DataFrame:
    """
    Filter metadata to train / val / test rows using strat_fold.

    Parameters
    ----------
    df : pd.DataFrame
        Full metadata from load_metadata().
    split : str
        "train", "val", or "test".

    Returns
    -------
    pd.DataFrame
        Filtered rows for the requested split.
    """
    folds = {"train": TRAIN_FOLDS, "val": VAL_FOLDS, "test": TEST_FOLDS}
    if split not in folds:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")
    return df[df["strat_fold"].isin(folds[split])].copy()


# ── Single record loader ───────────────────────────────────────────────

def load_record(filename_hr: str) -> np.ndarray:
    """
    Stream one 500 Hz PTB-XL record from PhysioNet.

    Parameters
    ----------
    filename_hr : str
        The filename_hr field from ptbxl_database.csv, e.g.
        "records500/00000/00001_hr"  (without extension).

    Returns
    -------
    np.ndarray, shape (5000, 12)
        Raw signal in physical units (mV). 10 seconds × 500 Hz = 5000 samples.
        Columns = 12 leads in standard order.

    Notes
    -----
    wfdb 4.3.1 requires the subdirectory path in pn_dir and only the
    bare record name (no subdirs) as the first argument. We split
    filename_hr accordingly:
        filename_hr = "records500/00000/00001_hr"
        → pn_dir    = "ptb-xl/1.0.3/records500/00000"
        → rec_name  = "00001_hr"
    which resolves to the correct PhysioNet URL:
        https://physionet.org/files/ptb-xl/1.0.3/records500/00000/00001_hr.hea
    """
    # Normalize separators and split into subdir + bare record name
    parts = filename_hr.replace("\\", "/").rsplit("/", 1)
    if len(parts) == 2:
        pn_subdir = PN_DIR + "/" + parts[0]
        rec_name  = parts[1]
    else:
        pn_subdir = PN_DIR
        rec_name  = parts[0]

    rec = wfdb.rdrecord(rec_name, pn_dir=pn_subdir)
    if rec.fs != FS_500:
        raise ValueError(
            f"Expected 500 Hz record, got {rec.fs} Hz for {filename_hr}. "
            "Check that filename_hr points to records500/, not records100/."
        )
    return rec.p_signal.astype(np.float32)   # (5000, 12)


def extract_lead_ii(signal: np.ndarray) -> np.ndarray:
    """
    Extract Lead II from a multi-lead signal array.

    Parameters
    ----------
    signal : np.ndarray, shape (T, 12)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    return signal[:, LEAD_II]


# ── QC ────────────────────────────────────────────────────────────────

def qc_window(window: np.ndarray, min_std: float = 0.01) -> bool:
    """
    Basic quality check for a single Lead II window.

    Returns True if the window passes, False if it should be discarded.

    Checks
    ------
    *   No NaN or Inf values.
    *   Standard deviation above min_std (rejects flatlines).
    *   No clipping: max < 5 mV, min > -5 mV (PTB-XL physical range).
    """
    if not np.isfinite(window).all():
        return False
    if window.std() < min_std:
        return False
    if window.max() > 5.0 or window.min() < -5.0:
        return False
    return True


# ── Dataset ───────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for Lead II ECG windows from PTB-XL.

    Each item is one normalized 8-second Lead II window at 500 Hz.
    Windows are extracted by sliding a WINDOW_LEN-sample window over
    the 5000-sample (10 s) recording with the given step size.

    Parameters
    ----------
    metadata : pd.DataFrame
        Filtered metadata (one split), from split_metadata().
    global_mean : float
        Training set mean for normalization. Pass 0.0 to skip.
    global_std : float
        Training set std for normalization. Pass 1.0 to skip.
    step : int
        Hop between windows in samples. Default 2500 → 2 windows per record
        (no overlap for 5000-sample records with 4000-sample windows).
    max_records : int or None
        Cap on number of records to load (for fast debugging). None = all.

    Returns (per item)
    ------------------
    x_norm : torch.Tensor, shape (1, WINDOW_LEN)
        Normalized Lead II ECG window.
    meta : dict
        {"ecg_id": int, "window_idx": int, "mean": float, "std": float}
        Stores the per-record mean/std used in normalization for
        denormalization at evaluation time.

    Notes
    -----
    *   Normalization is global (training set statistics), not per-window,
        so the model sees consistent scale across all windows.
    *   Records that fail QC are silently skipped.
    *   This class streams records lazily on first access and caches them
        in memory. For large-scale training, download PTB-XL locally and
        set local_dir in the class method.
    """

    def __init__(
        self,
        metadata    : pd.DataFrame,
        global_mean : float = 0.0,
        global_std  : float = 1.0,
        step        : int = 2500,
        max_records : Optional[int] = None,
    ) -> None:
        self.global_mean = global_mean
        self.global_std  = max(global_std, 1e-8)
        self._index: list[dict] = []   # list of {filename, window_start, ecg_id}

        records = metadata.reset_index()
        if max_records is not None:
            records = records.iloc[:max_records]

        print(f"[PTBXLDataset] indexing {len(records)} records …")
        for _, row in records.iterrows():
            fn = row["filename_hr"]
            for start in range(0, 5000 - WINDOW_LEN + 1, step):
                self._index.append({
                    "filename"   : fn,
                    "ecg_id"     : int(row["ecg_id"]) if "ecg_id" in row else 0,
                    "window_start": start,
                })

        self._cache: dict[str, np.ndarray] = {}   # filename → full signal
        print(f"[PTBXLDataset] {len(self._index)} windows indexed")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        entry  = self._index[idx]
        fn     = entry["filename"]
        start  = entry["window_start"]

        # Load and cache the full record
        if fn not in self._cache:
            sig = load_record(fn)                  # (5000, 12)
            self._cache[fn] = extract_lead_ii(sig) # (5000,)

        window = self._cache[fn][start : start + WINDOW_LEN]   # (WINDOW_LEN,)

        if not qc_window(window):
            # Return zeros with a flag — DataLoader caller should filter these
            x_norm = torch.zeros(1, WINDOW_LEN)
            return x_norm, {"ecg_id": entry["ecg_id"], "window_idx": idx,
                            "mean": 0.0, "std": 1.0, "qc_fail": True}

        x_norm = (window - self.global_mean) / self.global_std
        x_norm = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # (1, L)

        return x_norm, {
            "ecg_id"    : entry["ecg_id"],
            "window_idx": idx,
            "mean"      : self.global_mean,
            "std"       : self.global_std,
            "qc_fail"   : False,
        }


# ── Global normalization stats ─────────────────────────────────────────

def compute_global_stats(
    metadata    : pd.DataFrame,
    max_records : Optional[int] = None,
    step        : int = 2500,
) -> tuple[float, float]:
    """
    Compute global mean and std over Lead II from a metadata split.

    Uses Welford's online algorithm — does not load all signals at once.

    Parameters
    ----------
    metadata : pd.DataFrame
        Training split metadata.
    max_records : int or None
        Cap for fast approximation.
    step : int
        Window hop.

    Returns
    -------
    (mean, std) : tuple[float, float]
    """
    records = metadata.reset_index()
    if max_records is not None:
        records = records.iloc[:max_records]

    n, mean, M2 = 0, 0.0, 0.0
    for _, row in records.iterrows():
        try:
            sig = load_record(row["filename_hr"])
            lead = extract_lead_ii(sig)
        except Exception:
            continue
        for start in range(0, 5000 - WINDOW_LEN + 1, step):
            w = lead[start : start + WINDOW_LEN]
            if not qc_window(w):
                continue
            for v in w:
                n += 1
                delta = float(v) - mean
                mean += delta / n
                M2   += delta * (float(v) - mean)

    std = float(np.sqrt(M2 / max(1, n - 1)))
    return float(mean), std


# ── Setup verification script ──────────────────────────────────────────

def verify_access() -> None:
    """
    Quick smoke test: stream one record, print shape and Lead II stats.
    Run this after setting up PhysioNet credentials:

        python -c "from data.ptbxl import verify_access; verify_access()"
    """
    print("Testing PTB-XL PhysioNet API access …")
    try:
        sig = load_record("records500/00000/00001_hr")
    except Exception as e:
        print(f"FAILED: {e}")
        print("\nSetup instructions:")
        print("  pip install wfdb")
        print("  python -c \"import wfdb; wfdb.dl_database('ptb-xl/1.0.3',")
        print("      dl_dir='./tmp', records=['records500/00000/00001_hr'])\"")
        print("  (enter your PhysioNet username and password when prompted)")
        return

    lead_ii = extract_lead_ii(sig)
    print(f"  signal shape : {sig.shape}  (5000 samples × 12 leads)")
    print(f"  sampling rate: {FS_500} Hz  →  {sig.shape[0]/FS_500:.1f} s")
    print(f"  Lead II range: [{lead_ii.min():.3f}, {lead_ii.max():.3f}] mV")
    print(f"  Lead II std  : {lead_ii.std():.4f} mV")
    print("PTB-XL ACCESS OK")


if __name__ == "__main__":
    verify_access()
