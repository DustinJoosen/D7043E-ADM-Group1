"""
prepare_data.py

CRISP-DM Data Preparation for MIT-BIH Arrhythmia ECG heartbeat classification
(N / S / V / F, per assignment spec).

What this does:
  1. Streams (or reads locally cached) MIT-BIH Arrhythmia DB records via `wfdb`.
  2. Selects a single ECG lead per record (MLII preferred, else channel 0).
  3. Extracts fixed-size beat windows centered on each annotated R-peak.
  4. Maps raw MIT-BIH annotation symbols to the AAMI N/S/V/F classes
     (paced / unclassifiable beats excluded and logged).
  5. Splits records into DS1 (train/val) and DS2 (final test) -- by RECORD,
     never by individual beat, to avoid patient leakage.
     The DS1 train/val split is chosen by searching for a record subset whose
     S/V/F class shares are closest to --val_frac, instead of a blind shuffle
     (a blind shuffle can put the few F-heavy records entirely into val).
  6. Bandpass filters + per-beat normalizes each window.
  7. Saves .npz arrays for train/val/test, plus CSV summaries for EDA
     (e.g. to load into Tableau).

Usage:
    pip install wfdb scipy numpy pandas
    python prepare_data.py --outdir ./data --local_dir ./mit-bih-arrhythmia-database-1.0.0 --pre 90 --post 110 --val_frac 0.2 --seed 42

Notes:
  - wfdb can stream directly from PhysioNet (pn_dir='mitdb') so you do not need
    to manually download the raw dataset -- just internet access when you run
    this on your own machine.
  - Re-run with --local_dir pointing at a local copy if you've already
    downloaded the records (e.g. via `wfdb.dl_database('mitdb', './mitdb')`).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

try:
    import wfdb
except ImportError as e:
    raise SystemExit(
        "This script requires the 'wfdb' package. Install with:\n"
        "  pip install wfdb"
    ) from e

# --------------------------------------------------------------------------
# Record splits (from the assignment spec -- the standard inter-patient
# DS1/DS2 partition). Do not change these without documenting why.
# --------------------------------------------------------------------------
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

# --------------------------------------------------------------------------
# AAMI EC57 annotation -> class mapping.
# Anything not listed here (paced beats, unclassifiable, etc.) is excluded
# and logged, per the assignment's "document all exclusions" requirement.
# --------------------------------------------------------------------------
AAMI_MAP = {
    # N -- Normal-type beats
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S -- Supraventricular ectopic beats
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V -- Ventricular ectopic beats
    "V": "V", "E": "V",
    # F -- Fusion beats
    "F": "F",
}
EXCLUDED_SYMBOLS = {"/", "f", "Q", "?", "!", "|", "~", "+", "x", "["}  # paced/unclassifiable/non-beat markers

FS = 360  # MIT-BIH native sampling frequency (Hz)


def bandpass_filter(sig, fs=FS, low=0.5, high=40.0, order=3):
    """Remove baseline wander (<0.5Hz) and high-frequency noise (>40Hz)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def select_channel(record):
    """Prefer MLII lead (standard for beat morphology); else channel 0."""
    names = [n.upper() for n in record.sig_name]
    if "MLII" in names:
        return names.index("MLII")
    return 0


def load_record(record_id, local_dir=None):
    if local_dir:
        rec = wfdb.rdrecord(os.path.join(local_dir, record_id))
        ann = wfdb.rdann(os.path.join(local_dir, record_id), "atr")
    else:
        rec = wfdb.rdrecord(record_id, pn_dir="mitdb")
        ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb")
    return rec, ann


# --------------------------------------------------------------------------
# Class-balanced record-level train/val selection
# --------------------------------------------------------------------------
def count_classes(record_id, local_dir=None):
    """Cheap per-record AAMI class counts from the .atr file only (no signal load)."""
    if local_dir:
        ann = wfdb.rdann(os.path.join(local_dir, record_id), "atr")
    else:
        ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb")
    counts = dict.fromkeys("NSVF", 0)
    for sym in ann.symbol:
        cls = AAMI_MAP.get(sym)
        if cls:
            counts[cls] += 1
    return counts


def choose_val_records(records, n_val, val_frac, local_dir, seed, tries=5000):
    """
    Pick n_val records for validation (the rest go to train) so that the share of
    each minority class (S, V, F) in val is as close to val_frac as possible.
    Splitting is still strictly by RECORD, so there is no patient leakage.
    Returns (sorted val record list, per-record class counts).
    """
    counts = {r: count_classes(r, local_dir) for r in records}
    totals = {c: sum(counts[r][c] for r in records) for c in "NSVF"}
    rng = np.random.default_rng(seed)

    best, best_score = None, float("inf")
    for _ in range(tries):
        val = [str(r) for r in rng.choice(records, n_val, replace=False)]
        score = sum(
            (sum(counts[r][c] for r in val) / max(totals[c], 1) - val_frac) ** 2
            for c in "SVF"
        )
        if score < best_score:
            best, best_score = val, score
    return sorted(best), counts


def extract_beats(record_id, pre, post, local_dir=None, exclusion_log=None):
    """Return (windows, labels) for one record, plus per-record exclusion counts."""
    rec, ann = load_record(record_id, local_dir)
    ch = select_channel(rec)
    raw_sig = rec.p_signal[:, ch].astype(np.float64)
    filt_sig = bandpass_filter(raw_sig)

    windows, labels = [], []
    excl_counts = {}

    for sample, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI_MAP:
            if sym in EXCLUDED_SYMBOLS or sym not in AAMI_MAP:
                excl_counts[sym] = excl_counts.get(sym, 0) + 1
            continue

        start, end = sample - pre, sample + post
        if start < 0 or end > len(filt_sig):
            excl_counts["<edge-of-record>"] = excl_counts.get("<edge-of-record>", 0) + 1
            continue

        window = filt_sig[start:end]
        # Per-beat z-score normalization
        std = window.std()
        window = (window - window.mean()) / std if std > 1e-8 else window - window.mean()

        windows.append(window)
        labels.append(AAMI_MAP[sym])

    if exclusion_log is not None:
        exclusion_log[record_id] = excl_counts

    return np.array(windows, dtype=np.float32), np.array(labels), rec.sig_name[ch]


def build_split(record_ids, pre, post, local_dir, exclusion_log):
    X_list, y_list, rec_id_list = [], [], []
    for rid in record_ids:
        print(f"  Extracting record {rid} ...")
        X, y, lead = extract_beats(rid, pre, post, local_dir, exclusion_log)
        X_list.append(X)
        y_list.append(y)
        rec_id_list.extend([rid] * len(y))
        print(f"    lead={lead}  beats={len(y)}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    rec_ids = np.array(rec_id_list)
    return X, y, rec_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./data")
    p.add_argument("--local_dir", default=None,
                    help="Local dir with pre-downloaded mitdb records; omit to stream from PhysioNet")
    p.add_argument("--pre", type=int, default=90, help="samples before R-peak")
    p.add_argument("--post", type=int, default=110, help="samples after R-peak")
    p.add_argument("--val_frac", type=float, default=0.2, help="target fraction of DS1 beats (per minority class) held out for validation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Split DS1 by RECORD into train/val (not by beat), balancing S/V/F shares
    n_val = max(1, int(round(len(DS1_RECORDS) * args.val_frac)))
    val_records, ds1_counts = choose_val_records(
        DS1_RECORDS, n_val, args.val_frac, args.local_dir, args.seed
    )
    train_records = sorted(r for r in DS1_RECORDS if r not in val_records)
    test_records = DS2_RECORDS

    print(f"Train records ({len(train_records)}): {train_records}")
    print(f"Val records   ({len(val_records)}): {val_records}")
    print(f"Test records  ({len(test_records)}): {test_records}")

    # Show the class totals the chosen split will produce (from .atr counts)
    for name, recs in [("train", train_records), ("val", val_records)]:
        tot = {c: sum(ds1_counts[r][c] for r in recs) for c in "NSVF"}
        print(f"  {name} planned class counts: {tot}")

    exclusion_log = {}
    splits = {}
    for name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        print(f"\nBuilding {name} split ...")
        X, y, rec_ids = build_split(records, args.pre, args.post, args.local_dir, exclusion_log)
        splits[name] = (X, y, rec_ids)
        np.savez_compressed(
            os.path.join(args.outdir, f"{name}.npz"),
            X=X, y=y, record_id=rec_ids,
        )
        print(f"{name}: {X.shape[0]} beats, window length {X.shape[1]}")

    # ---- EDA summaries (for Tableau / your report) ----
    class_counts = []
    for name, (X, y, rec_ids) in splits.items():
        for cls in ["N", "S", "V", "F"]:
            class_counts.append({"split": name, "class": cls, "count": int((y == cls).sum())})
    class_df = pd.DataFrame(class_counts)
    class_df.to_csv(os.path.join(args.outdir, "class_distribution.csv"), index=False)
    print("\nFinal class distribution:")
    print(class_df.pivot(index="class", columns="split", values="count"))

    record_counts = []
    for name, (X, y, rec_ids) in splits.items():
        for rid in np.unique(rec_ids):
            mask = rec_ids == rid
            row = {"split": name, "record_id": rid, "total_beats": int(mask.sum())}
            for cls in ["N", "S", "V", "F"]:
                row[cls] = int((y[mask] == cls).sum())
            record_counts.append(row)
    pd.DataFrame(record_counts).to_csv(os.path.join(args.outdir, "record_distribution.csv"), index=False)

    with open(os.path.join(args.outdir, "exclusions.json"), "w") as f:
        json.dump(exclusion_log, f, indent=2)

    config = {
        "pre_samples": args.pre, "post_samples": args.post,
        "window_length": args.pre + args.post, "fs": FS,
        "train_records": train_records, "val_records": val_records,
        "test_records": test_records, "seed": args.seed,
        "val_selection": "class-balanced record search over S/V/F shares",
        "aami_map": AAMI_MAP,
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone. Outputs written to:", args.outdir)
    print(" - train.npz / val.npz / test.npz  (X: beat windows, y: labels, record_id: source record)")
    print(" - class_distribution.csv, record_distribution.csv  (import into Tableau for EDA)")
    print(" - exclusions.json  (excluded annotation symbols per record, for your report)")
    print(" - config.json  (reproducibility: window size, seed, splits)")


if __name__ == "__main__":
    main()