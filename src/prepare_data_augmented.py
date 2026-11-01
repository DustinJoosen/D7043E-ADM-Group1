"""
Preparation of ECG data (MIT-BIH Arrhythmia Database) with the addition of R-R features.
Retains the DS1/DS2 split and the `choose_val_records()` method from `prepare_data.py`.
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
        "wfdb package required, please install it:\n"
        "  pip install wfdb"
    ) from e

# --------------------------------------------------------------------------
# Splits (DS1/DS2)
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

# Mapping AAMI EC57
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
EXCLUDED_SYMBOLS = {"/", "f", "Q", "?", "!", "|", "~", "+", "x", "["}

FS = 360  # Native sampling rate (Hz)


def bandpass_filter(sig, fs=FS, low=0.5, high=40.0, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def select_channel(record):
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


def count_classes(record_id, local_dir=None):
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


def extract_beats_with_rr(record_id, pre, post, local_dir=None, exclusion_log=None, window_size=10):
    rec, ann = load_record(record_id, local_dir)
    ch = select_channel(rec)
    raw_sig = rec.p_signal[:, ch].astype(np.float64)
    filt_sig = bandpass_filter(raw_sig)

    samples = ann.sample
    symbols = ann.symbol
    n_ann = len(samples)

    # Preliminary calculation of RR intervals (in samples / FS for seconds)
    rr_intervals = np.diff(samples) / float(FS)  # in seconds

    windows, rr_features, labels = [], [], []
    excl_counts = {}

    for i in range(n_ann):
        sym = symbols[i]
        sample = samples[i]

        if sym not in AAMI_MAP:
            if sym in EXCLUDED_SYMBOLS or sym not in AAMI_MAP:
                excl_counts[sym] = excl_counts.get(sym, 0) + 1
            continue

        # Condition regarding the presence of neighbors for the calculation of pre_RR and post_RR
        if i == 0 or i == n_ann - 1:
            excl_counts["<edge-rr-unavailable>"] = excl_counts.get("<edge-rr-unavailable>", 0) + 1
            continue

        start, end = sample - pre, sample + post
        if start < 0 or end > len(filt_sig):
            excl_counts["<edge-of-record>"] = excl_counts.get("<edge-of-record>", 0) + 1
            continue

        # R-R Characteristics 
        pre_rr = (sample - samples[i - 1]) / float(FS)
        post_rr = (samples[i + 1] - sample) / float(FS)

        # Calculation of the local R-R average (sliding window)
        w_start = max(0, i - window_size // 2)
        w_end = min(len(rr_intervals), i + window_size // 2)
        local_avg_rr = np.mean(rr_intervals[w_start:w_end])
        local_rr = pre_rr / (local_avg_rr + 1e-8)

        # signal windowing and Z-score normalization
        window = filt_sig[start:end]
        std = window.std()
        window = (window - window.mean()) / std if std > 1e-8 else window - window.mean()

        windows.append(window)
        rr_features.append([pre_rr, post_rr, local_rr])
        labels.append(AAMI_MAP[sym])

    if exclusion_log is not None:
        exclusion_log[record_id] = excl_counts

    return (
        np.array(windows, dtype=np.float32),
        np.array(rr_features, dtype=np.float32),
        np.array(labels),
        rec.sig_name[ch],
    )


def build_split(record_ids, pre, post, local_dir, exclusion_log):
    X_sig_list, X_rr_list, y_list, rec_id_list = [], [], [], []
    for rid in record_ids:
        print(f"  Record extraction {rid} ...")
        X_sig, X_rr, y, lead = extract_beats_with_rr(rid, pre, post, local_dir, exclusion_log)
        X_sig_list.append(X_sig)
        X_rr_list.append(X_rr)
        y_list.append(y)
        rec_id_list.extend([rid] * len(y))
        print(f"    lead={lead} | seen beats={len(y)}")

    X_signal = np.concatenate(X_sig_list, axis=0)
    X_rr = np.concatenate(X_rr_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    rec_ids = np.array(rec_id_list)
    return X_signal, X_rr, y, rec_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./src/data_augmented")
    p.add_argument("--local_dir", default="./physionet.org/files/mitdb/1.0.0",
                       help="Local dir with pre-downloaded mitdb records; omit to stream from PhysioNet")
    p.add_argument("--pre", type=int, default=90)
    p.add_argument("--post", type=int, default=110)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    n_val = max(1, int(round(len(DS1_RECORDS) * args.val_frac)))
    val_records, ds1_counts = choose_val_records(
        DS1_RECORDS, n_val, args.val_frac, args.local_dir, args.seed
    )
    train_records = sorted(r for r in DS1_RECORDS if r not in val_records)
    test_records = DS2_RECORDS

    print(f"Train records ({len(train_records)}): {train_records}")
    print(f"Val records   ({len(val_records)}): {val_records}")
    print(f"Test records  ({len(test_records)}): {test_records}")

    exclusion_log = {}
    splits = {}
    for name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        print(f"\nBuilding {name} split...")
        X_sig, X_rr, y, rec_ids = build_split(records, args.pre, args.post, args.local_dir, exclusion_log)
        splits[name] = (X_sig, X_rr, y, rec_ids)
        np.savez_compressed(
            os.path.join(args.outdir, f"{name}.npz"),
            X_signal=X_sig,
            X_rr=X_rr,
            y=y,
            record_id=rec_ids,
        )
        print(f"{name}: {X_sig.shape[0]} beats, signal size={X_sig.shape[1]}, R-R features size={X_rr.shape[1]}")

    class_counts = []
    for name, (X_sig, X_rr, y, rec_ids) in splits.items():
        for cls in ["N", "S", "V", "F"]:
            class_counts.append({"split": name, "class": cls, "count": int((y == cls).sum())})
    class_df = pd.DataFrame(class_counts)
    class_df.to_csv(os.path.join(args.outdir, "class_distribution.csv"), index=False)

    with open(os.path.join(args.outdir, "exclusions.json"), "w") as f:
        json.dump(exclusion_log, f, indent=2)

    config = {
        "pre_samples": args.pre, "post_samples": args.post,
        "window_length": args.pre + args.post, "fs": FS,
        "rr_features": ["pre_RR", "post_RR", "local_RR"],
        "train_records": train_records, "val_records": val_records,
        "test_records": test_records, "seed": args.seed,
        "aami_map": AAMI_MAP,
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nCompleted successfully. Files generated in:", args.outdir)


if __name__ == "__main__":
    main()