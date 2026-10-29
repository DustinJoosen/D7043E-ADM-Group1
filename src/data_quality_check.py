"""
data_quality_check.py

CRISP-DM Section 2.4 (Data Quality) investigation.

Scans the RAW MIT-BIH records (before beat extraction/filtering) and checks:
  - missing data (NaN/Inf, truncated recordings)
  - extreme amplitudes (physiologically implausible mV values, ADC saturation)
  - corrupted/flat segments (flatlined stretches -- electrode disconnect)
  - differences between recordings (per-record signal statistics)
  - differences between ECG channels (which lead each record uses, how
    similar/dissimilar the two channels are)
  - possible noisy regions (windows with unusually high high-frequency energy)

This is deliberately run on the RAW signal, not the filtered/segmented beat
windows -- Section 2.4 is about characterizing the SOURCE data quality, which
is a different question from "did my preprocessing work."

Usage:
    python data_quality_check.py --outdir ./data/quality
    python data_quality_check.py --outdir ./data/quality --local_dir "./mit-bih-arrhythmia-database-1.0.0"
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt

FS = 360
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Physiological sanity bound: a real ECG in mV essentially never exceeds this.
# Anything beyond it is almost certainly electrode artifact or a data issue.
PLAUSIBLE_MV_LIMIT = 5.0

FLAT_WINDOW_SEC = 1.0          # window size for flatline detection
FLAT_STD_THRESHOLD = 0.005     # mV std below this in a window = "flat"
NOISE_WINDOW_SEC = 1.0         # window size for noise-energy scoring
NOISE_PERCENTILE_FLAG = 95     # flag windows in the top 5% of noise energy, per record


def load_raw(record_id, local_dir=None):
    if local_dir:
        rec = wfdb.rdrecord(os.path.join(local_dir, record_id))
    else:
        rec = wfdb.rdrecord(record_id, pn_dir="mitdb")
    return rec


def high_freq_residual(sig, fs=FS):
    """Signal minus its 0.5-40Hz bandpassed version = energy outside the
    clean ECG band, i.e. a proxy for noise contamination."""
    nyq = fs / 2.0
    b, a = butter(3, [0.5 / nyq, 40 / nyq], btype="band")
    clean = filtfilt(b, a, sig)
    return sig - clean


def windowed_std(x, win_len):
    n_win = len(x) // win_len
    trimmed = x[: n_win * win_len].reshape(n_win, win_len)
    return trimmed.std(axis=1)


def analyze_record(record_id, local_dir=None):
    rec = load_raw(record_id, local_dir)
    sig = rec.p_signal  # shape (n_samples, n_channels), physical units (mV)
    lead_names = rec.sig_name
    n_samples, n_channels = sig.shape
    duration_min = n_samples / FS / 60.0

    issues = []
    row = {
        "record_id": record_id,
        "n_channels": n_channels,
        "leads": ",".join(lead_names),
        "n_samples": n_samples,
        "duration_min": round(duration_min, 2),
    }

    # ---- Missing data ----
    n_nan = int(np.isnan(sig).sum())
    n_inf = int(np.isinf(sig).sum())
    row["n_nan"] = n_nan
    row["n_inf"] = n_inf
    if n_nan or n_inf:
        issues.append(f"{n_nan} NaN / {n_inf} Inf samples found")
    expected_samples = int(30 * 60 * FS * 0.95)  # allow some tolerance below ~30min
    if n_samples < expected_samples:
        issues.append(f"recording shorter than expected ({duration_min:.1f} min)")

    # ---- Extreme amplitudes ----
    ch_max_abs = []
    for c in range(n_channels):
        col = sig[:, c]
        col = col[~np.isnan(col)]
        cmax = float(np.max(np.abs(col))) if len(col) else float("nan")
        ch_max_abs.append(cmax)
        pct_extreme = float((np.abs(col) > PLAUSIBLE_MV_LIMIT).mean() * 100) if len(col) else 0.0
        row[f"ch{c}_max_abs_mV"] = round(cmax, 3)
        row[f"ch{c}_pct_over_{PLAUSIBLE_MV_LIMIT}mV"] = round(pct_extreme, 4)
        if pct_extreme > 0:
            issues.append(f"channel {c} ({lead_names[c]}): {pct_extreme:.3f}% of samples exceed {PLAUSIBLE_MV_LIMIT}mV")

    # ---- Corrupted / flat segments ----
    win_len = int(FLAT_WINDOW_SEC * FS)
    for c in range(n_channels):
        col = sig[:, c]
        col = col[~np.isnan(col)]
        if len(col) < win_len:
            continue
        stds = windowed_std(col, win_len)
        pct_flat = float((stds < FLAT_STD_THRESHOLD).mean() * 100)
        row[f"ch{c}_pct_flat_windows"] = round(pct_flat, 3)
        if pct_flat > 0.5:
            issues.append(f"channel {c}: {pct_flat:.2f}% of 1s windows are flat (possible electrode disconnect)")

    # ---- Possible noisy regions ----
    for c in range(n_channels):
        col = sig[:, c]
        col = col[~np.isnan(col)]
        if len(col) < FS * 2:
            continue
        residual = high_freq_residual(col)
        win_len_n = int(NOISE_WINDOW_SEC * FS)
        noise_energy = windowed_std(residual, win_len_n)
        threshold = np.percentile(noise_energy, NOISE_PERCENTILE_FLAG)
        pct_noisy = float((noise_energy > max(threshold, 0.05)).mean() * 100)
        row[f"ch{c}_pct_noisy_windows"] = round(pct_noisy, 3)

    # ---- Channel-to-channel difference (for 2-channel records) ----
    if n_channels == 2:
        a, b = sig[:, 0], sig[:, 1]
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() > 10:
            corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
            row["channel_correlation"] = round(corr, 3)

    row["issues"] = "; ".join(issues) if issues else ""
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./data/quality")
    p.add_argument("--local_dir", default=None)
    p.add_argument("--records", nargs="*", default=None,
                    help="specific record IDs to check; omit to check all 48")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    records = args.records or ALL_RECORDS

    rows = []
    for rid in records:
        print(f"Checking record {rid} ...")
        try:
            row = analyze_record(rid, args.local_dir)
        except Exception as e:
            row = {"record_id": rid, "issues": f"FAILED TO LOAD: {e}"}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "data_quality_report.csv"), index=False)

    flagged = df[df["issues"].fillna("") != ""][["record_id", "issues"]]
    flagged.to_json(os.path.join(args.outdir, "flagged_issues.json"), orient="records", indent=2)

    # Lead usage summary -- answers "differences between ECG channels" across the dataset
    lead_summary = df["leads"].value_counts().to_dict()
    with open(os.path.join(args.outdir, "lead_usage_summary.json"), "w") as f:
        json.dump(lead_summary, f, indent=2)

    print("\nDone.")
    print(f" - {os.path.join(args.outdir, 'data_quality_report.csv')}: full per-record stats (import into Tableau)")
    print(f" - {os.path.join(args.outdir, 'flagged_issues.json')}: records with any flagged issue")
    print(f" - {os.path.join(args.outdir, 'lead_usage_summary.json')}: which lead combinations appear across the dataset")
    print(f"\n{len(flagged)} / {len(df)} records had at least one flagged issue.")


if __name__ == "__main__":
    main()