"""
noise_robustness.py — Section 5.4: Noise Robustness Evaluation
Evaluates the trained 1D-CNN + RR model (FP32 and optional INT8 PTQ) under
MIT-BIH Noise Stress Test Database (nstdb) artifacts at SNR = 24, 18, 12, 6, 0 dB.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import wfdb

SEED = 42
FS = 360
CLASSES = ["N", "S", "V", "F"]
LABELS = list(range(len(CLASSES)))
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
SNR_LEVELS_DB = [None, 24, 18, 12, 6, 0]  # None = Clean ECG
NOISE_TYPES = {
    "bw": "Baseline Wander (bw)",
    "ma": "Muscle Artifact (ma)",
    "em": "Electrode Motion (em)",
}

# Automatically locate project src directory
SRC_DIR = next(
    (p / sub for p in [Path.cwd(), *Path.cwd().parents] for sub in ["", "src"]
     if (p / sub / "data_augmented" / "test.npz").exists()),
    Path.cwd()
)
DATA_DIR = SRC_DIR / "data_augmented"
CNN_DIR = SRC_DIR / "1D-CNN-augmented"
OUT_DIR = CNN_DIR / "evaluation_5_4_to_5_6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ECG_RR_MAX78002_Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.1),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        self.cnn_fc = nn.Linear(64 * 4, 32, bias=True)
        self.rr_branch = nn.Sequential(
            nn.Linear(3, 16, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 16, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes, bias=True),
        )

    def forward(self, x_sig, x_rr):
        feat_sig = torch.flatten(self.adaptive_pool(self.cnn_branch(x_sig)), 1)
        feat_sig = F.relu(self.cnn_fc(feat_sig))
        feat_rr = self.rr_branch(x_rr)
        return self.classifier(torch.cat((feat_sig, feat_rr), dim=1))


def bandpass_filter(sig, fs=FS, low=0.5, high=40.0, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)


def load_nstdb_noise(noise_name, local_nstdb_dir=None):
    """Stream or load a noise recording ('bw', 'ma', 'em') from PhysioNet nstdb."""
    if local_nstdb_dir and (Path(local_nstdb_dir) / f"{noise_name}.hea").exists():
        rec = wfdb.rdrecord(str(Path(local_nstdb_dir) / noise_name))
    else:
        print(f"Streaming '{noise_name}' from PhysioNet (nstdb)...")
        rec = wfdb.rdrecord(noise_name, pn_dir="nstdb")
    return rec.p_signal[:, 0].astype(np.float64)


def add_Calibrated_noise(X_clean, noise_stream, snr_db, rng, apply_filter=False):
    """
    Superimposes random windows of real NSTDB noise onto held-out test beats
    at a target Signal-to-Noise Ratio (SNR in dB), then re-applies per-beat z-score.
    """
    if snr_db is None:
        return X_clean.copy()

    n_beats, win_len = X_clean.shape
    max_start = len(noise_stream) - win_len
    starts = rng.integers(0, max_start, size=n_beats)
    noise_windows = np.stack([noise_stream[s : s + win_len] for s in starts])

    # Zero-mean each noise segment before power calculation
    noise_windows = noise_windows - noise_windows.mean(axis=1, keepdims=True)

    if apply_filter:
        # Simulates adding noise before the 0.5-40 Hz Butterworth filter
        noise_windows = bandpass_filter(noise_windows)

    sig_power = np.mean(X_clean ** 2, axis=1, keepdims=True)
    noise_power = np.mean(noise_windows ** 2, axis=1, keepdims=True) + 1e-12

    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)

    X_noisy = X_clean + scale * noise_windows

    # Re-normalize per beat (z-score) just like prepare_data_augmented.py
    mean = X_noisy.mean(axis=1, keepdims=True)
    std = X_noisy.std(axis=1, keepdims=True)
    X_noisy = np.where(std > 1e-8, (X_noisy - mean) / std, X_noisy - mean)
    return X_noisy.astype(np.float32)


@torch.no_grad()
def predict_batches(model, X_sig_np, X_rr_np, device, batch_size=2048):
    x_sig = torch.tensor(X_sig_np, dtype=torch.float32, device=device).unsqueeze(1)
    x_rr = torch.tensor(X_rr_np, dtype=torch.float32, device=device)
    preds = [
        model(x_sig[i : i + batch_size], x_rr[i : i + batch_size]).argmax(dim=1).cpu().numpy()
        for i in range(0, len(x_sig), batch_size)
    ]
    return np.concatenate(preds)


def compute_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": float(np.mean(r)),
    }
    for c, f_val, r_val in zip(CLASSES, f1, r):
        res[f"F1_{c}"] = float(f_val)
        res[f"Recall_{c}"] = float(r_val)
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load held-out DS2 test data
    test_data = np.load(DATA_DIR / "test.npz")
    X_test_sig = test_data["X_signal"].astype(np.float32)
    X_test_rr = test_data["X_rr"].astype(np.float32)
    y_test = np.array([LABEL_MAP[str(v)] for v in test_data["y"]])

    # 2. Load trained FP32 model
    model_path = CNN_DIR / "1d_cnn_classifier_augmented.pt"
    model = ECG_RR_MAX78002_Classifier().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval()

    # 3. Stream NSTDB noise signals and also build a combined (bw + ma + em) stream
    noise_streams = {k: load_nstdb_noise(k) for k in NOISE_TYPES}
    min_len = min(len(v) for v in noise_streams.values())
    noise_streams["combined"] = (
        noise_streams["bw"][:min_len]
        + noise_streams["ma"][:min_len]
        + noise_streams["em"][:min_len]
    ) / 3.0
    all_noise_labels = {**NOISE_TYPES, "combined": "Combined (bw+ma+em)"}

    rows = []
    for apply_filter in [False, True]:
        mode_name = "Bandpass-Filtered (0.5-40Hz)" if apply_filter else "Direct Unfiltered Injection"
        print(f"\n=== Evaluating Mode: {mode_name} ===")
        for n_key, n_label in all_noise_labels.items():
            rng = np.random.default_rng(SEED)
            for snr in SNR_LEVELS_DB:
                X_Perturbed = add_Calibrated_noise(
                    X_test_sig, noise_streams[n_key], snr, rng, apply_filter=apply_filter
                )
                preds = predict_batches(model, X_Perturbed, X_test_rr, device)
                m = compute_metrics(y_test, preds)
                snr_str = "Clean" if snr is None else f"{snr} dB"
                rows.append({
                    "mode": mode_name,
                    "noise_type": n_key,
                    "noise_label": n_label,
                    "snr_db": np.inf if snr is None else snr,
                    "condition": snr_str,
                    **m,
                })
                print(
                    f"  [{n_key:8s} | {snr_str:6s}] "
                    f"Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f} | "
                    f"F1 (N/S/V/F): {m['F1_N']:.3f} / {m['F1_S']:.3f} / {m['F1_V']:.3f} / {m['F1_F']:.3f}"
                )

    df_res = pd.DataFrame(rows)
    csv_path = OUT_DIR / "noise_robustness_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved full noise robustness table to: {csv_path}")

    # 4. Plot Macro-F1 degradation curves across SNR levels
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (mode_name, grp) in zip(axes, df_res.groupby("mode", sort=False)):
        finite_grp = grp[grp["snr_db"] != np.inf]
        clean_f1 = grp[grp["snr_db"] == np.inf]["macro_f1"].iloc[0]
        ax.axhline(clean_f1, color="black", linestyle="--", alpha=0.7, label=f"Clean baseline ({clean_f1:.3f})")
        ax.axhline(0.25, color="red", linestyle=":", alpha=0.7, label="Random-guess floor (0.250)")

        for n_key, sub in finite_grp.groupby("noise_type", sort=False):
            sub_sorted = sub.sort_values("snr_db", ascending=False)
            ax.plot(
                sub_sorted["snr_db"].astype(str) + " dB",
                sub_sorted["macro_f1"],
                marker="o",
                linewidth=2,
                label=all_noise_labels[n_key],
            )
        ax.set_title(f"Noise Robustness — {mode_name}")
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
        ax.set_ylabel("Test Macro F1-Score (DS2)")
        ax.set_ylim(0.15, 0.58)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = OUT_DIR / "noise_robustness_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
