# %% Data loading and configuration
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Fixed seed so training (weighted sampling, noise, weight init) is reproducible
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dynamic determination of the project's true root
CWD = Path.cwd()
ROOT_DIR = CWD
while ROOT_DIR.parent != ROOT_DIR:
    if (ROOT_DIR / "src" / "data_augmented").exists():
        break
    ROOT_DIR = ROOT_DIR.parent

train_npz_path = ROOT_DIR / "src" / "data_augmented" / "train.npz"
val_npz_path = ROOT_DIR / "src" / "data_augmented" / "val.npz"

print(f"Current Working Directory (CWD) : {CWD}")
print(f"Detected root (ROOT_DIR): {ROOT_DIR}")
print(f"LOading train.npz  : {train_npz_path}")

# Custom PyTorch dataset supporting 1D signals + R-R features
class ECG_RR_Dataset(Dataset):
    def __init__(self, npz_path, label_map):
        data = np.load(npz_path)
        self.X_signal = torch.tensor(data["X_signal"], dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
        self.X_rr = torch.tensor(data["X_rr"], dtype=torch.float32)                      # (N, 3)
        
        # Mapping text labels to integer indices
        y_raw = data["y"]
        y_idx = np.array([label_map[lbl] for lbl in y_raw])
        self.y = torch.tensor(y_idx, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_signal[idx], self.X_rr[idx], self.y[idx]

# Data loading
label_map = {"N": 0, "S": 1, "V": 2, "F": 3}

train_dataset = ECG_RR_Dataset(train_npz_path, label_map)
val_dataset = ECG_RR_Dataset(val_npz_path, label_map)

# Softened balancing using WeightedRandomSampler (avoids over-correcting N)
y_train_np = train_dataset.y.numpy()
class_counts = np.bincount(y_train_np)
class_weights = 1.0 / np.sqrt(class_counts)  # Square root to smooth out weights
sample_weights = class_weights[y_train_np]

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# %% Hybrid model compatible with MAX78002
class ECG_RR_MAX78002_Classifier(nn.Module):
    """
    Hybrid Architecture (1D Signal + R-R Features)
    MAX78002 Restrictions:
      - Bias=True for Conv1d and Linear layers (No BatchNorm)
      - Only permitted layers: Conv1d, ReLU, MaxPool1d, Dropout, Linear
    """
    def __init__(self, num_classes=4):
        super(ECG_RR_MAX78002_Classifier, self).__init__()

        # Branch 1 : 1D CNN for the ECG signal
        self.cnn_branch = nn.Sequential(
            # Bloc 1: (N, 1, 200) -> (N, 16, 100)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # Bloc 2: (N, 16, 100) -> (N, 32, 50)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # Bloc 3: (N, 32, 50) -> (N, 64, 25)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        self.cnn_fc = nn.Linear(64 * 4, 32, bias=True)

        # Branch 2: Dense MLP for R-R features (pre_RR, post_RR, local_RR)
        self.rr_branch = nn.Sequential(
            nn.Linear(3, 16, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16, bias=True),
            nn.ReLU()
        )

        # Final fusion
        self.classifier = nn.Sequential(
            nn.Linear(32 + 16, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes, bias=True)
        )

    def forward(self, x_sig, x_rr):
        # Signal Processing
        feat_sig = self.cnn_branch(x_sig)
        feat_sig = self.adaptive_pool(feat_sig)
        feat_sig = torch.flatten(feat_sig, 1)
        feat_sig = F.relu(self.cnn_fc(feat_sig))

        # R-R Features Processing
        feat_rr = self.rr_branch(x_rr)

        # Merging
        combined = torch.cat((feat_sig, feat_rr), dim=1)
        out = self.classifier(combined)
        return out


# %% Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECG_RR_MAX78002_Classifier(num_classes=4).to(device)

# Weights for CrossEntropyLoss to complement resampling
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_sig, x_rr, targets in train_loader:
        x_sig, x_rr, targets = x_sig.to(device), x_rr.to(device), targets.to(device)

        # Simple data augmentation on the signal branch
        if model.training:
            noise = torch.randn_like(x_sig) * 0.01
            x_sig = x_sig + noise

        optimizer.zero_grad()
        outputs = model(x_sig, x_rr)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_sig.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    scheduler.step(epoch_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss Training: {epoch_loss:.4f} | LR: {current_lr:.6f}")


# %% Evaluation on Validation and Metrics
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for x_sig, x_rr, targets in val_loader:
        x_sig, x_rr = x_sig.to(device), x_rr.to(device)
        outputs = model(x_sig, x_rr)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

target_names = ["N", "S", "V", "F"]

print("\nConfusion Matrix:")
print(confusion_matrix(all_targets, all_preds))

print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=target_names))


# %% Final Evaluation on Test Set (DS2)
test_npz_path = ROOT_DIR / "src" / "data_augmented" / "test.npz"
print(f"Loading test.npz : {test_npz_path}")

# Load the test dataset (DS2)
test_dataset = ECG_RR_Dataset(test_npz_path, label_map)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluate the model on the test set
model.eval()
test_preds, test_targets = [], []

with torch.no_grad():
    for x_sig, x_rr, targets in test_loader:
        x_sig, x_rr = x_sig.to(device), x_rr.to(device)
        outputs = model(x_sig, x_rr)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(targets.numpy())

# Compute macro metrics
test_cm = confusion_matrix(test_targets, test_preds)
report_dict = classification_report(
    test_targets, test_preds, target_names=target_names, output_dict=True
)

test_acc = (np.array(test_preds) == np.array(test_targets)).mean()
macro_f1 = report_dict["macro avg"]["f1-score"]

print("\n" + "=" * 50)
print("FINAL TEST RESULTS (DS2)")
print("=" * 50)
print(f"Overall Accuracy : {test_acc:.4f}")
print(f"Macro F1-Score   : {macro_f1:.4f}")

print("\nTest Confusion Matrix:")
print(test_cm)

print("\nTest Classification Report:")
print(classification_report(test_targets, test_preds, target_names=target_names))


# %% Save the trained model and its predictions (used by the model comparison in EDA.ipynb)
save_dir = ROOT_DIR / "src" / "1D-CNN-augmented"
torch.save({"seed": SEED, "epochs": epochs, "state_dict": model.state_dict()},
           save_dir / "1d_cnn_classifier_augmented.pt")
np.savez(save_dir / "1d_cnn_predictions.npz",
         val_pred=np.array(all_preds), val_true=np.array(all_targets),
         test_pred=np.array(test_preds), test_true=np.array(test_targets),
         test_record_id=np.load(test_npz_path)["record_id"],
         classes=np.array(target_names))
print(f"Saved model and predictions to: {save_dir}")


# %%
