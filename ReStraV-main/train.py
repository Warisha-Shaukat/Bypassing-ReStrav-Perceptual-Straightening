import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix
from pathlib import Path

# Reuse the same architecture
class MLP(nn.Module):
    def __init__(self, in_dim=21, h1=64, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1)
        )
    def forward(self, x): return self.net(x)

def run_testing():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    h5_path = Path("features/repaired3_features.h5")
    with h5py.File(h5_path, "r") as f:
        X = f["features"][:].astype(np.float32)
        y = f["label"][:].astype(np.int64)
        paths = f["path"][:].astype(str)

    # 2. Load Saved Parameters & Model
    mean = np.load("mean.npy")
    std = np.load("std.npy")
    best_tau = np.load("best_tau.npy")
    
    model = MLP().to(device)
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()

    # 3. Preprocess Test Data (Use saved mean/std)
    X_norm = (X - mean) / std
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y)), 
                              batch_size=32, shuffle=False)

    # 4. Inference
    test_logits, test_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel()
            test_logits.append(out)
            test_labels.append(yb.numpy())

    test_logits = np.concatenate(test_logits)
    test_labels = np.concatenate(test_labels)
    test_preds = (test_logits >= best_tau).astype(int)

    # 5. Save Results & Print Metrics
    df = pd.DataFrame({"path": paths, "true_label": test_labels, "pred_label": test_preds, "prob_1": test_logits})
    df.to_csv("test_predictions_all.csv", index=False)

    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary")
    auc = roc_auc_score(test_labels, test_logits)
    acc = accuracy_score(test_labels, test_preds)
    
    print("\n=== Test Performance ===")
    print(f"Accuracy : {acc:.3f} | AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(test_labels, test_preds))

if __name__ == "__main__":
    run_testing()
