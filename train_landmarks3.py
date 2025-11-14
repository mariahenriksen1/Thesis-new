import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# -------------------
# CONFIG
# -------------------
data_path = "/Users/mariahenriksen/Library/Mobile Documents/com~apple~CloudDocs/daicwoz/cleaned_participants_features_final"
image_size = 16       # Smaller heatmap for speed
batch_size = 64
num_epochs = 5
frame_step = 100
window_size = 1       # sliding window frames per sample
device = "cpu"
model_save_path = "train_landmarks_v3.pth"
test_size = 0.2
use_smote = True      # balance data

print(f"Using device: {device}")

# -------------------
# UTILITIES
# -------------------
def normalize_landmarks(xs, ys):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    xs_norm = (xs - x_min) / (x_max - x_min + 1e-8)
    ys_norm = (ys - y_min) / (y_max - y_min + 1e-8)
    return xs_norm, ys_norm

def landmarks_to_heatmap(xs, ys, image_size=16, sigma=1.0):
    heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    xs = np.clip((xs * image_size).astype(int), 0, image_size - 1)
    ys = np.clip((ys * image_size).astype(int), 0, image_size - 1)
    xv, yv = np.meshgrid(np.arange(image_size), np.arange(image_size))
    for x, y in zip(xs, ys):
        heatmap += np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
    return np.clip(heatmap, 0, 1)

# -------------------
# DATASET
# -------------------
class LandmarkDataset(Dataset):
    def __init__(self, heatmaps, labels):
        self.heatmaps = heatmaps
        self.labels = labels

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, idx):
        return torch.tensor(self.heatmaps[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)

# -------------------
# LOAD & PREPROCESS DATA
# -------------------
def load_data(csv_folder, frame_step=100, window_size=1):
    all_data = []
    files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
    print(f"Loading data from {len(files)} participants...")
    
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        x_cols = [c for c in df.columns if c.startswith("x")]
        y_cols = [c for c in df.columns if c.startswith("y")]
        df = df.iloc[::frame_step, :]
        for j in range(len(df) - window_size + 1):
            xs_window = df.iloc[j:j+window_size][x_cols].values.flatten()
            ys_window = df.iloc[j:j+window_size][y_cols].values.flatten()
            label = df.iloc[j + window_size - 1]["PHQ8_Binary"]
            all_data.append((xs_window, ys_window, label))
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(files)} participants...")
    return all_data

raw_data = load_data(data_path, frame_step=frame_step, window_size=window_size)
print(f"Total frames after sampling: {len(raw_data)}")

# Precompute heatmaps
heatmaps, labels = [], []
for xs, ys, label in raw_data:
    xs_norm, ys_norm = normalize_landmarks(xs, ys)
    heatmaps.append(landmarks_to_heatmap(xs_norm, ys_norm, image_size))
    labels.append(label)
heatmaps = np.array(heatmaps)
labels = np.array(labels)

# -------------------
# BALANCE DATA WITH SMOTE
# -------------------
if use_smote:
    # Flatten heatmaps for SMOTE
    heatmaps_flat = heatmaps.reshape(len(heatmaps), -1)
    smote = SMOTE(random_state=42)
    heatmaps_res, labels_res = smote.fit_resample(heatmaps_flat, labels)
    heatmaps = heatmaps_res.reshape(-1, image_size, image_size)
    labels = labels_res
    print(f"After SMOTE, total samples: {len(labels)}")

# Train/test split
train_hm, test_hm, train_labels, test_labels = train_test_split(heatmaps, labels, test_size=test_size, random_state=42)
train_dataset = LandmarkDataset(train_hm, train_labels)
test_dataset = LandmarkDataset(test_hm, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# -------------------
# Tiny Vision Transformer
# -------------------
class TinyViT(nn.Module):
    def __init__(self, image_size=16, patch_size=4, num_classes=1, dim=64, depth=4, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.to_patch_embedding = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x).flatten(2).transpose(1, 2)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        for layer in self.layers:
            x = layer(x)
        return self.mlp_head(x[:, 0])

# -------------------
# TRAINING
# -------------------
model = TinyViT().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
best_acc = 0.0
best_loss = float('inf')

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {avg_test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ New best model saved! (Accuracy: {best_acc:.2f}%)")
        print("="*60)

    print(f"\n🎉 Training complete!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Best test loss: {best_loss:.4f}")
    print(f"Model saved as {model_save_path}")
