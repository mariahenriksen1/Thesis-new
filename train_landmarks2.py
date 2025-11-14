import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# -------------------
# CONFIG
# -------------------
data_path = "/Users/mariahenriksen/Library/Mobile Documents/com~apple~CloudDocs/daicwoz/cleaned_participants_features_final"
image_size = 32
patch_size = 4   # 4x4 patches -> 8x8 patches for 32x32 image
batch_size = 64
num_epochs = 5
frame_step = 100
device = "cpu"
model_save_path = "train_landmarks_v2.pth"
test_size = 0.2

print(f"Using device: {device}")

# -------------------
# UTILITIES
# -------------------
def normalize_landmarks(xs, ys):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    xs_norm = (xs - xs.min()) / (xs.max() - xs.min() + 1e-8)
    ys_norm = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8)
    return xs_norm, ys_norm

def landmarks_to_heatmap(xs, ys, image_size=32, sigma=1.0):
    heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    xs = np.clip((xs * image_size).astype(int), 0, image_size - 1)
    ys = np.clip((ys * image_size).astype(int), 0, image_size - 1)
    for x, y in zip(xs, ys):
        xv, yv = np.meshgrid(np.arange(image_size), np.arange(image_size))
        heatmap += np.exp(-((xv - x)**2 + (yv - y)**2) / (2 * sigma**2))
    return np.clip(heatmap, 0, 1)

# -------------------
# DATASET
# -------------------
class LandmarkDataset(Dataset):
    def __init__(self, data_list, image_size=32):
        self.data = data_list
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xs, ys, label = self.data[idx]
        xs_norm, ys_norm = normalize_landmarks(xs, ys)
        heatmap = landmarks_to_heatmap(xs_norm, ys_norm, self.image_size)
        return torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

# -------------------
# LOAD DATA
# -------------------
def load_data(csv_folder, frame_step=100):
    all_data = []
    files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
    print(f"Loading data from {len(files)} participants...")
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        x_cols = [c for c in df.columns if c.startswith("x")]
        y_cols = [c for c in df.columns if c.startswith("y")]
        df = df.iloc[::frame_step, :]
        for _, row in df.iterrows():
            xs = row[x_cols].values
            ys = row[y_cols].values
            label = row["PHQ8_Binary"]
            all_data.append((xs, ys, label))
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(files)} participants...")
    return all_data

all_data = load_data(data_path, frame_step=frame_step)
print(f"Total frames after sampling: {len(all_data)}")

train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=42)
train_dataset = LandmarkDataset(train_data, image_size=image_size)
test_dataset = LandmarkDataset(test_data, image_size=image_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# -------------------
# VISION TRANSFORMER MODEL
# -------------------
class TinyViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=128, depth=4, heads=4, mlp_dim=256, num_classes=1):
        super(TinyViT, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (image_size // patch_size)**2

        # Patch embedding
        self.to_patch_embedding = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b = x.size(0)
        x = self.to_patch_embedding(x)           # [B, dim, H', W']
        x = x.flatten(2).transpose(1, 2)        # [B, num_patches, dim]

        cls_tokens = self.cls_token.expand(b, -1, -1)  # [B,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)          # [B,1+num_patches,dim]
        pos_embed = torch.cat((torch.zeros(1,1,self.dim,device=x.device), self.pos_embedding), dim=1)
        x = x + pos_embed                              # broadcast pos embeddings

        x = self.transformer(x)                        # Transformer
        x = x[:, 0]                                    # CLS token
        x = self.mlp_head(x)
        return x

model = TinyViT(image_size=image_size, patch_size=patch_size).to(device)
print(f"Tiny Vision Transformer created with {sum(p.numel() for p in model.parameters()):,} parameters")

# -------------------
# TRAINING
# -------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_acc = 0.0
best_loss = float('inf')

print("Starting training...\n" + "="*60)

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
        train_correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
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

print("\n🎉 Training complete!")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"Best test loss: {best_loss:.4f}")
print(f"Model saved as {model_save_path}")
