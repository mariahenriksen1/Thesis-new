import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
    'AU25_r', 'AU26_r'
]
MAX_SEQ_LEN = 500       # shorter sequence length
WINDOW_STEP = 250       # overlap of 50%
BATCH_SIZE = 8          # can increase since sequences are shorter
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD AND PREPROCESS DATA
# -----------------------------
data, labels = [], []
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))

for path in file_paths:
    df = pd.read_csv(path)
    df = df[df["success"] == 1]
    
    X = df[AU_COLUMNS].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create sliding windows
    start = 0
    while start + MAX_SEQ_LEN <= X.shape[0]:
        window = X[start:start + MAX_SEQ_LEN]
        data.append(window)
        labels.append(int(df["PHQ8_Binary"].iloc[0]))
        start += WINDOW_STEP

X = np.array(data)
y = np.array(labels)

print(f"Data shape after sliding windows: {X.shape}, Labels shape: {y.shape}")

# -----------------------------
# SPLIT DATA (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# OVERSAMPLE MINORITY CLASS IN TRAIN SET
# -----------------------------
X_train_min = X_train[y_train == 1]
y_train_min = y_train[y_train == 1]

X_train_maj = X_train[y_train == 0]
y_train_maj = y_train[y_train == 0]

X_train_min_upsampled, y_train_min_upsampled = resample(
    X_train_min, y_train_min,
    replace=True,
    n_samples=len(y_train_maj),
    random_state=42
)

X_train_bal = np.concatenate([X_train_maj, X_train_min_upsampled])
y_train_bal = np.concatenate([y_train_maj, y_train_min_upsampled])

# Shuffle
perm = np.random.permutation(len(y_train_bal))
X_train_bal, y_train_bal = X_train_bal[perm], y_train_bal[perm]

print(f"Balanced training set shape: {X_train_bal.shape}, Labels: {y_train_bal.shape}")

# -----------------------------
# DATASET CLASS
# -----------------------------
class AUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(AUDataset(X_train_bal, y_train_bal), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(AUDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# TRANSFORMER MODEL
# -----------------------------
class AUTransformer(nn.Module):
    def __init__(self, num_features, num_classes=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

model = AUTransformer(num_features=X.shape[2]).to(DEVICE)

# -----------------------------
# WEIGHTED LOSS
# -----------------------------
num_class0 = np.sum(y_train_bal == 0)
num_class1 = np.sum(y_train_bal == 1)
class_weights = torch.tensor([1.0, num_class0 / num_class1], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/total:.4f} - Acc: {correct/total:.4f}")

# -----------------------------
# EVALUATION
# -----------------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, digits=4))
