import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
    'AU25_r', 'AU26_r'
]
MAX_SEQ_LEN = 5000  # adjust based on GPU/CPU memory
BATCH_SIZE = 4      # smaller batch for CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD DATA
# -----------------------------
data = []
labels = []

file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))

for path in file_paths:
    df = pd.read_csv(path)
    df = df[df['success'] == 1]
    
    X = df[AU_COLUMNS].values.astype(np.float32)
    
    # Normalize each participant individually
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    data.append(X)
    labels.append(int(df['PHQ8_Binary'].iloc[0]))

def pad_or_truncate_sequences(sequences, max_len):
    padded = np.zeros((len(sequences), max_len, sequences[0].shape[1]), dtype=np.float32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length, :] = seq[:length]
    return padded

X_padded = pad_or_truncate_sequences(data, MAX_SEQ_LEN)
y = np.array(labels, dtype=np.int64)

print(f"Padded sequences shape: {X_padded.shape}")
print(f"Labels shape: {y.shape}")

# -----------------------------
# CUSTOM DATASET
# -----------------------------
class AUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = AUDataset(X_padded, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# TRANSFORMER MODEL (CPU-friendly)
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
            batch_first=True   # fix warning
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len, num_features)
        x = self.input_fc(x)              # -> (batch_size, seq_len, d_model)
        x = self.transformer(x)           # batch_first=True, no permute needed
        x = x.mean(dim=1)                 # average over seq_len -> (batch_size, d_model)
        out = self.classifier(x)          # -> (batch_size, num_classes)
        return out

model = AUTransformer(num_features=X_padded.shape[2]).to(DEVICE)

# -----------------------------
# WEIGHTED LOSS
# -----------------------------
num_class0 = np.sum(y == 0)
num_class1 = np.sum(y == 1)
class_weights = torch.tensor([1.0, num_class0/num_class1], dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# TRAINING LOOP (EXAMPLE)
# -----------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == batch_y).sum().item()
        total += batch_X.size(0)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/total:.4f} - Accuracy: {correct/total:.4f}")
