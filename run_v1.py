# landmark_model_v1.py
# BASELINE VERSION with COMPREHENSIVE EVALUATION
# Integrates full metrics reporting

import os
import random
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# -------------------
# CONFIG - V1 BASELINE
# -------------------
data_csv_folder = "/Users/mariahenriksen/Library/Mobile Documents/com~apple~CloudDocs/daicwoz/cleaned_participants_features_final"

sequence_length = 30
sequence_stride = 30
batch_size = 32
num_epochs = 5
test_size = 0.20
seed = 42

model_save_path = "landmark_v1.pth"
results_dir = "./landmark_v1_results"
os.makedirs(results_dir, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*70)
print("MODEL VERSION: V1 - BASELINE")
print("  Sequence Length: 30 frames")
print("  Epochs: 5")
print("  Embed Dim: 64")
print("  Layers: 2")
print("="*70)

random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# -------------------
# DATASET
# -------------------
class SequentialLandmarkDataset(Dataset):
    def __init__(self, csv_paths, sequence_length=60, sequence_stride=30, 
                 min_confidence=0.8, max_sequences_per_participant=50):
        self.sequences = []
        
        for csv_path in tqdm(csv_paths, desc="Loading sequences"):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            
            if not all(f"x{i}" in df.columns for i in range(68)):
                continue
            if not all(f"y{i}" in df.columns for i in range(68)):
                continue
            
            label_key = None
            for k in ("PHQ8_Binary", "PHQ8", "phq8", "phq8_binary"):
                if k in df.columns:
                    label_key = k
                    break
            if label_key is None:
                continue
            
            label = int(df.iloc[0][label_key])
            
            gender = None
            if "Gender" in df.columns:
                gender_val = df.iloc[0]["Gender"]
                if isinstance(gender_val, str):
                    gender = 1 if gender_val.lower() in ['male', 'm', '1'] else 0
                else:
                    gender = int(gender_val)
            
            participant_id = os.path.basename(csv_path).replace('.csv', '')
            
            if "confidence" in df.columns:
                df = df[df["confidence"] >= min_confidence].reset_index(drop=True)
            
            total_frames = len(df)
            
            xs_all = df[[f"x{i}" for i in range(68)]].values.astype(np.float32)
            ys_all = df[[f"y{i}" for i in range(68)]].values.astype(np.float32)
            all_landmarks = np.stack([xs_all, ys_all], axis=2)
            
            sequence_starts = list(range(0, total_frames - sequence_length + 1, sequence_stride))
            
            if len(sequence_starts) > max_sequences_per_participant:
                sequence_starts = np.random.choice(sequence_starts, size=max_sequences_per_participant, replace=False).tolist()
            
            for start_idx in sequence_starts:
                end_idx = start_idx + sequence_length
                sequence_array = all_landmarks[start_idx:end_idx].copy()
                sequence_array = sequence_array - sequence_array.mean(axis=(0, 1), keepdims=True)
                std = sequence_array.std()
                if std > 1e-6:
                    sequence_array = sequence_array / std
                
                self.sequences.append({
                    'landmarks': sequence_array,
                    'label': label,
                    'gender': gender,
                    'participant_id': participant_id
                })
        
        print(f"Created {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sample = self.sequences[idx]
        landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        gender = -1 if sample['gender'] is None else sample['gender']
        gender = torch.tensor(gender, dtype=torch.long)
        return landmarks, gender, label

# -------------------
# MODEL
# -------------------
class SequentialTransformer(nn.Module):
    def __init__(self, landmark_dim=136, embed_dim=64, num_heads=4, 
                 num_layers=2, dropout=0.1, max_seq_length=100):
        super().__init__()
        
        self.landmark_projection = nn.Linear(landmark_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))
        self.gender_embedding = nn.Embedding(3, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, landmarks, gender):
        batch_size, seq_len = landmarks.shape[0], landmarks.shape[1]
        landmarks_flat = landmarks.reshape(batch_size, seq_len, -1)
        
        x = self.landmark_projection(landmarks_flat)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        cls_output = self.norm(x[:, 0])
        
        gender_input = torch.where(gender >= 0, gender, torch.tensor(2, device=gender.device))
        gender_emb = self.gender_embedding(gender_input)
        
        combined = torch.cat([cls_output, gender_emb], dim=1)
        logits = self.classifier(combined)
        
        return logits

# -------------------
# LOAD DATA
# -------------------
csv_paths = glob(os.path.join(data_csv_folder, "*.csv"))
print(f"\nFound {len(csv_paths)} participant CSV files")

ds = SequentialLandmarkDataset(csv_paths, sequence_length=sequence_length, sequence_stride=sequence_stride, 
                                min_confidence=0.8, max_sequences_per_participant=50)

participant_labels = {}
for seq in ds.sequences:
    if seq['participant_id'] not in participant_labels:
        participant_labels[seq['participant_id']] = seq['label']

train_participants, val_participants = train_test_split(
    list(participant_labels.keys()), test_size=test_size, random_state=seed,
    stratify=list(participant_labels.values())
)

train_idx = [i for i, seq in enumerate(ds.sequences) if seq['participant_id'] in train_participants]
val_idx = [i for i, seq in enumerate(ds.sequences) if seq['participant_id'] in val_participants]

train_ds = Subset(ds, train_idx)
val_ds = Subset(ds, val_idx)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train: {len(train_participants)} participants, {len(train_idx)} sequences")
print(f"Val: {len(val_participants)} participants, {len(val_idx)} sequences")

# -------------------
# MODEL SETUP
# -------------------
model = SequentialTransformer(landmark_dim=136, embed_dim=64, num_heads=4, num_layers=2, 
                              dropout=0.1, max_seq_length=sequence_length + 1).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

train_labels = [ds.sequences[i]['label'] for i in train_idx]
pos_weight = torch.tensor([train_labels.count(0) / max(train_labels.count(1), 1)], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# -------------------
# TRAINING
# -------------------
print(f"\n{'='*70}")
print("TRAINING V1 MODEL")
print(f"{'='*70}\n")

best_val_auc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for landmarks, gender, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
        landmarks, gender, labels = landmarks.to(device), gender.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(landmarks, gender)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * landmarks.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_total += landmarks.size(0)
    
    train_loss /= train_total
    train_acc = 100.0 * train_correct / train_total
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for landmarks, gender, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
            landmarks, gender, labels = landmarks.to(device), gender.to(device), labels.to(device).unsqueeze(1)
            
            logits = model(landmarks, gender)
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * landmarks.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += landmarks.size(0)
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
    
    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total
    val_auc = roc_auc_score(all_labels, all_probs)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_auc'].append(val_auc)
    
    scheduler.step()
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc:.4f}")
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_probs = all_probs
        best_labels = all_labels
        best_preds = all_preds
        torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'val_auc': val_auc}, model_save_path)
        print(f"  ✓ Best model saved!")

# -------------------
# COMPREHENSIVE EVALUATION
# -------------------
print(f"\n{'='*70}")
print("COMPREHENSIVE EVALUATION - V1 RESULTS")
print(f"{'='*70}\n")

# Calculate all metrics
cm = confusion_matrix(best_labels, best_preds)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(best_labels, best_preds)
precision_dep = precision_score(best_labels, best_preds, pos_label=1, zero_division=0)
recall_dep = recall_score(best_labels, best_preds, pos_label=1, zero_division=0)
f1_dep = f1_score(best_labels, best_preds, pos_label=1, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
auc = best_val_auc

# Print results
print(f"📊 CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"             Not Dep  Depressed")
print(f"Actual Not   {tn:4d}     {fp:4d}")
print(f"Actual Dep   {fn:4d}     {tp:4d}")

print(f"\n🎯 KEY METRICS:")
print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  AUC-ROC:            {auc:.4f}")
print(f"  Precision (Dep):    {precision_dep:.4f}")
print(f"  Recall (Dep):       {recall_dep:.4f}")
print(f"  F1-Score (Dep):     {f1_dep:.4f}")
print(f"  Specificity:        {specificity:.4f}")

# Save metrics
metrics_v1 = {
    'model': 'Landmark-V1',
    'accuracy': float(accuracy),
    'precision': float(precision_dep),
    'recall': float(recall_dep),
    'f1_score': float(f1_dep),
    'auc': float(auc),
    'specificity': float(specificity),
    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
}

with open(f"{results_dir}/v1_metrics.json", 'w') as f:
    json.dump(metrics_v1, f, indent=2)

# Save for thesis table
pd.DataFrame([metrics_v1]).to_csv(f"{results_dir}/v1_metrics.csv", index=False)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Not Dep', 'Dep'], yticklabels=['Not Dep', 'Dep'], ax=axes[0,0])
axes[0,0].set_title('V1: Confusion Matrix')
axes[0,0].set_ylabel('True Label')
axes[0,0].set_xlabel('Predicted Label')

# Training curves
axes[0,1].plot(history['train_acc'], label='Train', marker='o')
axes[0,1].plot(history['val_acc'], label='Val', marker='s')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy (%)')
axes[0,1].set_title('V1: Accuracy')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# AUC curve
axes[1,0].plot(history['val_auc'], marker='s', color='green')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('AUC')
axes[1,0].set_title('V1: Validation AUC')
axes[1,0].grid(alpha=0.3)

# Metrics summary
metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'AUC']
values_plot = [accuracy, precision_dep, recall_dep, f1_dep, specificity, auc]
bars = axes[1,1].bar(metrics_plot, values_plot, color=['#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71', '#9b59b6'])
for bar in bars:
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
axes[1,1].set_ylabel('Score')
axes[1,1].set_title('V1: Metrics Summary')
axes[1,1].set_ylim([0, 1.0])
plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f"{results_dir}/v1_comprehensive_results.png", dpi=300)
plt.close()

print(f"\n✓ Results saved to {results_dir}/")
print(f"\n{'='*70}")
print("V1 TRAINING COMPLETE!")
print(f"Best Validation AUC: {best_val_auc:.4f}")
print(f"{'='*70}\n")