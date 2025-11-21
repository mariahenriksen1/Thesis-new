import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed. Install with 'pip install tqdm' for progress bars.")

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
    'AU25_r', 'AU26_r'
]
MAX_SEQ_LEN = 500       # sliding window length
WINDOW_STEP = 250       # 50% overlap
MIN_CONFIDENCE = 0.8    # confidence threshold
BATCH_SIZE = 8
EPOCHS = 5
TEST_SIZE = 0.2
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print("="*70)
print("AU MODEL V5 - CLASS WEIGHTS ONLY (NO SMOTE)")
print("  + Participant-level splits (no data leakage)")
print("  + Confidence filtering (>= 0.8)")
print("  + Gender embedding")
print("  + Class weights [1.0, 133/56] - NO synthetic sampling")
print("="*70)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -----------------------------
# LOAD DATA WITH PARTICIPANT TRACKING
# -----------------------------
participant_data = {}  # {participant_id: {'windows': [], 'label': int, 'gender': int}}
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))

print(f"\nLoading {len(file_paths)} participant files...")

for path in file_paths:
    # Extract participant ID from filename
    participant_id = os.path.basename(path).replace('_CLNF_AUs_final.csv', '')
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {participant_id}: {e}")
        continue
    
    # Filter by confidence if column exists
    if 'confidence' in df.columns:
        df = df[df['confidence'] >= MIN_CONFIDENCE].reset_index(drop=True)
    
    # Filter by success
    if 'success' in df.columns:
        df = df[df['success'] == 1].reset_index(drop=True)
    
    if len(df) < MAX_SEQ_LEN:
        print(f"Skipping {participant_id}: insufficient frames ({len(df)})")
        continue
    
    # Check for required columns
    if not all(col in df.columns for col in AU_COLUMNS):
        print(f"Skipping {participant_id}: missing AU columns")
        continue
    
    # Get label
    label_key = None
    for k in ['PHQ8_Binary', 'PHQ8', 'phq8', 'phq8_binary']:
        if k in df.columns:
            label_key = k
            break
    
    if label_key is None:
        print(f"Skipping {participant_id}: no label column found")
        continue
    
    label = int(df[label_key].iloc[0])
    
    # Get gender
    gender = None
    if 'Gender' in df.columns:
        gender_val = df.iloc[0]['Gender']
        if isinstance(gender_val, str):
            gender = 1 if gender_val.lower() in ['male', 'm', '1'] else 0
        else:
            gender = int(gender_val)
    
    # Extract and normalize AU data
    X = df[AU_COLUMNS].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create sliding windows
    windows = []
    start = 0
    while start + MAX_SEQ_LEN <= X.shape[0]:
        window = X[start:start + MAX_SEQ_LEN]
        windows.append(window)
        start += WINDOW_STEP
    
    if len(windows) > 0:
        participant_data[participant_id] = {
            'windows': windows,
            'label': label,
            'gender': gender if gender is not None else -1  # -1 for unknown
        }

print(f"\nSuccessfully loaded {len(participant_data)} participants")

# -----------------------------
# PARTICIPANT-LEVEL SPLIT
# -----------------------------
participant_ids = list(participant_data.keys())
participant_labels = [participant_data[pid]['label'] for pid in participant_ids]

train_pids, test_pids = train_test_split(
    participant_ids,
    test_size=TEST_SIZE,
    stratify=participant_labels,
    random_state=RANDOM_SEED
)

print(f"\nParticipant split:")
print(f"  Train: {len(train_pids)} participants")
print(f"  Test:  {len(test_pids)} participants")

# Flatten windows for train/test sets
train_windows = []
train_labels = []
train_genders = []

for pid in train_pids:
    for window in participant_data[pid]['windows']:
        train_windows.append(window)
        train_labels.append(participant_data[pid]['label'])
        train_genders.append(participant_data[pid]['gender'])

test_windows = []
test_labels = []
test_genders = []

for pid in test_pids:
    for window in participant_data[pid]['windows']:
        test_windows.append(window)
        test_labels.append(participant_data[pid]['label'])
        test_genders.append(participant_data[pid]['gender'])

X_train = np.array(train_windows)
y_train = np.array(train_labels)
gender_train = np.array(train_genders)

X_test = np.array(test_windows)
y_test = np.array(test_labels)
gender_test = np.array(test_genders)

print(f"\nWindow counts:")
print(f"  Train: {len(train_windows)} windows")
print(f"  Test:  {len(test_windows)} windows")
print(f"\nClass distribution in training:")
print(f"  Class 0 (Not Depressed): {np.sum(y_train == 0)}")
print(f"  Class 1 (Depressed):     {np.sum(y_train == 1)}")

# -----------------------------
# NO SMOTE - USE ORIGINAL DATA
# -----------------------------
print("\n** NO SMOTE APPLIED - Using original imbalanced data **")
X_train_final = X_train
y_train_final = y_train
gender_train_final = gender_train

# Shuffle
perm = np.random.permutation(len(y_train_final))
X_train_final = X_train_final[perm]
y_train_final = y_train_final[perm]
gender_train_final = gender_train_final[perm]

print(f"Training set shape: {X_train_final.shape}")
print(f"  Class 0: {np.sum(y_train_final == 0)}")
print(f"  Class 1: {np.sum(y_train_final == 1)}")

# -----------------------------
# DATASET CLASS
# -----------------------------
class AUDataset(Dataset):
    def __init__(self, X, y, gender):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.gender = torch.tensor(gender, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.gender[idx], self.y[idx]

train_loader = DataLoader(
    AUDataset(X_train_final, y_train_final, gender_train_final),
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    AUDataset(X_test, y_test, gender_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# TRANSFORMER MODEL WITH GENDER
# -----------------------------
class AUTransformerWithGender(nn.Module):
    def __init__(self, num_features, num_classes=2, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        # AU feature projection
        self.input_fc = nn.Linear(num_features, d_model)
        
        # Gender embedding (0: female, 1: male, 2: unknown)
        self.gender_embedding = nn.Embedding(3, d_model)
        
        # Positional encoding (optional, for better temporal modeling)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Classifier (combines AU representation + gender)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x, gender):
        batch_size = x.size(0)
        
        # Project AU features
        x = self.input_fc(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        au_representation = x.mean(dim=1)
        au_representation = self.norm(au_representation)
        
        # Gender embedding (map -1 to 2 for unknown)
        gender_input = torch.where(gender >= 0, gender, torch.tensor(2, device=gender.device))
        gender_emb = self.gender_embedding(gender_input)
        
        # Combine AU and gender representations
        combined = torch.cat([au_representation, gender_emb], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits

model = AUTransformerWithGender(num_features=X_train.shape[2]).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# -----------------------------
# LOSS & OPTIMIZER WITH CLASS WEIGHTS
# -----------------------------
num_class0 = np.sum(y_train_final == 0)
num_class1 = np.sum(y_train_final == 1)
weight_ratio = num_class0 / num_class1
class_weights = torch.tensor([1.0, weight_ratio], dtype=torch.float32).to(DEVICE)

print(f"\nClass weights: [1.0, {weight_ratio:.4f}]")
print(f"  (Class 0: {num_class0}, Class 1: {num_class1})")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# TRAINING LOOP
# -----------------------------
print(f"\n{'='*70}")
print("TRAINING")
print(f"{'='*70}\n")

best_val_auc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    # Add progress tracking
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]") if TQDM_AVAILABLE else train_loader
    num_batches = len(train_loader)
    
    for batch_idx, (X_batch, gender_batch, y_batch) in enumerate(train_iter):
        X_batch = X_batch.to(DEVICE)
        gender_batch = gender_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch, gender_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
        # Update progress
        if TQDM_AVAILABLE:
            train_iter.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0*correct/total:.2f}%'})
        elif (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}, Acc: {100.0*correct/total:.2f}%")
    
    train_loss = total_loss / total
    train_acc = 100.0 * correct / total
    
    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_probs, all_labels, all_preds = [], [], []
    
    with torch.no_grad():
        val_iter = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]") if TQDM_AVAILABLE else test_loader
        for X_batch, gender_batch, y_batch in val_iter:
            X_batch = X_batch.to(DEVICE)
            gender_batch = gender_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            outputs = model(X_batch, gender_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item() * X_batch.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class 1
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    val_loss = val_loss / val_total
    val_acc = 100.0 * val_correct / val_total
    val_auc = roc_auc_score(all_labels, all_probs)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_auc'].append(val_auc)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc:.4f}")
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_probs = all_probs
        best_labels = all_labels
        best_preds = all_preds
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'val_auc': val_auc,
            'history': history
        }, 'au_v5_no_smote.pth')
        print(f"  ✓ Best model saved!")

# -----------------------------
# COMPREHENSIVE EVALUATION
# -----------------------------
print(f"\n{'='*70}")
print("EVALUATION RESULTS")
print(f"{'='*70}\n")

cm = confusion_matrix(best_labels, best_preds)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(best_labels, best_preds)
precision = precision_score(best_labels, best_preds, zero_division=0)
recall = recall_score(best_labels, best_preds, zero_division=0)
f1 = f1_score(best_labels, best_preds, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
auc = best_val_auc

print(f"📊 CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"             Not Dep  Depressed")
print(f"Actual Not   {tn:4d}     {fp:4d}")
print(f"Actual Dep   {fn:4d}     {tp:4d}")

print(f"\n🎯 METRICS:")
print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  AUC-ROC:     {auc:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")
print(f"  F1-Score:    {f1:.4f}")
print(f"  Specificity: {specificity:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(best_labels, best_preds, digits=4))

# Save metrics
results_dir = "./au_v5_no_smote_results"
os.makedirs(results_dir, exist_ok=True)

metrics = {
    'model': 'AU-V5-No-SMOTE',
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc': float(auc),
    'specificity': float(specificity),
    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    'train_participants': len(train_pids),
    'test_participants': len(test_pids),
    'train_windows': len(train_windows),
    'test_windows': len(test_windows),
    'class_weight_ratio': float(weight_ratio)
}

with open(f"{results_dir}/metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Results saved to {results_dir}/")
print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"Best Validation AUC: {best_val_auc:.4f}")
print(f"{'='*70}\n")