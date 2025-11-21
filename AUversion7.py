# v7_transformer.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
from scipy import stats

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
EPOCHS = 8
TEST_SIZE = 0.2
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# v7 options
TRAIN_GENDER_MODELS = False   # If True, train separate models for male/female
USE_FOCAL_LOSS = False        # If True, use focal loss instead of CE
BALANCE_WITH_SAMPLER = True   # If True, use WeightedRandomSampler for class balance
ADD_DELTA_FEATURES = True     # Add first-difference features (temporal derivative)
ADD_BINARY_FEATURES = True    # Add binary activation (AU > threshold) as extra channels
BINARY_THRESHOLD = 0.5        # Threshold for binary activation (adjustable)

print(f"Device: {DEVICE}")
print("="*70)
print("AU MODEL V7 - Transformer with improved balancing & feature aug")
print("="*70)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -----------------------------
# LOAD DATA WITH PARTICIPANT TRACKING + FEATURE AUGMENTATION
# -----------------------------
participant_data = {}  # {participant_id: {'windows': [], 'label': int, 'gender': int}}
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))

print(f"\nLoading {len(file_paths)} participant files...")

# We'll collect scalers across participants and fit later on train windows only
all_participant_raw = {}

for path in file_paths:
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
        # skip if insufficient frames
        continue

    # required columns
    if not all(col in df.columns for col in AU_COLUMNS):
        continue

    # label
    label_key = next((k for k in ['PHQ8_Binary', 'PHQ8', 'phq8', 'phq8_binary'] if k in df.columns), None)
    if label_key is None:
        continue
    label = int(df[label_key].iloc[0])

    # gender
    gender = -1
    if 'Gender' in df.columns:
        gender_val = df.iloc[0]['Gender']
        if isinstance(gender_val, str):
            gender = 1 if gender_val.lower() in ['male', 'm', '1'] else 0
        else:
            try:
                gender = int(gender_val)
            except:
                gender = -1

    # raw AU matrix (frames x features)
    X_raw = df[AU_COLUMNS].values.astype(np.float32)
    all_participant_raw[participant_id] = {'X_raw': X_raw, 'label': label, 'gender': gender}

# We'll create windows after we determine train/test split so scaling is fit on train only
participant_ids = list(all_participant_raw.keys())
participant_labels = [all_participant_raw[pid]['label'] for pid in participant_ids]

train_pids, test_pids = train_test_split(
    participant_ids, test_size=TEST_SIZE, stratify=participant_labels, random_state=RANDOM_SEED
)

print(f"\nParticipant split: Train={len(train_pids)}, Test={len(test_pids)}")

# Fit scaler on train frames only (all frames concatenated across train participants)
train_frames_list = []
for pid in train_pids:
    train_frames_list.append(all_participant_raw[pid]['X_raw'])
train_frames_concat = np.vstack(train_frames_list)
scaler = StandardScaler()
scaler.fit(train_frames_concat)

def augment_and_windowize(X, scaler):
    """Given raw frames X (n_frames x n_feats), returns list of windows (seq_len x feats_aug)"""
    Xs = scaler.transform(X)
    windows = []
    start = 0
    while start + MAX_SEQ_LEN <= Xs.shape[0]:
        w = Xs[start:start + MAX_SEQ_LEN]  # (T, F)
        channels = [w]
        # add delta (first difference) channel if requested
        if ADD_DELTA_FEATURES:
            delta = np.vstack([np.zeros((1, w.shape[1])), np.diff(w, axis=0)])
            channels.append(delta)
        # add binary activation per AU (threshold) if requested
        if ADD_BINARY_FEATURES:
            binary = (w > BINARY_THRESHOLD).astype(np.float32)
            channels.append(binary)
        # concatenate along feature axis (so each time-step feature vector is extended)
        w_aug = np.concatenate(channels, axis=1)  # (T, F * n_channels)
        windows.append(w_aug.astype(np.float32))
        start += WINDOW_STEP
    return windows

# Build participant_data with augmented windows
for pid, meta in all_participant_raw.items():
    windows = augment_and_windowize(meta['X_raw'], scaler)
    if len(windows) == 0:
        continue
    participant_data[pid] = {
        'windows': windows,
        'label': meta['label'],
        'gender': meta['gender']
    }

print(f"\nSuccessfully prepared {len(participant_data)} participants with windows")

# Flatten windows into train/test sets
def flatten_pids(pids):
    windows = []
    labels = []
    genders = []
    pids_used = []
    for pid in pids:
        if pid not in participant_data:
            continue
        for w in participant_data[pid]['windows']:
            windows.append(w)
            labels.append(participant_data[pid]['label'])
            genders.append(participant_data[pid]['gender'])
        pids_used.append(pid)
    return np.array(windows), np.array(labels), np.array(genders), pids_used

X_train, y_train, gender_train, train_pids_used = flatten_pids(train_pids)
X_test, y_test, gender_test, test_pids_used = flatten_pids(test_pids)

print(f"\nWindow counts: Train={len(X_train)}, Test={len(X_test)}")
print(f"Class distribution (train): 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")

# -----------------------------
# BALANCING: WeightedRandomSampler for training
# -----------------------------
if BALANCE_WITH_SAMPLER:
    # compute sample weights inverse to class frequency
    class_sample_counts = np.bincount(y_train)
    class_weights = 1.0 / class_sample_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("Using WeightedRandomSampler to balance training classes.")
else:
    sampler = None

# -----------------------------
# DATASET & DATALOADER
# -----------------------------
class AUDataset(Dataset):
    def __init__(self, X, y, gender):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.gender = torch.tensor(gender, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.gender[idx], self.y[idx]

train_loader = DataLoader(
    AUDataset(X_train, y_train, gender_train),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=(sampler is None)
)
test_loader = DataLoader(
    AUDataset(X_test, y_test, gender_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# TRANSFORMER MODEL (unchanged architecture but adapt input dims)
# -----------------------------
class AUTransformerWithGender(nn.Module):
    def __init__(self, num_features, num_classes=2, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(num_features, d_model)
        self.gender_embedding = nn.Embedding(3, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, gender):
        # x shape: (B, T, F)
        x = self.input_fc(x)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        au_representation = x.mean(dim=1)
        au_representation = self.norm(au_representation)
        gender_input = torch.where(gender >= 0, gender, torch.tensor(2, device=gender.device))
        gender_emb = self.gender_embedding(gender_input)
        combined = torch.cat([au_representation, gender_emb], dim=1)
        logits = self.classifier(combined)
        return logits

num_input_features = X_train.shape[2]
model = AUTransformerWithGender(num_features=num_input_features).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# -----------------------------
# LOSS & OPTIMIZER
# -----------------------------
# recompute class weight ratio on training windows (for loss)
num_class0 = int(np.sum(y_train == 0))
num_class1 = int(np.sum(y_train == 1))
weight_ratio = num_class0 / (num_class1 + 1e-8)
ce_class_weights = torch.tensor([1.0, weight_ratio], dtype=torch.float32).to(DEVICE)
print(f"Class weights (CE): [1.0, {weight_ratio:.4f}]")

if USE_FOCAL_LOSS:
    # Simple focal loss implementation
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, weight=None):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
            self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        def forward(self, logits, targets):
            logpt = -self.ce(logits, targets)  # negative cross-entropy per sample
            pt = torch.exp(logpt)
            loss = -((1 - pt) ** self.gamma) * logpt
            return loss.mean()
    criterion = FocalLoss(gamma=2.0, weight=ce_class_weights)
else:
    criterion = nn.CrossEntropyLoss(weight=ce_class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# TRAINING (single or separate gender models)
# -----------------------------
def train_and_evaluate(train_loader, test_loader, model, criterion, optimizer, epochs=EPOCHS, out_prefix="au_v7"):
    best_val_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_snapshot = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        iter_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") if TQDM_AVAILABLE else train_loader
        for X_batch, gender_batch, y_batch in iter_loader:
            X_batch = X_batch.to(DEVICE); gender_batch = gender_batch.to(DEVICE); y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch, gender_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs, all_labels, all_preds = [], [], []

        with torch.no_grad():
            iter_val = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]") if TQDM_AVAILABLE else test_loader
            for X_batch, gender_batch, y_batch in iter_val:
                X_batch = X_batch.to(DEVICE); gender_batch = gender_batch.to(DEVICE); y_batch = y_batch.to(DEVICE)
                outputs = model(X_batch, gender_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train loss={train_loss:.4f}, Train acc={train_acc:.4f} | Val loss={val_loss:.4f}, Val acc={val_acc:.4f}, Val AUC={val_auc:.4f}")

        # Save best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_snapshot = {
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'history': history
            }
            torch.save(best_snapshot, f"{out_prefix}_best.pth")
            print("  ✓ Best model saved")

    # Final evaluation on best snapshot (reload for stability)
    if best_snapshot is not None:
        model.load_state_dict(best_snapshot['model_state_dict'])

    # Compute final metrics on test set
    all_probs, all_preds, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for X_batch, gender_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE); gender_batch = gender_batch.to(DEVICE)
            outputs = model(X_batch, gender_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # handle degenerate cases
        tn = fp = fn = tp = 0

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = best_val_auc

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'auc': float(auc),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

    print("\nEVALUATION METRICS:")
    print(json.dumps(metrics, indent=2))
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, digits=4))
    return metrics

# -----------------------------
# Run training (single model or per-gender)
# -----------------------------
results = {}
if TRAIN_GENDER_MODELS:
    for gender_code, label in [(1, 'male'), (0, 'female')]:
        print(f"\n--- TRAINING gender={label} ---")
        # filter windows by gender
        train_mask = gender_train == gender_code
        test_mask = gender_test == gender_code
        if train_mask.sum() < 10 or test_mask.sum() < 10:
            print(f"Not enough samples for gender={label}: train {train_mask.sum()}, test {test_mask.sum()}, skipping.")
            continue
        train_loader_g = DataLoader(AUDataset(X_train[train_mask], y_train[train_mask], gender_train[train_mask]),
                                    batch_size=BATCH_SIZE,
                                    sampler=WeightedRandomSampler(weights=(1.0/np.bincount(y_train[train_mask]))[y_train[train_mask]], num_samples=len(y_train[train_mask]), replacement=True),
                                    shuffle=False)
        test_loader_g = DataLoader(AUDataset(X_test[test_mask], y_test[test_mask], gender_test[test_mask]),
                                   batch_size=BATCH_SIZE, shuffle=False)
        model_g = AUTransformerWithGender(num_features=num_input_features).to(DEVICE)
        optimizer_g = torch.optim.Adam(model_g.parameters(), lr=1e-4)
        metrics_g = train_and_evaluate(train_loader_g, test_loader_g, model_g, criterion, optimizer_g, epochs=EPOCHS, out_prefix=f"au_v7_{label}")
        results[f"gender_{label}"] = metrics_g
else:
    metrics_all = train_and_evaluate(train_loader, test_loader, model, criterion, optimizer, epochs=EPOCHS, out_prefix="au_v7_all")
    results['all'] = metrics_all

# Save summary
out_dir = "./au_v7_results"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/summary_metrics.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\nDone. Results saved to", out_dir)
print("="*70)


# EVALUATION METRICS:
# {
#   "accuracy": 0.3951074870274277,
#   "precision": 0.265642151481888,
#   "recall": 0.622107969151671,
#   "f1": 0.3723076923076923,
#   "specificity": 0.303125,
#   "auc": 0.4751216914214986,
#   "tp": 726,
#   "tn": 873,
#   "fp": 2007,
#   "fn": 441
# }

# CLASSIFICATION REPORT:
#               precision    recall  f1-score   support

#            0     0.6644    0.3031    0.4163      2880
#            1     0.2656    0.6221    0.3723      1167

#     accuracy                         0.3951      4047
#    macro avg     0.4650    0.4626    0.3943      4047
# weighted avg     0.5494    0.3951    0.4036      4047