import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
MIN_CONFIDENCE = 0.8
OUTPUT_DIR = "./emotion_coactivation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_PATTERNS = {
    'Happiness': ['AU06_r', 'AU12_r'],
    'Sadness': ['AU01_r', 'AU04_r', 'AU15_r'],
    'Surprise': ['AU01_r', 'AU02_r', 'AU05_r', 'AU26_r'],
    'Fear': ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU20_r', 'AU26_r'],
    'Disgust': ['AU09_r', 'AU15_r'],
    'Anger': ['AU04_r', 'AU05_r']
}

ALL_AUS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
    'AU25_r', 'AU26_r'
]

# -----------------------------
# LOAD PARTICIPANT DATA
# -----------------------------
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))
participant_data = []

for path in tqdm(file_paths, desc="Processing participants"):
    participant_id = os.path.basename(path).replace('_CLNF_AUs_final.csv', '')
    
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    
    if 'confidence' in df.columns:
        df = df[df['confidence'] >= MIN_CONFIDENCE]
    if 'success' in df.columns:
        df = df[df['success'] == 1]
    if len(df) < 10:
        continue
    if not all(col in df.columns for col in ALL_AUS):
        continue
    
    label_key = next((k for k in ['PHQ8_Binary', 'PHQ8', 'phq8', 'phq8_binary'] if k in df.columns), None)
    if label_key is None:
        continue
    label = int(df[label_key].iloc[0])
    
    gender = None
    if 'Gender' in df.columns:
        val = df.iloc[0]['Gender']
        if isinstance(val, str):
            gender = 1 if val.lower() in ['male', 'm', '1'] else 0
        else:
            gender = int(val)
    
    au_means = df[ALL_AUS].mean()
    participant_entry = {
        'participant_id': participant_id,
        'PHQ8_Binary': label,
        'Gender': gender if gender is not None else -1,
        **{au: au_means[au] for au in ALL_AUS}
    }
    
    for emotion, aus in EMOTION_PATTERNS.items():
        participant_entry[f"{emotion}_score"] = au_means[aus].mean()
    
    participant_data.append(participant_entry)

df_participant = pd.DataFrame(participant_data)

# -----------------------------
# CREATE GROUP COLUMN
# -----------------------------
def group_label(row):
    if row['PHQ8_Binary']==0 and row['Gender']==1:
        return 'NonDep_Men'
    elif row['PHQ8_Binary']==0 and row['Gender']==0:
        return 'NonDep_Women'
    elif row['PHQ8_Binary']==1 and row['Gender']==1:
        return 'Dep_Men'
    elif row['PHQ8_Binary']==1 and row['Gender']==0:
        return 'Dep_Women'
    else:
        return 'Unknown'

df_participant['Group'] = df_participant.apply(group_label, axis=1)

# -----------------------------
# PLOT VIOLIN PLOTS FOR EACH EMOTION
# -----------------------------
sns.set(style="whitegrid")
for emotion in EMOTION_PATTERNS.keys():
    plt.figure(figsize=(8,5))
    sns.violinplot(
        x='Group',
        y=f'{emotion}_score',
        data=df_participant,
        palette="Set2",
        inner='quartile'
    )
    plt.title(f"Distribution of {emotion} Scores Across Groups")
    plt.ylabel(f"{emotion} Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
