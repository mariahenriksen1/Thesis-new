import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
MIN_CONFIDENCE = 0.8
OUTPUT_DIR = "./emotion_coactivation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion AU patterns from literature
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

print("="*70)
print("EMOTION-SPECIFIC AU CO-ACTIVATION ANALYSIS")
print("="*70)
print("\nAnalyzing theoretical emotion patterns:")
for emotion, aus in EMOTION_PATTERNS.items():
    print(f"  {emotion}: {', '.join(aus)}")

# -----------------------------
# LOAD PARTICIPANT DATA
# -----------------------------
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))
print(f"\nLoading {len(file_paths)} participant files...")

participant_data = []
frame_level_data = []

# Wrap the loop with tqdm for progress display
for path in tqdm(file_paths, desc="Processing participants"):
    participant_id = os.path.basename(path).replace('_CLNF_AUs_final.csv', '')
    
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    
    # Filter by confidence and success
    if 'confidence' in df.columns:
        df = df[df['confidence'] >= MIN_CONFIDENCE]
    if 'success' in df.columns:
        df = df[df['success'] == 1]
    
    if len(df) < 10:
        continue
    
    if not all(col in df.columns for col in ALL_AUS):
        continue
    
    # Get label
    label_key = next((k for k in ['PHQ8_Binary', 'PHQ8', 'phq8', 'phq8_binary'] if k in df.columns), None)
    if label_key is None:
        continue
    label = int(df[label_key].iloc[0])
    
    # Get gender
    gender = None
    if 'Gender' in df.columns:
        val = df.iloc[0]['Gender']
        if isinstance(val, str):
            gender = 1 if val.lower() in ['male', 'm', '1'] else 0
        else:
            gender = int(val)
    
    # Participant-level averages
    au_means = df[ALL_AUS].mean()
    participant_entry = {
        'participant_id': participant_id,
        'PHQ8_Binary': label,
        'Gender': gender if gender is not None else -1,
        **{au: au_means[au] for au in ALL_AUS}
    }

    # Calculate emotion scores
    for emotion, aus in EMOTION_PATTERNS.items():
        participant_entry[f"{emotion}_score"] = au_means[aus].mean()
    
    participant_data.append(participant_entry)
    
    # Frame-level data (use all frames)
    sampled_df = df
    for _, row in sampled_df.iterrows():
        frame_entry = {
            'participant_id': participant_id,
            'PHQ8_Binary': label,
            'Gender': gender if gender is not None else -1,
            **{au: row[au] for au in ALL_AUS}
        }
        for emotion, aus in EMOTION_PATTERNS.items():
            frame_entry[f"{emotion}_score"] = row[aus].mean()
        frame_level_data.append(frame_entry)

df_participant = pd.DataFrame(participant_data)
df_frames = pd.DataFrame(frame_level_data)

print(f"\nLoaded {len(df_participant)} participants")
print(f"Total frames processed: {len(df_frames)}")
print(f"  Not Depressed: {len(df_participant[df_participant['PHQ8_Binary']==0])}")
print(f"  Depressed: {len(df_participant[df_participant['PHQ8_Binary']==1])}")

# -----------------------------
# ANALYSIS 4: GENDER-SPECIFIC EMOTION PATTERNS
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 4: GENDER-SPECIFIC EMOTION EXPRESSION")
print("="*70)

df_male = df_participant[df_participant['Gender'] == 1]
df_female = df_participant[df_participant['Gender'] == 0]

print(f"\nMale participants: {len(df_male)}")
print(f"Female participants: {len(df_female)}")

results = []

for emotion in EMOTION_PATTERNS.keys():
    score_col = f"{emotion}_score"
    male_scores = df_male[score_col].dropna()
    female_scores = df_female[score_col].dropna()

    male_mean = male_scores.mean()
    female_mean = female_scores.mean()

    # Independent samples t-test
    t_stat, p_value = ttest_ind(male_scores, female_scores, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt(((male_scores.std() ** 2) + (female_scores.std() ** 2)) / 2)
    cohens_d = (male_mean - female_mean) / pooled_std if pooled_std > 0 else 0

    results.append({
        "Emotion": emotion,
        "Male_Mean": male_mean,
        "Female_Mean": female_mean,
        "Difference": male_mean - female_mean,
        "p-value": p_value,
        "Cohens_d": cohens_d
    })

df_gender_stats = pd.DataFrame(results).sort_values("p-value")

print("\nEmotion        Male Mean   Female Mean   Diff       p-value    Cohens d   Sig")
print("--------------------------------------------------------------------------")
for _, r in df_gender_stats.iterrows():
    sig = "*" if r["p-value"] < 0.05 else ""
    print(f"{r['Emotion']:<13}{r['Male_Mean']:<12.4f}{r['Female_Mean']:<12.4f}"
          f"{r['Difference']:<11.4f}{r['p-value']:<11.4g}{r['Cohens_d']:<11.4f}{sig}")
