import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
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

# Emotion AU patterns
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
total_frames = 0

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
    
    total_frames += len(df)
    
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

    # Emotion scores per participant
    for emotion, aus in EMOTION_PATTERNS.items():
        participant_entry[f"{emotion}_score"] = au_means[aus].mean()
    
    participant_data.append(participant_entry)

df_participant = pd.DataFrame(participant_data)
print(f"\nLoaded {len(df_participant)} participants")
print(f"  Not Depressed: {len(df_participant[df_participant['PHQ8_Binary']==0])}")
print(f"  Depressed: {len(df_participant[df_participant['PHQ8_Binary']==1])}")
print(f"Total frames processed: {total_frames}")

# -----------------------------
# FUNCTION TO RUN ANALYSIS
# -----------------------------
def run_group_comparison(df_group1, df_group2, group1_name, group2_name, subgroup_name):
    results = []
    for emotion in EMOTION_PATTERNS.keys():
        score_col = f"{emotion}_score"
        scores1 = df_group1[score_col].dropna()
        scores2 = df_group2[score_col].dropna()
        mean1 = scores1.mean()
        mean2 = scores2.mean()
        t_stat, p_value = ttest_ind(scores1, scores2, equal_var=False)
        pooled_std = np.sqrt(((scores1.std()**2) + (scores2.std()**2)) / 2)
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        results.append({
            "Emotion": emotion,
            group1_name: mean1,
            group2_name: mean2,
            "Difference": mean1 - mean2,
            "p-value": p_value,
            "Cohens_d": cohens_d
        })
    
    df_results = pd.DataFrame(results).sort_values("p-value")
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {subgroup_name}")
    print(f"{'='*70}")
    print(f"{group1_name}: {len(df_group1)}, {group2_name}: {len(df_group2)}")
    print(f"\nEmotion        {group1_name:<11}{group2_name:<11}Diff       p-value    Cohens d   Sig")
    print("-"*73)
    
    for _, r in df_results.iterrows():
        sig = "*" if r["p-value"] < 0.05 else ""
        print(f"{r['Emotion']:<13}{r[group1_name]:<11.4f}{r[group2_name]:<11.4f}"
              f"{r['Difference']:<11.4f}{r['p-value']:<11.4g}{r['Cohens_d']:<11.4f}{sig}")

# -----------------------------
# ANALYSIS 1: Non-Depressed Men vs Depressed Men
# -----------------------------
df_men = df_participant[df_participant['Gender'] == 1]
run_group_comparison(
    df_men[df_men['PHQ8_Binary'] == 0],
    df_men[df_men['PHQ8_Binary'] == 1],
    "NonDep_Men", "Dep_Men", "Non-Depressed Men vs Depressed Men"
)

# -----------------------------
# ANALYSIS 2: Non-Depressed Women vs Depressed Women
# -----------------------------
df_women = df_participant[df_participant['Gender'] == 0]
run_group_comparison(
    df_women[df_women['PHQ8_Binary'] == 0],
    df_women[df_women['PHQ8_Binary'] == 1],
    "NonDep_Women", "Dep_Women", "Non-Depressed Women vs Depressed Women"
)

# -----------------------------
# ANALYSIS 3: Non-Depressed Men vs Non-Depressed Women
# -----------------------------
df_non_dep = df_participant[df_participant['PHQ8_Binary'] == 0]
run_group_comparison(
    df_non_dep[df_non_dep['Gender'] == 1],
    df_non_dep[df_non_dep['Gender'] == 0],
    "NonDep_Men", "NonDep_Women", "Non-Depressed Men vs Non-Depressed Women"
)

# -----------------------------
# ANALYSIS 4: Depressed Men vs Depressed Women
# -----------------------------
df_dep = df_participant[df_participant['PHQ8_Binary'] == 1]
run_group_comparison(
    df_dep[df_dep['Gender'] == 1],
    df_dep[df_dep['Gender'] == 0],
    "Dep_Men", "Dep_Women", "Depressed Men vs Depressed Women"
)
