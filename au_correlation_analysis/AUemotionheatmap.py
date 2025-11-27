import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
MIN_CONFIDENCE = 0.8
OUTPUT_DIR = "./emotion_coactivation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion AU patterns from literature (Table II)
EMOTION_PATTERNS = {
    'Happiness': ['AU06_r', 'AU12_r'],
    'Sadness': ['AU01_r', 'AU04_r', 'AU15_r'],
    'Surprise': ['AU01_r', 'AU02_r', 'AU05_r', 'AU26_r'],
    'Fear': ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU20_r', 'AU26_r'],
    'Disgust': ['AU09_r', 'AU15_r'],  # AU16 not in your data
    'Anger': ['AU04_r', 'AU05_r']  # AU7, AU23 not in your data
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

for path in file_paths:
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
    label_key = None
    for k in ['PHQ8_Binary', 'PHQ8', 'phq8', 'phq8_binary']:
        if k in df.columns:
            label_key = k
            break
    if label_key is None:
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
    
    # Participant-level averages
    au_means = df[ALL_AUS].mean()
    participant_data.append({
        'participant_id': participant_id,
        'PHQ8_Binary': label,
        'Gender': gender if gender is not None else -1,
        **{au: au_means[au] for au in ALL_AUS}
    })
    
    # Frame-level data (sample for computational efficiency)
    sample_size = min(1000, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)
    for _, row in sampled_df.iterrows():
        frame_level_data.append({
            'participant_id': participant_id,
            'PHQ8_Binary': label,
            'Gender': gender if gender is not None else -1,
            **{au: row[au] for au in ALL_AUS}
        })

df_participant = pd.DataFrame(participant_data)
df_frames = pd.DataFrame(frame_level_data)

print(f"Loaded {len(df_participant)} participants")
print(f"Sampled {len(df_frames)} frames for frame-level analysis")
print(f"  Not Depressed: {len(df_participant[df_participant['PHQ8_Binary']==0])}")
print(f"  Depressed: {len(df_participant[df_participant['PHQ8_Binary']==1])}")

# -----------------------------
# ANALYSIS 1: EMOTION PATTERN CO-ACTIVATION SCORES
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 1: EMOTION PATTERN CO-ACTIVATION")
print("="*70)

def compute_emotion_score(df, emotion_aus):
    """Compute how strongly an emotion pattern is expressed"""
    # Average intensity of AUs in the emotion pattern
    return df[emotion_aus].mean(axis=1)

def compute_coactivation_strength(df, emotion_aus):
    """Measure how often AUs in the pattern occur together (correlation)"""
    if len(emotion_aus) < 2:
        return np.nan
    
    # Get all pairwise correlations within the emotion pattern
    correlations = []
    for au1, au2 in combinations(emotion_aus, 2):
        if au1 in df.columns and au2 in df.columns:
            corr = df[au1].corr(df[au2])
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else np.nan

# Compute emotion scores for each participant
emotion_scores = {}
for emotion, aus in EMOTION_PATTERNS.items():
    available_aus = [au for au in aus if au in df_participant.columns]
    if len(available_aus) > 0:
        df_participant[f'{emotion}_score'] = compute_emotion_score(df_participant, available_aus)
        emotion_scores[emotion] = {
            'mean_score': df_participant[f'{emotion}_score'].mean(),
            'coactivation': compute_coactivation_strength(df_participant, available_aus),
            'aus': available_aus
        }

print("\nEmotion Pattern Strength (Participant-Level):")
print(f"{'Emotion':<12} {'Mean Score':<12} {'Co-activation':<15} {'AUs'}")
print("-" * 70)
for emotion, stats in sorted(emotion_scores.items(), key=lambda x: x[1]['mean_score'], reverse=True):
    print(f"{emotion:<12} {stats['mean_score']:<12.4f} {stats['coactivation']:<15.4f} {', '.join(stats['aus'])}")

# -----------------------------
# ANALYSIS 2: EMOTION PATTERNS BY DEPRESSION STATUS
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 2: EMOTION PATTERNS BY DEPRESSION STATUS")
print("="*70)

depression_emotion_comparison = []

for emotion in EMOTION_PATTERNS.keys():
    score_col = f'{emotion}_score'
    if score_col not in df_participant.columns:
        continue
    
    not_dep = df_participant[df_participant['PHQ8_Binary'] == 0][score_col]
    depressed = df_participant[df_participant['PHQ8_Binary'] == 1][score_col]
    
    # T-test
    t_stat, p_val = scipy.stats.ttest_ind(not_dep, depressed)

    # Effect size (Cohen's d)
    cohens_d = (not_dep.mean() - depressed.mean()) / np.sqrt(
        ((len(not_dep)-1)*not_dep.std()**2 + (len(depressed)-1)*depressed.std()**2) / 
        (len(not_dep) + len(depressed) - 2)
    )
    
    depression_emotion_comparison.append({
        'Emotion': emotion,
        'Not_Depressed_Mean': not_dep.mean(),
        'Depressed_Mean': depressed.mean(),
        'Difference': not_dep.mean() - depressed.mean(),
        'p_value': p_val,
        'cohens_d': cohens_d,
        'significant': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    })

df_comparison = pd.DataFrame(depression_emotion_comparison).sort_values('Difference', key=abs, ascending=False)

print("\nEmotion Expression Differences:")
print(f"{'Emotion':<12} {'Not Dep':<10} {'Depressed':<10} {'Diff':<10} {'p-value':<10} {'Cohens d':<10} {'Sig'}")
print("-" * 80)
for _, row in df_comparison.iterrows():
    print(f"{row['Emotion']:<12} {row['Not_Depressed_Mean']:<10.4f} {row['Depressed_Mean']:<10.4f} "
          f"{row['Difference']:<10.4f} {row['p_value']:<10.4f} {row['cohens_d']:<10.4f} {row['significant']}")

# -----------------------------
# ANALYSIS 3: WITHIN-EMOTION AU CORRELATIONS
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 3: AU CO-ACTIVATION WITHIN EMOTION PATTERNS")
print("="*70)

for emotion, aus in EMOTION_PATTERNS.items():
    available_aus = [au for au in aus if au in df_participant.columns]
    if len(available_aus) < 2:
        continue
    
    print(f"\n{emotion.upper()} Pattern ({', '.join(available_aus)}):")
    print("-" * 50)
    
    # Calculate all pairwise correlations
    for au1, au2 in combinations(available_aus, 2):
        corr = df_participant[au1].corr(df_participant[au2])
        print(f"  {au1} ↔ {au2}: r = {corr:.4f}")

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

gender_emotion_comparison = []

for emotion in EMOTION_PATTERNS.keys():
    score_col = f'{emotion}_score'
    if score_col not in df_participant.columns:
        continue
    
    male_score = df_male[score_col].mean()
    female_score = df_female[score_col].mean()
    
    gender_emotion_comparison.append({
        'Emotion': emotion,
        'Male': male_score,
        'Female': female_score,
        'Difference': abs(male_score - female_score)
    })

df_gender = pd.DataFrame(gender_emotion_comparison).sort_values('Difference', ascending=False)

print(f"\n{'Emotion':<12} {'Male Mean':<12} {'Female Mean':<12} {'|Difference|'}")
print("-" * 50)
for _, row in df_gender.iterrows():
    print(f"{row['Emotion']:<12} {row['Male']:<12.4f} {row['Female']:<12.4f} {row['Difference']:<12.4f}")

# -----------------------------
# VISUALIZATIONS
# -----------------------------
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Emotion Pattern Heatmap (Within-Emotion Correlations)
ax1 = fig.add_subplot(gs[0, :2])
emotion_corr_matrix = []
emotion_labels = []

for emotion, aus in EMOTION_PATTERNS.items():
    available_aus = [au for au in aus if au in df_participant.columns]
    if len(available_aus) >= 2:
        corr_matrix = df_participant[available_aus].corr()
        # Get upper triangle values
        for i in range(len(available_aus)):
            for j in range(i+1, len(available_aus)):
                emotion_corr_matrix.append([emotion, available_aus[i], available_aus[j], corr_matrix.iloc[i, j]])

df_emotion_corr = pd.DataFrame(emotion_corr_matrix, columns=['Emotion', 'AU1', 'AU2', 'Correlation'])
emotion_pivot = df_emotion_corr.pivot_table(values='Correlation', index='Emotion', aggfunc='mean')

colors = ['red' if x < 0 else 'green' for x in emotion_pivot['Correlation'].values]
ax1.barh(emotion_pivot.index, emotion_pivot['Correlation'].values, color=colors)
ax1.set_xlabel('Average Within-Emotion AU Correlation')
ax1.set_title('Emotion Pattern Co-Activation Strength\n(Higher = AUs occur together more)', 
              fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Emotion Scores by Depression Status
ax2 = fig.add_subplot(gs[0, 2])
emotions_to_plot = df_comparison['Emotion'].tolist()
x = np.arange(len(emotions_to_plot))
width = 0.35

not_dep_scores = [df_comparison[df_comparison['Emotion']==e]['Not_Depressed_Mean'].values[0] for e in emotions_to_plot]
dep_scores = [df_comparison[df_comparison['Emotion']==e]['Depressed_Mean'].values[0] for e in emotions_to_plot]

ax2.bar(x - width/2, not_dep_scores, width, label='Not Depressed', color='lightblue')
ax2.bar(x + width/2, dep_scores, width, label='Depressed', color='salmon')
ax2.set_xticks(x)
ax2.set_xticklabels(emotions_to_plot, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Mean Emotion Score')
ax2.set_title('Emotion Expression by Depression', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Happiness Pattern Detail (AU06 vs AU12)
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(df_participant[df_participant['PHQ8_Binary']==0]['AU06_r'],
           df_participant[df_participant['PHQ8_Binary']==0]['AU12_r'],
           alpha=0.6, s=30, label='Not Depressed', color='lightblue')
ax3.scatter(df_participant[df_participant['PHQ8_Binary']==1]['AU06_r'],
           df_participant[df_participant['PHQ8_Binary']==1]['AU12_r'],
           alpha=0.6, s=30, label='Depressed', color='salmon')
ax3.set_xlabel('AU06 (Cheek Raiser)')
ax3.set_ylabel('AU12 (Smile)')
ax3.set_title('Happiness Pattern: AU06 vs AU12\n(Duchenne Smile)', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Sadness Pattern Detail (AU01+AU04+AU15)
ax4 = fig.add_subplot(gs[1, 1])
sadness_aus = ['AU01_r', 'AU04_r', 'AU15_r']
sadness_not_dep = df_participant[df_participant['PHQ8_Binary']==0][sadness_aus].mean()
sadness_dep = df_participant[df_participant['PHQ8_Binary']==1][sadness_aus].mean()

x = np.arange(len(sadness_aus))
width = 0.35
ax4.bar(x - width/2, sadness_not_dep.values, width, label='Not Depressed', color='lightblue')
ax4.bar(x + width/2, sadness_dep.values, width, label='Depressed', color='salmon')
ax4.set_xticks(x)
ax4.set_xticklabels(sadness_aus, fontsize=9)
ax4.set_ylabel('Mean AU Intensity')
ax4.set_title('Sadness Pattern Components', fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Disgust Pattern (AU09+AU15)
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(df_participant[df_participant['PHQ8_Binary']==0]['AU09_r'],
           df_participant[df_participant['PHQ8_Binary']==0]['AU15_r'],
           alpha=0.6, s=30, label='Not Depressed', color='lightblue')
ax5.scatter(df_participant[df_participant['PHQ8_Binary']==1]['AU09_r'],
           df_participant[df_participant['PHQ8_Binary']==1]['AU15_r'],
           alpha=0.6, s=30, label='Depressed', color='salmon')
ax5.set_xlabel('AU09 (Nose Wrinkler)')
ax5.set_ylabel('AU15 (Lip Corner Depressor)')
ax5.set_title('Disgust Pattern: AU09 vs AU15', fontsize=10, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# Plot 6: Gender Differences in Emotion Expression
ax6 = fig.add_subplot(gs[2, 0])
emotions_gender = df_gender['Emotion'].tolist()
x = np.arange(len(emotions_gender))
width = 0.35

male_scores = df_gender['Male'].values
female_scores = df_gender['Female'].values

ax6.bar(x - width/2, male_scores, width, label='Male', color='steelblue')
ax6.bar(x + width/2, female_scores, width, label='Female', color='coral')
ax6.set_xticks(x)
ax6.set_xticklabels(emotions_gender, rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('Mean Emotion Score')
ax6.set_title('Emotion Expression by Gender', fontsize=10, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Effect Sizes
ax7 = fig.add_subplot(gs[2, 1])
emotions_effect = df_comparison['Emotion'].tolist()
effect_sizes = df_comparison['cohens_d'].values
colors_effect = ['red' if x < 0 else 'blue' for x in effect_sizes]

ax7.barh(emotions_effect, effect_sizes, color=colors_effect)
ax7.set_xlabel("Cohen's d (Effect Size)")
ax7.set_title('Effect Sizes: Not Depressed vs Depressed\n(Negative = Lower in Depression)', 
              fontsize=10, fontweight='bold')
ax7.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax7.axvline(x=-0.2, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax7.axvline(x=0.2, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax7.grid(axis='x', alpha=0.3)

# Plot 8: Emotion Pattern Correlation Matrix
ax8 = fig.add_subplot(gs[2, 2])
emotion_score_cols = [f'{e}_score' for e in EMOTION_PATTERNS.keys() if f'{e}_score' in df_participant.columns]
emotion_corr = df_participant[emotion_score_cols].corr()
emotion_corr.columns = [c.replace('_score', '') for c in emotion_corr.columns]
emotion_corr.index = [c.replace('_score', '') for c in emotion_corr.index]

sns.heatmap(emotion_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax8)
ax8.set_title('Inter-Emotion Correlations', fontsize=10, fontweight='bold')

plt.savefig(f"{OUTPUT_DIR}/emotion_coactivation_analysis.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to {OUTPUT_DIR}/emotion_coactivation_analysis.png")

# -----------------------------
# SAVE DETAILED RESULTS
# -----------------------------
df_comparison.to_csv(f"{OUTPUT_DIR}/emotion_depression_comparison.csv", index=False)
df_gender.to_csv(f"{OUTPUT_DIR}/emotion_gender_comparison.csv", index=False)

# Save emotion scores per participant
emotion_score_data = df_participant[['participant_id', 'PHQ8_Binary', 'Gender'] + 
                                     [col for col in df_participant.columns if '_score' in col]]
emotion_score_data.to_csv(f"{OUTPUT_DIR}/participant_emotion_scores.csv", index=False)

print(f"✓ Results saved to {OUTPUT_DIR}/")

# -----------------------------
# KEY FINDINGS SUMMARY
# -----------------------------
print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)

print("\n🎭 STRONGEST EMOTION PATTERNS (Co-activation):")
top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1]['coactivation'], reverse=True)[:3]
for i, (emotion, stats) in enumerate(top_emotions, 1):
    print(f"  {i}. {emotion}: r = {stats['coactivation']:.4f} ({', '.join(stats['aus'])})")

print("\n😔 EMOTIONS MOST AFFECTED BY DEPRESSION:")
most_affected = df_comparison.nlargest(3, 'Difference', keep='all')
for _, row in most_affected.iterrows():
    direction = "↓" if row['Difference'] > 0 else "↑"
    print(f"  • {row['Emotion']}: {direction} {abs(row['Difference']):.4f} (Cohen's d = {row['cohens_d']:.3f}) {row['significant']}")

print("\n🚹🚺 LARGEST GENDER DIFFERENCES:")
gender_diffs = df_gender.nlargest(3, 'Difference')
for _, row in gender_diffs.iterrows():
    higher = "Male" if row['Male'] > row['Female'] else "Female"
    print(f"  • {row['Emotion']}: {higher} higher by {row['Difference']:.4f}")

print("\n💡 RECOMMENDATIONS FOR YOUR MODEL:")
print("  Based on co-activation strength, prioritize:")
for emotion, stats in top_emotions:
    print(f"  → {emotion} interaction: " + " × ".join(stats['aus']))

print("\n" + "="*70)