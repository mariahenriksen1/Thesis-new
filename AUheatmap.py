import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r',
    'AU25_r', 'AU26_r'
]
MIN_CONFIDENCE = 0.8
OUTPUT_DIR = "./au_correlation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("ACTION UNIT CORRELATION ANALYSIS FOR DEPRESSION DETECTION")
print("="*70)

# -----------------------------
# LOAD ALL PARTICIPANT DATA
# -----------------------------
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))
print(f"\nLoading {len(file_paths)} participant files...")

all_data = []
participant_stats = []

for path in file_paths:
    participant_id = os.path.basename(path).replace('_CLNF_AUs_final.csv', '')
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {participant_id}: {e}")
        continue
    
    # Filter by confidence and success
    if 'confidence' in df.columns:
        df = df[df['confidence'] >= MIN_CONFIDENCE]
    if 'success' in df.columns:
        df = df[df['success'] == 1]
    
    if len(df) < 10:  # Skip if too few frames
        continue
    
    # Check for required columns
    if not all(col in df.columns for col in AU_COLUMNS):
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
    
    # Store frame-level data
    frame_data = df[AU_COLUMNS].copy()
    frame_data['PHQ8_Binary'] = label
    frame_data['Gender'] = gender if gender is not None else -1
    frame_data['participant_id'] = participant_id
    all_data.append(frame_data)
    
    # Compute participant-level statistics (mean, std, max, min)
    au_means = df[AU_COLUMNS].mean().values
    au_stds = df[AU_COLUMNS].std().values
    au_maxs = df[AU_COLUMNS].max().values
    au_mins = df[AU_COLUMNS].min().values
    
    stat_dict = {
        'participant_id': participant_id,
        'PHQ8_Binary': label,
        'Gender': gender if gender is not None else -1
    }
    
    for i, col in enumerate(AU_COLUMNS):
        stat_dict[f'{col}_mean'] = au_means[i]
        stat_dict[f'{col}_std'] = au_stds[i]
        stat_dict[f'{col}_max'] = au_maxs[i]
        stat_dict[f'{col}_min'] = au_mins[i]
    
    participant_stats.append(stat_dict)

# Combine all frame-level data
df_all_frames = pd.concat(all_data, ignore_index=True)
df_participant_stats = pd.DataFrame(participant_stats)

print(f"Loaded {len(df_participant_stats)} participants")
print(f"Total frames: {len(df_all_frames)}")
print(f"  Not Depressed: {len(df_participant_stats[df_participant_stats['PHQ8_Binary']==0])}")
print(f"  Depressed: {len(df_participant_stats[df_participant_stats['PHQ8_Binary']==1])}")

# -----------------------------
# ANALYSIS 1: AU CORRELATION WITH DEPRESSION (FRAME-LEVEL)
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 1: FRAME-LEVEL AU-DEPRESSION CORRELATION")
print("="*70)

# Point-biserial correlation (continuous AU vs binary depression)
frame_correlations = {}
for au in AU_COLUMNS:
    corr, pval = stats.pointbiserialr(df_all_frames['PHQ8_Binary'], df_all_frames[au])
    frame_correlations[au] = {'correlation': corr, 'p_value': pval}

frame_corr_df = pd.DataFrame(frame_correlations).T
frame_corr_df = frame_corr_df.sort_values('correlation', key=abs, ascending=False)

print("\nTop AUs correlated with depression (frame-level):")
print(frame_corr_df.head(10).to_string())

# -----------------------------
# ANALYSIS 2: PARTICIPANT-LEVEL CORRELATIONS
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 2: PARTICIPANT-LEVEL AU-DEPRESSION CORRELATION")
print("="*70)

# Use mean AU values per participant
participant_correlations = {}
for au in AU_COLUMNS:
    au_mean_col = f'{au}_mean'
    corr, pval = stats.pointbiserialr(
        df_participant_stats['PHQ8_Binary'], 
        df_participant_stats[au_mean_col]
    )
    participant_correlations[au] = {'correlation': corr, 'p_value': pval}

participant_corr_df = pd.DataFrame(participant_correlations).T
participant_corr_df = participant_corr_df.sort_values('correlation', key=abs, ascending=False)

print("\nTop AUs correlated with depression (participant-level means):")
print(participant_corr_df.head(10).to_string())

# -----------------------------
# ANALYSIS 3: MUTUAL INFORMATION (NON-LINEAR RELATIONSHIPS)
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 3: MUTUAL INFORMATION (Captures Non-Linear Relationships)")
print("="*70)

# Participant-level mutual information
X_participant = df_participant_stats[[f'{au}_mean' for au in AU_COLUMNS]].values
y_participant = df_participant_stats['PHQ8_Binary'].values

mi_scores = mutual_info_classif(X_participant, y_participant, random_state=42)
mi_df = pd.DataFrame({
    'AU': AU_COLUMNS,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\nMutual Information Scores (higher = more informative):")
print(mi_df.to_string(index=False))

# -----------------------------
# ANALYSIS 4: F-STATISTIC (ANOVA)
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 4: F-STATISTIC (ANOVA - Group Differences)")
print("="*70)

f_scores, f_pvals = f_classif(X_participant, y_participant)
f_df = pd.DataFrame({
    'AU': AU_COLUMNS,
    'F_Score': f_scores,
    'p_value': f_pvals
}).sort_values('F_Score', ascending=False)

print("\nF-Statistics (higher = stronger group separation):")
print(f_df.to_string(index=False))

# -----------------------------
# ANALYSIS 5: GENDER-SPECIFIC CORRELATIONS
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS 5: GENDER-SPECIFIC AU-DEPRESSION CORRELATIONS")
print("="*70)

# Split by gender
df_male = df_participant_stats[df_participant_stats['Gender'] == 1]
df_female = df_participant_stats[df_participant_stats['Gender'] == 0]

print(f"\nMale participants: {len(df_male)}")
print(f"Female participants: {len(df_female)}")

gender_correlations = []

for au in AU_COLUMNS:
    au_mean_col = f'{au}_mean'
    
    # Male correlations
    if len(df_male) > 10:
        corr_m, pval_m = stats.pointbiserialr(df_male['PHQ8_Binary'], df_male[au_mean_col])
    else:
        corr_m, pval_m = np.nan, np.nan
    
    # Female correlations
    if len(df_female) > 10:
        corr_f, pval_f = stats.pointbiserialr(df_female['PHQ8_Binary'], df_female[au_mean_col])
    else:
        corr_f, pval_f = np.nan, np.nan
    
    gender_correlations.append({
        'AU': au,
        'Male_Corr': corr_m,
        'Male_p': pval_m,
        'Female_Corr': corr_f,
        'Female_p': pval_f,
        'Difference': abs(corr_m - corr_f) if not np.isnan(corr_m) and not np.isnan(corr_f) else np.nan
    })

gender_corr_df = pd.DataFrame(gender_correlations).sort_values('Difference', ascending=False)
print("\nGender-specific correlations (sorted by difference):")
print(gender_corr_df.to_string(index=False))

# -----------------------------
# VISUALIZATION 1: CORRELATION HEATMAP
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Participant-level AU correlations with depression
corr_values = participant_corr_df['correlation'].values
colors = ['red' if x < 0 else 'blue' for x in corr_values]
axes[0, 0].barh(range(len(AU_COLUMNS)), corr_values, color=colors)
axes[0, 0].set_yticks(range(len(AU_COLUMNS)))
axes[0, 0].set_yticklabels(participant_corr_df.index)
axes[0, 0].set_xlabel('Correlation with Depression')
axes[0, 0].set_title('Participant-Level AU-Depression Correlations\n(Blue=Positive, Red=Negative)', fontsize=12, fontweight='bold')
axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Mutual Information
axes[0, 1].barh(mi_df['AU'], mi_df['MI_Score'], color='green')
axes[0, 1].set_xlabel('Mutual Information Score')
axes[0, 1].set_title('Mutual Information with Depression\n(Higher = More Informative)', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: Gender comparison
x = np.arange(len(AU_COLUMNS))
width = 0.35
axes[1, 0].barh(x - width/2, gender_corr_df['Male_Corr'], width, label='Male', color='steelblue')
axes[1, 0].barh(x + width/2, gender_corr_df['Female_Corr'], width, label='Female', color='coral')
axes[1, 0].set_yticks(x)
axes[1, 0].set_yticklabels(gender_corr_df['AU'])
axes[1, 0].set_xlabel('Correlation with Depression')
axes[1, 0].set_title('Gender-Specific AU-Depression Correlations', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Combined ranking (normalized scores)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

combined_scores = pd.DataFrame({
    'AU': AU_COLUMNS,
    'Correlation': scaler.fit_transform(participant_corr_df['correlation'].abs().values.reshape(-1, 1)).flatten(),
    'MI_Score': scaler.fit_transform(mi_df['MI_Score'].values.reshape(-1, 1)).flatten(),
    'F_Score': scaler.fit_transform(f_df['F_Score'].values.reshape(-1, 1)).flatten()
})
combined_scores['Combined'] = combined_scores[['Correlation', 'MI_Score', 'F_Score']].mean(axis=1)
combined_scores = combined_scores.sort_values('Combined', ascending=False)

axes[1, 1].barh(combined_scores['AU'], combined_scores['Combined'], color='purple')
axes[1, 1].set_xlabel('Combined Importance Score (Normalized)')
axes[1, 1].set_title('Overall AU Importance for Depression Detection\n(Average of Correlation, MI, F-statistic)', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/au_correlation_analysis.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to {OUTPUT_DIR}/au_correlation_analysis.png")

# -----------------------------
# VISUALIZATION 2: AU INTER-CORRELATIONS
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Correlation matrix of AUs (participant-level means)
au_mean_cols = [f'{au}_mean' for au in AU_COLUMNS]
au_corr_matrix = df_participant_stats[au_mean_cols].corr()
au_corr_matrix.columns = AU_COLUMNS
au_corr_matrix.index = AU_COLUMNS

sns.heatmap(au_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('AU Inter-Correlations\n(Participant-Level Means)', fontsize=12, fontweight='bold')

# Depression vs non-depression AU patterns
depressed = df_participant_stats[df_participant_stats['PHQ8_Binary'] == 1][au_mean_cols].mean()
not_depressed = df_participant_stats[df_participant_stats['PHQ8_Binary'] == 0][au_mean_cols].mean()

x = np.arange(len(AU_COLUMNS))
width = 0.35
axes[1].bar(x - width/2, not_depressed.values, width, label='Not Depressed', color='lightblue')
axes[1].bar(x + width/2, depressed.values, width, label='Depressed', color='salmon')
axes[1].set_xticks(x)
axes[1].set_xticklabels(AU_COLUMNS, rotation=45, ha='right')
axes[1].set_ylabel('Mean AU Intensity')
axes[1].set_title('Average AU Patterns by Depression Status', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/au_patterns_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ AU patterns visualization saved to {OUTPUT_DIR}/au_patterns_analysis.png")

# -----------------------------
# SAVE RESULTS TO CSV
# -----------------------------
# Combined results table
results_summary = pd.DataFrame({
    'AU': AU_COLUMNS,
    'Correlation': participant_corr_df['correlation'].values,
    'Corr_pValue': participant_corr_df['p_value'].values,
    'MI_Score': mi_df['MI_Score'].values,
    'F_Score': f_df['F_Score'].values,
    'F_pValue': f_df['p_value'].values,
    'Combined_Score': combined_scores.set_index('AU').loc[AU_COLUMNS, 'Combined'].values
}).sort_values('Combined_Score', ascending=False)

results_summary.to_csv(f"{OUTPUT_DIR}/au_importance_rankings.csv", index=False)
print(f"✓ Results saved to {OUTPUT_DIR}/au_importance_rankings.csv")

# Gender-specific results
gender_corr_df.to_csv(f"{OUTPUT_DIR}/gender_specific_correlations.csv", index=False)
print(f"✓ Gender analysis saved to {OUTPUT_DIR}/gender_specific_correlations.csv")

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
print("\n" + "="*70)
print("RECOMMENDATIONS FOR MODEL SELECTION")
print("="*70)

top_aus = results_summary.head(7)['AU'].tolist()
print(f"\n🎯 TOP 7 MOST IMPORTANT AUs FOR DEPRESSION DETECTION:")
for i, au in enumerate(top_aus, 1):
    score = results_summary[results_summary['AU'] == au]['Combined_Score'].values[0]
    print(f"  {i}. {au} (Combined Score: {score:.4f})")

print(f"\n💡 SUGGESTED AU SUBSETS FOR YOUR MODEL:")
print(f"  • Minimal (Top 5):  {top_aus[:5]}")
print(f"  • Balanced (Top 7): {top_aus[:7]}")
print(f"  • Extended (Top 10): {results_summary.head(10)['AU'].tolist()}")
print(f"  • All AUs (14):     Keep all for maximum information")

# Gender differences
high_diff = gender_corr_df.head(3)['AU'].tolist()
print(f"\n🚹🚺 AUs WITH LARGEST GENDER DIFFERENCES:")
print(f"  {high_diff}")
print(f"  → These AUs may require gender-specific modeling!")

print(f"\n✓ All analysis complete! Results saved to {OUTPUT_DIR}/")
print("="*70)