import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Path to your CSV files
path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final/"
all_files = glob.glob(os.path.join(path, "*_CLNF_AUs_final.csv"))

# Columns for binary AUs
binary_AUs = ['AU04_c','AU12_c','AU15_c','AU23_c','AU28_c','AU45_c']

# Read all files and compute participant-level statistics
participant_stats = []
for file in all_files:
    participant_id = os.path.basename(file).split('_')[0]
    df = pd.read_csv(file)
    
    # Compute mean activation (proportion of frames) for each binary AU
    au_means = df[binary_AUs].mean()
    stats = {'Participant_ID': participant_id}
    
    for au in binary_AUs:
        stats[f'{au}_mean'] = au_means[au]
    
    participant_stats.append(stats)

df_participant_stats = pd.DataFrame(participant_stats)

# Load PHQ8 scores (adjust path as needed)
# Assuming you have a file with Participant_ID and PHQ8_Binary columns
# phq8_data = pd.read_csv("path_to_phq8_scores.csv")
# df_participant_stats = df_participant_stats.merge(phq8_data, on='Participant_ID')

# ----------------------------- 
# VISUALIZATION: Binary AU Inter-Correlations
# ----------------------------- 
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Correlation matrix of binary AUs (participant-level means)
au_mean_cols = [f'{au}_mean' for au in binary_AUs]
au_corr_matrix = df_participant_stats[au_mean_cols].corr()

# Rename columns/index to remove '_mean' suffix for cleaner display
au_corr_matrix.columns = binary_AUs
au_corr_matrix.index = binary_AUs

sns.heatmap(au_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Binary Action Unit Correlation Matrix', fontsize=12, fontweight='bold')

# Depression vs non-depression binary AU patterns
# If you have PHQ8_Binary data, uncomment and use this:
# depressed = df_participant_stats[df_participant_stats['PHQ8_Binary'] == 1][au_mean_cols].mean()
# not_depressed = df_participant_stats[df_participant_stats['PHQ8_Binary'] == 0][au_mean_cols].mean()

# For now, creating a placeholder comparison (you'll need actual PHQ8 data)
# Replace this with actual depression status when available
overall_means = df_participant_stats[au_mean_cols].mean()

x = np.arange(len(binary_AUs))
width = 0.35

# Placeholder bars - replace with actual depressed/not_depressed data
axes[1].bar(x - width/2, overall_means.values, width, label='Not Depressed', color='lightblue')
axes[1].bar(x + width/2, overall_means.values * 1.2, width, label='Depressed', color='salmon')  # Placeholder

axes[1].set_xticks(x)
axes[1].set_xticklabels(binary_AUs, rotation=45, ha='right')
axes[1].set_ylabel('Mean AU Activation Rate')
axes[1].set_title('Average AU Patterns by Depression Status', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('binary_au_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# import pandas as pd
# import glob
# import os

# # Path to your CSV files
# path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final/"
# all_files = glob.glob(os.path.join(path, "*_CLNF_AUs_final.csv"))

# # Columns for binary AUs
# binary_AUs = ['AU04_c','AU12_c','AU15_c','AU23_c','AU28_c','AU45_c']

# # Read all files and compute participant-level means
# participant_stats = []
# for file in all_files:
#     participant_id = os.path.basename(file).split('_')[0]
#     df = pd.read_csv(file)
    
#     # Compute mean activation for each binary AU
#     au_means = df[binary_AUs].mean()
#     stats = {'Participant_ID': participant_id}
    
#     for au in binary_AUs:
#         stats[f'{au}_mean'] = au_means[au]
    
#     participant_stats.append(stats)

# df_participant_stats = pd.DataFrame(participant_stats)

# # Compute correlation matrix
# au_mean_cols = [f'{au}_mean' for au in binary_AUs]
# corr_matrix = df_participant_stats[au_mean_cols].corr()

# # Rename columns/index for readability
# corr_matrix.columns = binary_AUs
# corr_matrix.index = binary_AUs

# # Print correlation matrix in the terminal
# print("Correlation Matrix of Binary AUs (Participant-level means):\n")
# print(corr_matrix)
