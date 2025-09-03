from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np

# Load data
non_depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/FemaleDepressionSplit/non_depressed_females.csv")
depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/FemaleDepressionSplit/depressed_females.csv")

# Select AU columns
au_columns = [col for col in non_depressed_females.columns if col.endswith('_c')]

# Compute flip counts per participant
def compute_flips(df, au_cols):
    flips = []
    for pid, group in df.groupby('Participant_ID'):
        group_flips = {}
        for col in au_cols:
            values = group[col].values
            clean_values = pd.Series(values).dropna().values
            if len(clean_values) > 1:
                flip_count = (clean_values[1:] != clean_values[:-1]).sum()
            else:
                flip_count = 0
            group_flips[col] = flip_count
        group_flips['Participant_ID'] = pid
        flips.append(group_flips)
    return pd.DataFrame(flips)

# Get flip data
nondep_flips = compute_flips(non_depressed_females, au_columns)
dep_flips = compute_flips(depressed_females, au_columns)

# Run Mann-Whitney U tests
results = {}
for au in au_columns:
    nondep_data = nondep_flips[au]
    dep_data = dep_flips[au]

    u_stat, p_value = mannwhitneyu(nondep_data, dep_data, alternative='two-sided')

    results[au] = {
        'U-statistic': u_stat,
        'p-value': p_value
    }

# Print results
print("Mann-Whitney U Test Results (Non-Depressed vs Depressed Females)\n")
for au, result in results.items():
    print(f"{au}:")
    print(f"  U-statistic = {result['U-statistic']:.3f}")
    print(f"  p-value = {result['p-value']:.4f}\n")
