from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np

non_depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
non_depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

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
female_flips = compute_flips(non_depressed_females, au_columns)
male_flips = compute_flips(non_depressed_males, au_columns)

# Run Mann-Whitney U tests
results = {}
for au in au_columns:
    female_data = female_flips[au]
    male_data = male_flips[au]

    u_stat, p_value = mannwhitneyu(female_data, male_data, alternative='two-sided')

    results[au] = {
        'U-statistic': u_stat,
        'p-value': p_value
    }

# Print results
print("Mann-Whitney U Test Results (Non-Depressed Females vs Males)\n")
for au, result in results.items():
    print(f"{au}:")
    print(f"  U-statistic = {result['U-statistic']:.3f}")
    print(f"  p-value = {result['p-value']:.4f}\n")
