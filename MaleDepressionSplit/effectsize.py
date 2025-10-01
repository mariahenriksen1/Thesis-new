import os
import pandas as pd
from scipy.stats import mannwhitneyu

# -----------------------------
# 1. Load all participant CSVs
# -----------------------------
folder_path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

all_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith("_CLNF_AUs_final.csv"):
        participant_id = int(file_name.split("_")[0])
        df = pd.read_csv(os.path.join(folder_path, file_name))
        df['Participant_ID'] = participant_id
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# -----------------------------
# 2. Subset only males
# -----------------------------
males = data[data['Gender'] == 1].copy()

# -----------------------------
# 3. Split males by depression status
# -----------------------------
male_depressed = males[males['PHQ8_Binary'] == 1].copy()
male_non_depressed = males[males['PHQ8_Binary'] == 0].copy()

# -----------------------------
# 4. Select regression AU columns (_r)
# -----------------------------
au_r_cols = [col for col in data.columns if col.endswith('_r')]

# Replace placeholder values (-100) with NaN
male_depressed.loc[:, au_r_cols] = male_depressed.loc[:, au_r_cols].replace(-100, pd.NA)
male_non_depressed.loc[:, au_r_cols] = male_non_depressed.loc[:, au_r_cols].replace(-100, pd.NA)

# -----------------------------
# 5. Compute mean AU per participant
# -----------------------------
dep_means = male_depressed.groupby('Participant_ID')[au_r_cols].mean()
nondep_means = male_non_depressed.groupby('Participant_ID')[au_r_cols].mean()

# -----------------------------
# 6. Function to compute Cliff's delta
# -----------------------------
def cliffs_delta(x, y):
    """
    Computes Cliff's delta effect size between two arrays.
    Returns value between -1 and 1.
    """
    n_x = len(x)
    n_y = len(y)
    more = sum([1 for xi in x for yi in y if xi > yi])
    less = sum([1 for xi in x for yi in y if xi < yi])
    delta = (more - less) / (n_x * n_y)
    return delta

# -----------------------------
# 7. Perform Mann-Whitney U and compute Cliff's delta
# -----------------------------
results = []
for au in au_r_cols:
    dep_values = dep_means[au].dropna().values
    nondep_values = nondep_means[au].dropna().values
    
    if len(dep_values) == 0 or len(nondep_values) == 0:
        continue
    
    stat, p = mannwhitneyu(dep_values, nondep_values, alternative='two-sided')
    delta = cliffs_delta(dep_values, nondep_values)
    
    results.append({
        'AU': au,
        'p_value': p,
        'Significant': p < 0.05,
        'Cliffs_Delta': delta
    })

results_df = pd.DataFrame(results).sort_values('p_value')

# -----------------------------
# 8. Display results
# -----------------------------
print("Mann-Whitney U test + Cliff's Delta for AU_r (Men Depressed vs Men Non-depressed):")
print(results_df.round(4))

# Mann-Whitney U test + Cliff's Delta for AU_r (Men Depressed vs Men Non-depressed):
#         AU  p_value  Significant  Cliffs_Delta
# 10  AU17_r   0.0999        False       -0.2203
# 5   AU09_r   0.2609        False        0.1506
# 6   AU10_r   0.4207        False       -0.1081
# 11  AU20_r   0.4207        False        0.1081
# 9   AU15_r   0.5622        False       -0.0779
# 0   AU01_r   0.5887        False       -0.0727
# 3   AU05_r   0.6158        False       -0.0675
# 7   AU12_r   0.6490        False       -0.0613
# 1   AU02_r   0.6659        False       -0.0582
# 8   AU14_r   0.6944        False        0.0530
# 4   AU06_r   0.7705        False       -0.0395
# 13  AU26_r   0.8917        False       -0.0187
# 12  AU25_r   0.9287        False        0.0125
# 2   AU04_r   0.9411        False        0.0104