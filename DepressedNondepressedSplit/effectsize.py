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
# 2. Split by depression status
# -----------------------------
depressed = data[data['PHQ8_Binary'] == 1].copy()
non_depressed = data[data['PHQ8_Binary'] == 0].copy()

# -----------------------------
# 3. Select regression AU columns (_r)
# -----------------------------
au_r_cols = [col for col in data.columns if col.endswith('_r')]

# Replace placeholder values (-100) with NaN
depressed.loc[:, au_r_cols] = depressed.loc[:, au_r_cols].replace(-100, pd.NA)
non_depressed.loc[:, au_r_cols] = non_depressed.loc[:, au_r_cols].replace(-100, pd.NA)

# -----------------------------
# 4. Compute mean AU per participant
# -----------------------------
depressed_means = depressed.groupby('Participant_ID')[au_r_cols].mean()
non_depressed_means = non_depressed.groupby('Participant_ID')[au_r_cols].mean()

# -----------------------------
# 5. Function to compute Cliff's delta
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
# 6. Perform Mann-Whitney U and compute Cliff's delta
# -----------------------------
results = []
for au in au_r_cols:
    dep_values = depressed_means[au].dropna().values
    nondep_values = non_depressed_means[au].dropna().values
    
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
# 7. Display results
# -----------------------------
print("Mann-Whitney U test + Cliff's Delta for AU_r (Depressed vs Non-depressed):")
print(results_df.round(4))

# Mann-Whitney U test + Cliff's Delta for AU_r (Depressed vs Non-depressed):
#         AU  p_value  Significant  Cliffs_Delta
# 10  AU17_r   0.1263        False       -0.1411
# 0   AU01_r   0.1970        False       -0.1191
# 11  AU20_r   0.2235        False        0.1124
# 9   AU15_r   0.2524        False       -0.1057
# 6   AU10_r   0.3137        False       -0.0930
# 1   AU02_r   0.3636        False       -0.0839
# 2   AU04_r   0.6247        False        0.0452
# 8   AU14_r   0.6309        False       -0.0444
# 3   AU05_r   0.7180        False       -0.0334
# 13  AU26_r   0.7289        False       -0.0321
# 5   AU09_r   0.8113        False       -0.0222
# 7   AU12_r   0.8385        False        0.0189
# 4   AU06_r   0.8773        False       -0.0144
# 12  AU25_r   0.9721        False        0.0034