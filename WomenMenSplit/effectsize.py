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
# 2. Split by gender
# -----------------------------
females = data[data['Gender'] == 0].copy()
males = data[data['Gender'] == 1].copy()

# -----------------------------
# 3. Select regression AU columns (_r)
# -----------------------------
au_r_cols = [col for col in data.columns if col.endswith('_r')]

# Replace placeholder values (-100) with NaN
females.loc[:, au_r_cols] = females.loc[:, au_r_cols].replace(-100, pd.NA)
males.loc[:, au_r_cols] = males.loc[:, au_r_cols].replace(-100, pd.NA)

# -----------------------------
# 4. Compute mean AU per participant
# -----------------------------
female_means = females.groupby('Participant_ID')[au_r_cols].mean()
male_means = males.groupby('Participant_ID')[au_r_cols].mean()

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
    female_values = female_means[au].dropna().values
    male_values = male_means[au].dropna().values
    
    if len(female_values) == 0 or len(male_values) == 0:
        continue
    
    stat, p = mannwhitneyu(female_values, male_values, alternative='two-sided')
    delta = cliffs_delta(female_values, male_values)
    
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
print("Mann-Whitney U test + Cliff's Delta for AU_r (Women vs Men):")
print(results_df.round(4))


# Mann-Whitney U test + Cliff's Delta for AU_r (Women vs Men):
#         AU  p_value  Significant  Cliffs_Delta
# 3   AU05_r   0.0296         True        0.1839
# 9   AU15_r   0.0300         True        0.1835
# 10  AU17_r   0.0450         True        0.1695
# 5   AU09_r   0.1601        False       -0.1188
# 7   AU12_r   0.2233        False        0.1030
# 4   AU06_r   0.2563        False        0.0960
# 0   AU01_r   0.3751        False        0.0751
# 1   AU02_r   0.3765        False       -0.0748
# 8   AU14_r   0.4559        False        0.0631
# 12  AU25_r   0.5654        False        0.0487
# 6   AU10_r   0.6588        False        0.0374
# 2   AU04_r   0.6763        False        0.0354
# 13  AU26_r   0.7702        False        0.0248
# 11  AU20_r   0.9989        False        0.0002