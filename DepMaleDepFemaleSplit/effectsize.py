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
# 2. Subset only depressed participants
# -----------------------------
depressed = data[data['PHQ8_Binary'] == 1].copy()

# -----------------------------
# 3. Split depressed by gender
# -----------------------------
dep_women = depressed[depressed['Gender'] == 0].copy()
dep_men   = depressed[depressed['Gender'] == 1].copy()

# -----------------------------
# 4. Select regression AU columns (_r)
# -----------------------------
au_r_cols = [col for col in data.columns if col.endswith('_r')]

# Replace placeholder values (-100) with NaN
dep_women.loc[:, au_r_cols] = dep_women.loc[:, au_r_cols].replace(-100, pd.NA)
dep_men.loc[:, au_r_cols]   = dep_men.loc[:, au_r_cols].replace(-100, pd.NA)

# -----------------------------
# 5. Compute mean AU per participant
# -----------------------------
women_means = dep_women.groupby('Participant_ID')[au_r_cols].mean()
men_means   = dep_men.groupby('Participant_ID')[au_r_cols].mean()

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
    women_values = women_means[au].dropna().values
    men_values   = men_means[au].dropna().values
    
    if len(women_values) == 0 or len(men_values) == 0:
        continue
    
    stat, p = mannwhitneyu(women_values, men_values, alternative='two-sided')
    delta = cliffs_delta(women_values, men_values)
    
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
print("Mann-Whitney U test + Cliff's Delta for AU_r (Women Depressed vs Men Depressed):")
print(results_df.round(4))

# Mann-Whitney U test + Cliff's Delta for AU_r (Women Depressed vs Men Depressed):
#         AU  p_value  Significant  Cliffs_Delta
# 5   AU09_r   0.0363         True       -0.3290
# 10  AU17_r   0.0993        False        0.2594
# 7   AU12_r   0.2486        False        0.1819
# 3   AU05_r   0.2840        False        0.1690
# 9   AU15_r   0.3914        False        0.1355
# 1   AU02_r   0.4683        False       -0.1148
# 12  AU25_r   0.5752        False        0.0890
# 2   AU04_r   0.6563        False        0.0710
# 6   AU10_r   0.6563        False        0.0710
# 4   AU06_r   0.6803        False        0.0658
# 11  AU20_r   0.7046        False       -0.0606
# 8   AU14_r   0.7169        False       -0.0581
# 13  AU26_r   0.8691        False        0.0271
# 0   AU01_r   0.9737        False        0.0065