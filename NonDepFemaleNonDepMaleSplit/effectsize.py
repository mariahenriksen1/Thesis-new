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
# 2. Subset only non-depressed participants
# -----------------------------
non_depressed = data[data['PHQ8_Binary'] == 0].copy()

# -----------------------------
# 3. Split non-depressed by gender
# -----------------------------
nondep_women = non_depressed[non_depressed['Gender'] == 0].copy()
nondep_men   = non_depressed[non_depressed['Gender'] == 1].copy()

# -----------------------------
# 4. Select regression AU columns (_r)
# -----------------------------
au_r_cols = [col for col in data.columns if col.endswith('_r')]

# Replace placeholder values (-100) with NaN
nondep_women.loc[:, au_r_cols] = nondep_women.loc[:, au_r_cols].replace(-100, pd.NA)
nondep_men.loc[:, au_r_cols]   = nondep_men.loc[:, au_r_cols].replace(-100, pd.NA)

# -----------------------------
# 5. Compute mean AU per participant
# -----------------------------
women_means = nondep_women.groupby('Participant_ID')[au_r_cols].mean()
men_means   = nondep_men.groupby('Participant_ID')[au_r_cols].mean()

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
print("Mann-Whitney U test + Cliff's Delta for AU_r (Non-depressed Women vs Non-depressed Men):")
print(results_df.round(4))

# Mann-Whitney U test + Cliff's Delta for AU_r (Non-depressed Women vs Non-depressed Men):
#         AU  p_value  Significant  Cliffs_Delta
# 9   AU15_r   0.0285         True        0.2231
# 3   AU05_r   0.0519        False        0.1981
# 10  AU17_r   0.1023        False        0.1665
# 4   AU06_r   0.2334        False        0.1215
# 0   AU01_r   0.2370        False        0.1206
# 8   AU14_r   0.2406        False        0.1197
# 7   AU12_r   0.5339        False        0.0635
# 1   AU02_r   0.5429        False       -0.0622
# 12  AU25_r   0.6934        False        0.0404
# 13  AU26_r   0.7480        False        0.0329
# 5   AU09_r   0.7968        False       -0.0264
# 6   AU10_r   0.8144        False        0.0241
# 2   AU04_r   0.9003        False        0.0130
# 11  AU20_r   0.9328        False        0.0088