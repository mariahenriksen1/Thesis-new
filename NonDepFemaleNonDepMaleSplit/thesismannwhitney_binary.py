import os
import pandas as pd
from scipy.stats import mannwhitneyu

folder_path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

all_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith("_CLNF_AUs_final.csv"):
        participant_id = int(file_name.split("_")[0])
        df = pd.read_csv(os.path.join(folder_path, file_name))
        df['Participant_ID'] = participant_id
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

non_depressed_data = data[data['PHQ8_Binary'] == 0]

# Split by gender
females = non_depressed_data[non_depressed_data['Gender'] == 0]
males = non_depressed_data[non_depressed_data['Gender'] == 1]

female_outlier_ids = [314, 438]
male_outlier_ids = [305, 364]

females = females[~females['Participant_ID'].isin(female_outlier_ids)]
males = males[~males['Participant_ID'].isin(male_outlier_ids)]

au_c_cols = [col for col in non_depressed_data.columns if col.endswith('_c')]
females[au_c_cols] = females[au_c_cols].replace(-100, pd.NA)
males[au_c_cols] = males[au_c_cols].replace(-100, pd.NA)

def compute_au_proportion(df, au_columns):
    grouped = df.groupby('Participant_ID')
    percentages = grouped[au_columns].mean() * 100  # % of frames AU active
    percentages['Participant_ID'] = grouped.size().index
    return percentages.reset_index(drop=True)

female_percentages = compute_au_proportion(females, au_c_cols)
male_percentages = compute_au_proportion(males, au_c_cols)

# -----------------------------
# 6. Perform Mann-Whitney U tests
# -----------------------------
results = {}
for au in au_c_cols:
    female_values = female_percentages[au]
    male_values = male_percentages[au]
    
    stat, p = mannwhitneyu(female_values, male_values, alternative='two-sided')
    results[au] = p

# Convert results to DataFrame
pvals_df = pd.DataFrame.from_dict(results, orient='index', columns=['p_value'])
pvals_df['Significant'] = pvals_df['p_value'] < 0.05


print("Mann-Whitney U test results for AU_c proportions (Non-Depressed Women vs Non-Depressed Men):")
print(pvals_df.round(4))

# Mann-Whitney U test results for AU_c proportions (Non-Depressed Women vs Non-Depressed Men):
#         p_value  Significant
# AU04_c   0.1916        False
# AU12_c   0.0039         True
# AU15_c   0.8504        False
# AU23_c   0.0173         True
# AU28_c   0.7007        False
# AU45_c   0.2470        False