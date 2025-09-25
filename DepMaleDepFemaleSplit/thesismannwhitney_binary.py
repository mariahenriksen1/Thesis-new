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

depressed_data = data[data['PHQ8_Binary'] == 1]

females = depressed_data[depressed_data['Gender'] == 0]
males = depressed_data[depressed_data['Gender'] == 1]

female_lengths = females.groupby('Participant_ID')['frame'].nunique().reset_index(name='Total_Frames')
male_lengths = males.groupby('Participant_ID')['frame'].nunique().reset_index(name='Total_Frames')

# -----------------------------
# 5. Identify IQR-based outliers
# -----------------------------
def find_outliers(lengths_df):
    Q1 = lengths_df['Total_Frames'].quantile(0.25)
    Q3 = lengths_df['Total_Frames'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = lengths_df[(lengths_df['Total_Frames'] < lower) | (lengths_df['Total_Frames'] > upper)]
    return outliers['Participant_ID'].tolist()

female_outlier_ids = find_outliers(female_lengths)
male_outlier_ids = find_outliers(male_lengths)

# -----------------------------
# 6. Remove outliers
# -----------------------------
females = females[~females['Participant_ID'].isin(female_outlier_ids)]
males = males[~males['Participant_ID'].isin(male_outlier_ids)]

# -----------------------------
# 7. Select AU columns and clean
# -----------------------------
au_c_cols = [col for col in depressed_data.columns if col.endswith('_c')]
females[au_c_cols] = females[au_c_cols].replace(-100, pd.NA)
males[au_c_cols] = males[au_c_cols].replace(-100, pd.NA)

# -----------------------------
# 8. Compute AU proportions per participant
# -----------------------------
def compute_au_proportion(df, au_columns):
    grouped = df.groupby('Participant_ID')
    percentages = grouped[au_columns].mean() * 100  # % of frames AU active
    percentages['Participant_ID'] = grouped.size().index
    return percentages.reset_index(drop=True)

female_percentages = compute_au_proportion(females, au_c_cols)
male_percentages = compute_au_proportion(males, au_c_cols)

# -----------------------------
# 9. Perform Mann-Whitney U tests
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

# -----------------------------
# 10. Show results
# -----------------------------
print("Mann-Whitney U test results for AU_c proportions (Depressed Women vs Depressed Men):")
print(pvals_df.round(4))

# Mann-Whitney U test results for AU_c proportions (Depressed Women vs Depressed Men):
#         p_value  Significant
# AU04_c   0.3198        False
# AU12_c   0.1614        False
# AU15_c   0.0084         True
# AU23_c   0.0508        False
# AU28_c   0.7823        False
# AU45_c   0.5679        False