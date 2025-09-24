import os
import pandas as pd
from scipy.stats import mannwhitneyu

# Folder containing all participant CSVs
folder_path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

# Load all participant data
all_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith("_CLNF_AUs_final.csv"):
        participant_id = int(file_name.split("_")[0])
        df = pd.read_csv(os.path.join(folder_path, file_name))
        df['Participant_ID'] = participant_id
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Filter male participants
male_data = data[data['Gender'] == 1]

# Split into depressed and non-depressed
depressed_males = male_data[male_data['PHQ8_Binary'] == 1]
non_depressed_males = male_data[male_data['PHQ8_Binary'] == 0]

# Count participants in each group for sanity check
male_counts = male_data.groupby('Participant_ID')['PHQ8_Binary'].first().value_counts()
print("Counts of Depressed vs Non-Depressed Males (by participant):")
print(f"Non-Depressed (0): {male_counts.get(0, 0)}")
print(f"Depressed (1): {male_counts.get(1, 0)}")

# Select AU columns
au_columns = [col for col in male_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID (mean per participant)
dep_agg = depressed_males.groupby('Participant_ID')[au_columns].mean()
nondep_agg = non_depressed_males.groupby('Participant_ID')[au_columns].mean()

# Mann-Whitney U test function
def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

# Run test for each AU
p_values = []
au_names = []

for au in au_columns:
    dep_values = dep_agg[au]
    nondep_values = nondep_agg[au]
    U, p_value = mann_whitney_u_test(dep_values, nondep_values)
    p_values.append(round(p_value, 5))
    au_names.append(au)

# Build results dataframe
results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

# Sort by P-Value
results_df = results_df.sort_values(by='P-Value')
print("Mann-Whitney U Test Results: Depressed vs Non-Depressed Males")
print(results_df)

# Mann-Whitney U Test Results: Depressed vs Non-Depressed Males
#         AU  P-Value
# 10  AU17_r  0.09989
# 5   AU09_r  0.26095
# 6   AU10_r  0.42071
# 11  AU20_r  0.42071
# 9   AU15_r  0.56220
# 0   AU01_r  0.58873
# 3   AU05_r  0.61582
# 7   AU12_r  0.64903
# 1   AU02_r  0.66591
# 8   AU14_r  0.69441
# 4   AU06_r  0.77049
# 13  AU26_r  0.89171
# 12  AU25_r  0.92871
# 2   AU04_r  0.94108