import os
import pandas as pd
from scipy.stats import mannwhitneyu

# Folder containing all participant CSVs
folder_path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

# Read all CSV files and add Participant_ID
all_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith("_CLNF_AUs_final.csv"):
        participant_id = int(file_name.split("_")[0])
        df = pd.read_csv(os.path.join(folder_path, file_name))
        df['Participant_ID'] = participant_id
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Filter only female participants
female_data = data[data['Gender'] == 0]

# Split female data into depressed and non-depressed
depressed_females = female_data[female_data['PHQ8_Binary'] == 1]
non_depressed_females = female_data[female_data['PHQ8_Binary'] == 0]

# Select AU columns (those ending with "_r")
au_columns = [col for col in female_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID (mean per participant)
dep_agg = depressed_females.groupby('Participant_ID')[au_columns].mean()
nondep_agg = non_depressed_females.groupby('Participant_ID')[au_columns].mean()

# Define Mann-Whitney U test
def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

# Run tests for each AU
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

# Print results
print("AUs with their P-Values (Depressed vs Non-Depressed Females):")
print(results_df)

# AUs with their P-Values (Depressed vs Non-Depressed Females):
#         AU  P-Value
# 0   AU01_r  0.11984
# 9   AU15_r  0.14486
# 8   AU14_r  0.22636
# 5   AU09_r  0.27761
# 11  AU20_r  0.32741
# 10  AU17_r  0.35898
# 1   AU02_r  0.45391
# 6   AU10_r  0.52051
# 7   AU12_r  0.56157
# 2   AU04_r  0.56756
# 13  AU26_r  0.58571
# 3   AU05_r  0.69329
# 4   AU06_r  0.80744
# 12  AU25_r  0.98939