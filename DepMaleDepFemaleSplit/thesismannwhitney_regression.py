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

# Filter only depressed participants
depressed_data = data[data['PHQ8_Binary'] == 1]

# Split by gender
females = depressed_data[depressed_data['Gender'] == 0]
males = depressed_data[depressed_data['Gender'] == 1]

# Select AU columns
au_columns = [col for col in depressed_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID (mean per participant)
females_agg = females.groupby('Participant_ID')[au_columns].mean()
males_agg = males.groupby('Participant_ID')[au_columns].mean()

# Define Mann-Whitney U test
def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

# Run tests for each AU
p_values = []
au_names = []

for au in au_columns:
    U, p_value = mann_whitney_u_test(females_agg[au], males_agg[au])
    au_names.append(au)
    p_values.append(round(p_value, 5))

# Build results dataframe
results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values (Depressed Women vs Depressed Men):")
print(results_df.sort_values(by='P-Value'))

# All AUs with their P-Values (Depressed Women vs Depressed Men):
#         AU  P-Value
# 5   AU09_r  0.03633
# 10  AU17_r  0.09932
# 7   AU12_r  0.24862
# 3   AU05_r  0.28403
# 9   AU15_r  0.39142
# 1   AU02_r  0.46833
# 12  AU25_r  0.57522
# 2   AU04_r  0.65632
# 6   AU10_r  0.65632
# 4   AU06_r  0.68031
# 11  AU20_r  0.70463
# 8   AU14_r  0.71691
# 13  AU26_r  0.86909
# 0   AU01_r  0.97370