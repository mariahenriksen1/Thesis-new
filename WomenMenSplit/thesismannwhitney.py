import os
import pandas as pd
from scipy.stats import mannwhitneyu
from glob import glob

# Folder with cleaned participant AU files
data_folder = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_with_gender"

# Load all participant files
all_files = glob(os.path.join(data_folder, "*_CLNF_AUs_cleaned_with_gender.csv"))

all_data_list = []
for file in all_files:
    df = pd.read_csv(file)
    # Ensure Participant_ID column exists
    if 'Participant_ID' not in df.columns:
        participant_id = int(os.path.basename(file).split('_')[0])
        df['Participant_ID'] = participant_id
    all_data_list.append(df)

all_data = pd.concat(all_data_list, ignore_index=True)

# Map numeric gender to strings
all_data['Gender'] = all_data['Gender'].map({0: 'Female', 1: 'Male'})

# Identify AU columns
au_columns = [col for col in all_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID (mean AU per participant)
aggregated = all_data.groupby(['Participant_ID', 'Gender'])[au_columns].mean().reset_index()

# Split by gender
females_agg = aggregated[aggregated['Gender'] == 'Female']
males_agg = aggregated[aggregated['Gender'] == 'Male']

# Mann-Whitney U test function
def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

# Run test for each AU
p_values = []
au_names = []

for au in au_columns:
    female_vals = females_agg[au]
    male_vals = males_agg[au]

    U, p_value = mann_whitney_u_test(female_vals, male_vals)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

# Save results
results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

# Sort by p-value
results_df = results_df.sort_values(by='P-Value')

print("Mann-Whitney U Test Results for All AUs:")
print(results_df)

