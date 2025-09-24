import os
import pandas as pd
from scipy.stats import mannwhitneyu

data_folder = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Load all files into one DataFrame
all_data = []
for file in all_files:
    df = pd.read_csv(os.path.join(data_folder, file))
    # Extract participant ID from filename
    participant_id = int(file.split('_')[0])
    df['Participant_ID'] = participant_id
    all_data.append(df)

all_data = pd.concat(all_data, ignore_index=True)

# Ensure column names are clean
all_data.columns = all_data.columns.str.strip()

# Split by depression status
depressed_data = all_data[all_data['PHQ8_Binary'] == 1]
nondepressed_data = all_data[all_data['PHQ8_Binary'] == 0]

# Aggregate by Participant_ID (mean values per participant)
depressed_agg = depressed_data.groupby('Participant_ID').mean()
nondepressed_agg = nondepressed_data.groupby('Participant_ID').mean()

# Select AU columns (those ending with "_r")
au_columns = [col for col in depressed_agg.columns if col.endswith('_r')]

# Run Mann-Whitney U test
results = []
for au in au_columns:
    U, p = mannwhitneyu(depressed_agg[au], nondepressed_agg[au], alternative='two-sided')
    results.append({'AU': au, 'P-Value': round(p, 5)})

results_df = pd.DataFrame(results).sort_values(by='P-Value')
print(results_df)

#Results 
#         AU  P-Value
# 10  AU17_r  0.12631
# 0   AU01_r  0.19704
# 11  AU20_r  0.22352
# 9   AU15_r  0.25245
# 6   AU10_r  0.31367
# 1   AU02_r  0.36359
# 2   AU04_r  0.62469
# 8   AU14_r  0.63088
# 3   AU05_r  0.71803
# 13  AU26_r  0.72894
# 5   AU09_r  0.81127
# 7   AU12_r  0.83848
# 4   AU06_r  0.87734
# 12  AU25_r  0.97212