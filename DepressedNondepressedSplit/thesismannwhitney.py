import os
import pandas as pd
from scipy.stats import mannwhitneyu

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
depressed_file = os.path.join(script_dir, "depressed_participants.csv")
nondepressed_file = os.path.join(script_dir, "non_depressed_participants.csv")

# Load your data
depressed_data = pd.read_csv(depressed_file)
nondepressed_data = pd.read_csv(nondepressed_file)

# Select AU columns (those ending with "_r")
au_columns = [col for col in depressed_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID (mean values per participant)
depressed_agg = depressed_data.groupby('Participant_ID').mean()
nondepressed_agg = nondepressed_data.groupby('Participant_ID').mean()

# Define Mann-Whitney U test
def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

# Run tests for each AU
p_values = []
au_names = []

for au in au_columns:
    dep_values = depressed_agg[au]
    nondep_values = nondepressed_agg[au]

    U, p_value = mann_whitney_u_test(dep_values, nondep_values)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

# Build results dataframe
results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values:")
print(results_df.sort_values(by='P-Value'))
