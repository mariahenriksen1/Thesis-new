import os
import pandas as pd
from scipy.stats import mannwhitneyu

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
nondepressed_female_file = os.path.join(script_dir, "non_depressed_females.csv")
nondepressed_male_file = os.path.join(script_dir, "non_depressed_males.csv")

# Load your data
nondepressed_females = pd.read_csv(nondepressed_female_file)
nondepressed_males = pd.read_csv(nondepressed_male_file)

# Select AU columns (those ending with "_r")
au_columns = [col for col in nondepressed_females.columns if col.endswith('_r')]

# Aggregate by Participant_ID (mean values per participant)
nondep_female_agg = nondepressed_females.groupby('Participant_ID').mean()
nondep_male_agg = nondepressed_males.groupby('Participant_ID').mean()

# Define Mann-Whitney U test
def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

# Run tests for each AU
p_values = []
au_names = []

for au in au_columns:
    female_values = nondep_female_agg[au]
    male_values = nondep_male_agg[au]

    U, p_value = mann_whitney_u_test(female_values, male_values)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

# Build results dataframe
results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values (Non-Depressed Females vs Non-Depressed Males):")
print(results_df.sort_values(by='P-Value'))
