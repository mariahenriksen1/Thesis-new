import pandas as pd
from scipy.stats import mannwhitneyu
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
non_depressed_females = pd.read_csv(os.path.join(script_dir, "non_depressed_females.csv"))
non_depressed_males = pd.read_csv(os.path.join(script_dir, "non_depressed_males.csv"))

# Get AU columns
au_columns = [col for col in non_depressed_females.columns if col.endswith('_r')]

# Aggregate by participant
non_depressed_females_agg = non_depressed_females.groupby('Participant_ID').mean()
non_depressed_males_agg = non_depressed_males.groupby('Participant_ID').mean()

def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

p_values = []
au_names = []

for au in au_columns:
    female_data = non_depressed_females_agg[au]
    male_data = non_depressed_males_agg[au]

    U, p_value = mann_whitney_u_test(female_data, male_data)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values:")
print(results_df.sort_values(by='P-Value'))
