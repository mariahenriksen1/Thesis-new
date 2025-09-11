import os
import pandas as pd
from scipy.stats import mannwhitneyu

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
female_file = os.path.join(script_dir, "female_participants.csv")
male_file = os.path.join(script_dir, "male_participants.csv")

# Load your data
female_data = pd.read_csv(female_file)
male_data = pd.read_csv(male_file)

au_columns = [col for col in female_data.columns if col.endswith('_r')]

females_agg = female_data.groupby('Participant_ID').mean()
males_agg = male_data.groupby('Participant_ID').mean()

def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

p_values = []
au_names = []

for au in au_columns:
    female_data = females_agg[au]
    male_data = males_agg[au]

    U, p_value = mann_whitney_u_test(female_data, male_data)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values:")
print(results_df.sort_values(by='P-Value'))
