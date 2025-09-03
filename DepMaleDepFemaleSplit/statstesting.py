import pandas as pd
from scipy.stats import mannwhitneyu

# Load data
depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv")
depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv")

# Get AU columns
au_columns = [col for col in depressed_females.columns if col.endswith('_r')]

# Aggregate by participant
depressed_females_agg = depressed_females.groupby('Participant_ID').mean()
depressed_males_agg = depressed_males.groupby('Participant_ID').mean()

def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

p_values = []
au_names = []

for au in au_columns:
    female_data = depressed_females_agg[au]
    male_data = depressed_males_agg[au]

    U, p_value = mann_whitney_u_test(female_data, male_data)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values:")
print(results_df.sort_values(by='P-Value'))
