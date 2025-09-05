import os 
import pandas as pd
import numpy as np

# Get the dicrectory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load files
depressed_females = pd.read_csv(os.path.join(script_dir, "depressed_females.csv"))
non_depressed_females = pd.read_csv(os.path.join(script_dir, "non_depressed_females.csv"))


au_columns = [col for col in depressed_females.columns if col.endswith('_c')]

def compute_flips_and_normalize(df, au_columns):
    flips = []

    for pid, group in df.groupby('Participant_ID'):
        group_flips = {}

        for col in au_columns:
            values = group[col].values
            clean_values = pd.Series(values).dropna().values
            if len(clean_values) > 1:
                flip_count = (clean_values[1:] != clean_values[:-1]).sum()  
            else:
                flip_count = 0
            group_flips[col] = flip_count

        group_flips['Participant_ID'] = pid
        flips.append(group_flips)

    return pd.DataFrame(flips)

depressed_female_flips = compute_flips_and_normalize(depressed_females, au_columns)
non_depressed_female_flips = compute_flips_and_normalize(non_depressed_females, au_columns)

non_depressed_females_AU15 = non_depressed_female_flips['AU45_c']
depressed_females_AU15 = depressed_female_flips['AU45_c']

mean_non_depressed = non_depressed_females_AU15.mean()
mean_depressed = depressed_females_AU15.mean()

std_non_depressed = non_depressed_females_AU15.std(ddof=1) 
std_depressed = depressed_females_AU15.std(ddof=1) 

n_non_depressed = len(non_depressed_females_AU15)
n_depressed = len(depressed_females_AU15)

sd_pooled = np.sqrt(((n_non_depressed - 1) * std_non_depressed**2 + (n_depressed - 1) * std_depressed**2) / (n_non_depressed + n_depressed - 2))

cohens_d = (mean_non_depressed - mean_depressed) / sd_pooled

print(f"Cohen's d for AU45 Female: {cohens_d:.3f}")
