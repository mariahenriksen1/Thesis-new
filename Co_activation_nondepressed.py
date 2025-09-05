import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to NonDepFemaleNonDepMaleSplit (assuming it's one folder up from the script)
nondep_split_dir = os.path.abspath(os.path.join(script_dir, "..", "NonDepFemaleNonDepMaleSplit"))

# Load datasets without hardcoded paths
nondep_females = pd.read_csv(os.path.join(nondep_split_dir, "non_depressed_females.csv"))
nondep_males = pd.read_csv(os.path.join(nondep_split_dir, "non_depressed_males.csv"))


nondep_females['Gender'] = 'Female'
nondep_males['Gender'] = 'Male'

au_c_cols = ['AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']

nondep_females[au_c_cols] = nondep_females[au_c_cols].replace(-100, pd.NA)
nondep_males[au_c_cols] = nondep_males[au_c_cols].replace(-100, pd.NA)

auc_combined_c = pd.concat([
    nondep_females[['Participant_ID', 'Gender'] + au_c_cols],
    nondep_males[['Participant_ID', 'Gender'] + au_c_cols]
])

# Sadness proxy: AU04 + AU15
auc_combined_c['Sadness'] = (auc_combined_c['AU04_c'] & auc_combined_c['AU15_c'])

# Anger proxy: AU04 + AU23
auc_combined_c['Anger'] = (auc_combined_c['AU04_c'] & auc_combined_c['AU23_c'])

# Contempt proxy: AU12 alone (weak, but some studies use AU12+14 and AU12 by itself for contempt smile)
auc_combined_c['Contempt/Contempt Smile/Happy-Like'] = (auc_combined_c['AU12_c'])  

emotion_coactivation_rate = auc_combined_c.groupby('Gender')[['Sadness', 'Anger', 'Contempt/Contempt Smile/Happy-Like']].mean().reset_index()

print(emotion_coactivation_rate.round(4))
