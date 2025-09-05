import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to DepMaleDepFemaleSplit (assuming it's one folder up from the script)
dep_split_dir = os.path.abspath(os.path.join(script_dir, "..", "DepMaleDepFemaleSplit"))

# Load CSVs without hardcoding absolute paths
dep_females = pd.read_csv(os.path.join(dep_split_dir, "depressed_females.csv"))
dep_males = pd.read_csv(os.path.join(dep_split_dir, "depressed_males.csv"))


dep_females['Gender'] = 'Female'
dep_males['Gender'] = 'Male'

au_c_cols = ['AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']


dep_females[au_c_cols] = dep_females[au_c_cols].replace(-100, pd.NA)
dep_males[au_c_cols] = dep_males[au_c_cols].replace(-100, pd.NA)

auc_combined_c = pd.concat([
    dep_females[['Participant_ID', 'Gender'] + au_c_cols],
    dep_males[['Participant_ID', 'Gender'] + au_c_cols]
])

# Sadness proxy: AU04 + AU15
auc_combined_c['Sadness'] = (auc_combined_c['AU04_c'] & auc_combined_c['AU15_c'])

# Anger proxy: AU04 + AU23
auc_combined_c['Anger'] = (auc_combined_c['AU04_c'] & auc_combined_c['AU23_c'])

# Contempt proxy: AU12 alone (weak, but some studies use AU12+14 and AU12 by itself for contempt smile)
auc_combined_c['Contempt/Contempt Smile/Happy-Like'] = (auc_combined_c['AU12_c'])  

emotion_coactivation_rate = auc_combined_c.groupby('Gender')[['Sadness', 'Anger', 'Contempt/Contempt Smile/Happy-Like']].mean().reset_index()

print(emotion_coactivation_rate.round(4))
