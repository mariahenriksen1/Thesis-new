import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Path to folder with cleaned participant AU files
data_folder = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_with_gender"

# Get list of all cleaned CSV files
all_files = glob(os.path.join(data_folder, "*_CLNF_AUs_cleaned_with_gender.csv"))

# Load and combine all files
all_data_list = []
for file in all_files:
    df = pd.read_csv(file)
    # Ensure 'Participant_ID' column exists
    if 'Participant_ID' not in df.columns:
        participant_id = int(os.path.basename(file).split('_')[0])
        df['Participant_ID'] = participant_id
    all_data_list.append(df)

all_data = pd.concat(all_data_list, ignore_index=True)

# Map numeric gender to strings
all_data['Gender'] = all_data['Gender'].map({0: 'Female', 1: 'Male'})

# Identify AU columns (those ending with "_r")
au_columns = [col for col in all_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID and Gender (mean AU values per participant)
aggregated = all_data.groupby(['Participant_ID', 'Gender'])[au_columns].mean().reset_index()

# Melt the data for seaborn plotting
aggregated_melted = aggregated.melt(
    id_vars=['Participant_ID', 'Gender'],
    value_vars=au_columns,
    var_name='AU',
    value_name='Value'
)

# Custom colors
darker_purple = "#9B4D96"  # Female
soft_orange = "#FF8C00"    # Male

# Plotting
plt.figure(figsize=(16, 6))
ax = sns.boxplot(
    data=aggregated_melted,
    x='AU', y='Value', hue='Gender',
    palette=[darker_purple, soft_orange],
    showfliers=False
)

plt.title('Boxplot of AUs by Gender (Aggregated per Participant)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
