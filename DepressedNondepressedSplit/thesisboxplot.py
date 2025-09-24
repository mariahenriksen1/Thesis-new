import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Folder with final cleaned participant files
data_folder = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

# Get all participant CSV files
all_files = glob(os.path.join(data_folder, "*_CLNF_AUs_final.csv"))

# Load and combine all participant files
all_data_list = []
for file in all_files:
    df = pd.read_csv(file)
    # Ensure Participant_ID exists
    if 'Participant_ID' not in df.columns:
        participant_id = int(os.path.basename(file).split('_')[0])
        df['Participant_ID'] = participant_id
    all_data_list.append(df)

all_data = pd.concat(all_data_list, ignore_index=True)

# Map PHQ8_Binary to labels
all_data['Depression_Status'] = all_data['PHQ8_Binary'].map({0: 'Non-Depressed', 1: 'Depressed'})

# Identify AU columns (those ending with "_r")
au_columns = [col for col in all_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID and Depression_Status
aggregated = all_data.groupby(['Participant_ID', 'Depression_Status'])[au_columns].mean().reset_index()

# Count participants by depression status
participant_counts = aggregated.groupby('Depression_Status')['Participant_ID'].nunique()
print("Number of participants by Depression Status:")
print(participant_counts)

# Melt the data for seaborn plotting
aggregated_melted = aggregated.melt(
    id_vars=['Participant_ID', 'Depression_Status'],
    value_vars=au_columns,
    var_name='AU',
    value_name='Value'
)

# Custom colors
custom_palette = {"Non-Depressed": "#1f77b4", "Depressed": "#FFD700"}  

# Plotting
plt.figure(figsize=(16, 6))
ax = sns.boxplot(
    data=aggregated_melted,
    x='AU', y='Value', hue='Depression_Status',
    palette=custom_palette,
    showfliers=False
)

plt.title('Boxplot of Action Units Split by Depression Status')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the figure
output_file = "/Users/raemarshall/Desktop/Thesis-new/DepressedNondepressedSplit/au_boxplot_by_depression_status.png"
plt.savefig(output_file, dpi=300)
print(f"Boxplot saved to {output_file}")

plt.show()

# Check which participants are missing PHQ8_Binary
missing_phq8 = all_data[all_data['PHQ8_Binary'].isna()]['Participant_ID'].unique()
print("Participants missing PHQ8_Binary:", missing_phq8)
