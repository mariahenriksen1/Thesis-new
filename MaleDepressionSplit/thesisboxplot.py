import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Folder containing all participant CSVs
folder_path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

# Read all CSV files and add Participant_ID
all_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith("_CLNF_AUs_final.csv"):
        participant_id = int(file_name.split("_")[0])
        df = pd.read_csv(os.path.join(folder_path, file_name))
        df['Participant_ID'] = participant_id
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Filter only male participants
male_data = data[data['Gender'] == 1]

# Split male data into depressed and non-depressed
depressed_males = male_data[male_data['PHQ8_Binary'] == 1]
non_depressed_males = male_data[male_data['PHQ8_Binary'] == 0]

# Count unique depressed vs non-depressed males
male_counts = male_data.groupby('Participant_ID')['PHQ8_Binary'].first().value_counts()
print("Counts of Depressed vs Non-Depressed Males (by participant):")
print(f"Non-Depressed (0): {male_counts.get(0, 0)}")
print(f"Depressed (1): {male_counts.get(1, 0)}")

# Select AU columns (those ending with "_r")
au_columns = [col for col in male_data.columns if col.endswith('_r')]

# Aggregate by Participant_ID (mean per participant)
dep_agg = depressed_males.groupby('Participant_ID')[au_columns].mean()
nondep_agg = non_depressed_males.groupby('Participant_ID')[au_columns].mean()

# Add Group column
dep_agg = dep_agg.copy()
nondep_agg = nondep_agg.copy()
dep_agg['Group'] = 'Depressed Men'
nondep_agg['Group'] = 'Non-Depressed Men'

# Combine datasets
combined_data = pd.concat([dep_agg, nondep_agg])

# Reset index and melt for plotting
combined_melted = combined_data.reset_index().melt(
    id_vars=['Participant_ID', 'Group'],
    value_vars=au_columns,
    var_name='AU',
    value_name='Value'
)

# Plot colors
darker_purple = "#9B4D96"
soft_orange = "#FF8C00"

# Plot boxplot
plt.figure(figsize=(16, 6))
sns.boxplot(
    data=combined_melted,
    x='AU',
    y='Value',
    hue='Group',
    palette=[darker_purple, soft_orange],
    showfliers=False
)
plt.title('Boxplot of Action Unit Intensities: Depressed vs. Non-Depressed Men')
plt.xticks(rotation=90)
plt.tight_layout()

# Folder where you want to save the figure
output_folder = "/Users/raemarshall/Desktop/Thesis-new/MaleDepressionSplit"

# Make sure the folder exists
os.makedirs(output_folder, exist_ok=True)

# Save the figure
output_path = os.path.join(output_folder, "boxplot_depressed_vs_nondepressed_men.png")
plt.savefig(output_path, dpi=300)
print(f"Boxplot saved to {output_path}")

plt.close()
