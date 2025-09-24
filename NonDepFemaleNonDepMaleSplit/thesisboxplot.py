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

# Filter non-depressed participants
non_depressed_data = data[data['PHQ8_Binary'] == 0]

# Count unique non-depressed males and females
counts = non_depressed_data.groupby('Participant_ID')['Gender'].first().value_counts()
print("Counts of Non-Depressed Participants by Gender (by participant):")
print(f"Non-Depressed Females (0): {counts.get(0, 0)}")
print(f"Non-Depressed Males (1): {counts.get(1, 0)}")

# Select AU columns
au_columns = [col for col in non_depressed_data.columns if col.endswith('_r')]

# Split by gender
females = non_depressed_data[non_depressed_data['Gender'] == 0]
males = non_depressed_data[non_depressed_data['Gender'] == 1]

# Aggregate by Participant_ID (mean per participant)
female_agg = females.groupby('Participant_ID')[au_columns].mean()
male_agg = males.groupby('Participant_ID')[au_columns].mean()

# Add Group column
female_agg = female_agg.copy()
female_agg['Group'] = 'Non-Depressed Women'
male_agg = male_agg.copy()
male_agg['Group'] = 'Non-Depressed Men'

# Combine datasets
combined_data = pd.concat([female_agg, male_agg])

# Melt for plotting
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
plt.title('Boxplot of Action Unit Intensities: Non-Depressed Women vs Non-Depressed Men')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the figure
output_path = "/Users/raemarshall/Desktop/Thesis-new/NonDepFemaleNonDepMaleSplit/boxplot_nondepressed_men_vs_women.png"
plt.savefig(output_path, dpi=300)
print(f"Boxplot saved to {output_path}")

plt.close()
