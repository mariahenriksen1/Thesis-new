import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Folder containing all participant CSVs
folder_path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

# Read all CSV files and add Participant_ID and Gender/PHQ8_Binary
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

# Split by gender
females_non_depressed = non_depressed_data[non_depressed_data['Gender'] == 0]
males_non_depressed = non_depressed_data[non_depressed_data['Gender'] == 1]

# Count unique participants in each group
female_count = females_non_depressed['Participant_ID'].nunique()
male_count = males_non_depressed['Participant_ID'].nunique()

print(f"Non-Depressed Females (unique participants): {female_count}")
print(f"Non-Depressed Males (unique participants): {male_count}")

# Select columns ending with "_c"
au_columns_c = [col for col in non_depressed_data.columns if col.endswith('_c')]

# Replace -100 with NaN safely
females_non_depressed.loc[:, au_columns_c] = females_non_depressed.loc[:, au_columns_c].replace(-100, pd.NA)
males_non_depressed.loc[:, au_columns_c] = males_non_depressed.loc[:, au_columns_c].replace(-100, pd.NA)

# Function to compute AU presence percentage per participant
def presence_percentage_per_participant(df, group_label):
    grouped = df.groupby('Participant_ID')
    percentages = grouped[au_columns_c].apply(lambda x: (x == 1).mean() * 100)
    percentages[participant_col] = grouped.size().index
    percentages['Group'] = group_label
    cols = [participant_col] + [col for col in percentages.columns if col not in [participant_col, 'Group']] + ['Group']
    return percentages[cols].reset_index(drop=True)

participant_col = 'Participant_ID'
female_percentages = presence_percentage_per_participant(females_non_depressed, 'Women')
male_percentages = presence_percentage_per_participant(males_non_depressed, 'Men')

# Combine and reorder columns
combined_percentages = pd.concat([female_percentages, male_percentages], ignore_index=True)
cols = ['Participant_ID', 'Group'] + [col for col in combined_percentages if col not in ['Participant_ID', 'Group']]
combined_percentages = combined_percentages[cols]

# Round percentages
for au in au_columns_c:
    combined_percentages[au] = combined_percentages[au].round(2)

# Melt to long format
long_df = combined_percentages.melt(
    id_vars=['Participant_ID', 'Group'], 
    value_vars=au_columns_c, 
    var_name='AU', 
    value_name='Percentage'
)

# Compute mean presence percentage per AU per group
bar_df = long_df.groupby(['AU', 'Group'])['Percentage'].mean().reset_index()

# Sort AUs numerically
bar_df['AU'] = pd.Categorical(bar_df['AU'], categories=sorted(au_columns_c, key=lambda x: int(x[2:4])), ordered=True)

# Plot
plt.figure(figsize=(16, 6))
ax = sns.barplot(
    data=bar_df, 
    x='AU', 
    y='Percentage', 
    hue='Group', 
    palette={'Women': '#9b4d96', 'Men': '#ff8c00'}
)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

plt.title('Average Action Unit Presence Percentage: Non-Depressed Women vs Non-Depressed Men')
plt.ylabel('Presence Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
