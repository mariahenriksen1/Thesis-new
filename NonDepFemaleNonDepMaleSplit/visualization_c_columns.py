import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
females_non_depressed = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
males_non_depressed = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

participant_col = 'Participant_ID'
au_columns_c = [col for col in females_non_depressed.columns if col.endswith('_c')]

females_non_depressed[au_columns_c] = females_non_depressed[au_columns_c].replace(-100, pd.NA)
males_non_depressed[au_columns_c] = males_non_depressed[au_columns_c].replace(-100, pd.NA)

def presence_percentage_per_participant(df, group_label):
    grouped = df.groupby(participant_col)
    percentages = grouped[au_columns_c].mean() * 100
    percentages[participant_col] = grouped.size().index
    percentages['Group'] = group_label
    cols = [participant_col] + [col for col in percentages.columns if col not in [participant_col, 'Group']] + ['Group']
    return percentages[cols].reset_index(drop=True)

female_percentages = presence_percentage_per_participant(females_non_depressed, 'Female')
male_percentages = presence_percentage_per_participant(males_non_depressed, 'Male')

combined_percentages = pd.concat([female_percentages, male_percentages], ignore_index=True)

for au in au_columns_c:
    combined_percentages[au] = combined_percentages[au].round(2)

grouped_avg = combined_percentages.groupby('Group')[au_columns_c].mean().reset_index()

melted = grouped_avg.melt(id_vars='Group', var_name='Action Unit', value_name='Mean Presence (%)')

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=melted, x='Action Unit', y='Mean Presence (%)', hue='Group', palette=['#1f77b4', '#ff7f0e'])

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=9, padding=3)

plt.title('Mean Presence of Binary Facial Action Units by Gender (Non-Depressed Participants)')
plt.ylabel('Mean % Presence')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


