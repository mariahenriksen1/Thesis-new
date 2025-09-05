import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
females_depressed = pd.read_csv(os.path.join(script_dir, "depressed_females.csv"))
males_depressed = pd.read_csv(os.path.join(script_dir, "depressed_males.csv"))

#females_depressed = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv")
#males_depressed = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv")

participant_col = 'Participant_ID'
au_columns_c = [col for col in females_depressed.columns if col.endswith('_c')]

females_depressed[au_columns_c] = females_depressed[au_columns_c].apply(pd.to_numeric, errors='coerce')
males_depressed[au_columns_c] = males_depressed[au_columns_c].apply(pd.to_numeric, errors='coerce')


def presence_percentage_per_participant(df, group_label):
    grouped = df.groupby(participant_col)
    percentages = grouped[au_columns_c].mean() * 100
    percentages[participant_col] = grouped.size().index
    percentages['Group'] = group_label
    cols = [participant_col] + [col for col in percentages.columns if col not in [participant_col, 'Group']] + ['Group']
    return percentages[cols].reset_index(drop=True)

female_percentages = presence_percentage_per_participant(females_depressed, 'Female')
male_percentages = presence_percentage_per_participant(males_depressed, 'Male')

combined_percentages = pd.concat([female_percentages, male_percentages], ignore_index=True)

for au in au_columns_c:
    combined_percentages[au] = combined_percentages[au].round(2)

grouped_avg = combined_percentages.groupby('Group')[au_columns_c].mean().reset_index()

melted = grouped_avg.melt(id_vars='Group', var_name='Action Unit', value_name='Mean Presence (%)')

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=melted, x='Action Unit', y='Mean Presence (%)', hue='Group', palette=['#1f77b4', '#ff7f0e'])

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=9, padding=3)

plt.title('Mean Presence of Binary Facial Action Units by Gender (Depressed Participants)')
plt.ylabel('Mean % Presence')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


