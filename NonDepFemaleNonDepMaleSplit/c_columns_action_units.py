import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
females_non_depressed = pd.read_csv("/Users/raemarshall/Desktop/initial Thesis Code/Thesis/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
males_non_depressed = pd.read_csv("/Users/raemarshall/Desktop/initial Thesis Code/Thesis/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

# Define columns
participant_col = 'Participant_ID'
au_columns_c = [col for col in females_non_depressed.columns if col.endswith('_c')]

# Replace -100 with NaN
females_non_depressed[au_columns_c] = females_non_depressed[au_columns_c].replace(-100, pd.NA)
males_non_depressed[au_columns_c] = males_non_depressed[au_columns_c].replace(-100, pd.NA)

def presence_percentage_per_participant(df, group_label):
    grouped = df.groupby(participant_col)
    percentages = grouped[au_columns_c].apply(lambda x: (x == 1).mean() * 100)
    percentages[participant_col] = grouped.size().index
    percentages['Group'] = group_label
    cols = [participant_col] + [col for col in percentages.columns if col not in [participant_col, 'Group']] + ['Group']
    return percentages[cols].reset_index(drop=True)

# Calculate percentages
female_percentages = presence_percentage_per_participant(females_non_depressed, 'Female')
male_percentages = presence_percentage_per_participant(males_non_depressed, 'Male')

# Combine
combined_percentages = pd.concat([female_percentages, male_percentages], ignore_index=True)

# Correct column order
cols = ['Participant_ID', 'Group'] + [col for col in combined_percentages if col not in ['Participant_ID', 'Group']]
combined_percentages = combined_percentages[cols]

# Round
for au in au_columns_c:
    combined_percentages[au] = combined_percentages[au].round(2)

# Melt to long format
long_df = combined_percentages.melt(id_vars=['Participant_ID', 'Group'], 
                                    value_vars=au_columns_c, 
                                    var_name='AU', value_name='Percentage')

# Compute mean presence percentage per AU per group
bar_df = long_df.groupby(['AU', 'Group'])['Percentage'].mean().reset_index()

# Sort AUs numerically
bar_df['AU'] = pd.Categorical(bar_df['AU'], categories=sorted(au_columns_c, key=lambda x: int(x[2:4])), ordered=True)

# Set plot size
plt.figure(figsize=(16, 6))

# Bar plot
ax = sns.barplot(data=bar_df, x='AU', y='Percentage', hue='Group', palette={'Female': '#9b4d96', 'Male': '#ff8c00'})

# Add percentage on top of the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

# Aesthetics
plt.title('Average AU Presence Percentage for Non-Depressed Females vs. Males')
plt.ylabel('Presence Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
