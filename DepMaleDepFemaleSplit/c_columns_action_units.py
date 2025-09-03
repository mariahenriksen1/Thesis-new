import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load your data
females_depressed = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv")
males_depressed = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv")

# Define columns for binary AU data (assuming you have binary columns for AU presence)
participant_col = 'Participant_ID'
au_columns_c = [col for col in females_depressed.columns if col.endswith('_c')]  # or adjust this if needed

# Replace -100 with NaN
females_depressed[au_columns_c] = females_depressed[au_columns_c].replace(-100, pd.NA)
males_depressed[au_columns_c] = males_depressed[au_columns_c].replace(-100, pd.NA)

# Convert to binary (1 = presence, 0 = absence)
females_depressed[au_columns_c] = females_depressed[au_columns_c].applymap(lambda x: 1 if x == 1 else 0)
males_depressed[au_columns_c] = males_depressed[au_columns_c].applymap(lambda x: 1 if x == 1 else 0)

# Calculate presence (1) vs absence (0) for each AU by group
def presence_counts(df, group_label):
    counts = df[au_columns_c].sum(axis=0).reset_index()
    counts.columns = ['AU', 'Present']
    counts['Absent'] = len(df) - counts['Present']
    counts['Group'] = group_label
    return counts

# Calculate for both groups
female_counts = presence_counts(females_depressed, 'Female')
male_counts = presence_counts(males_depressed, 'Male')

# Combine the data
combined_counts = pd.concat([female_counts, male_counts], ignore_index=True)

# Set up the plot
plt.figure(figsize=(16, 8))

# Define colors (darker for presence, lighter for absence)
palette = {'Female': '#9b4d96', 'Male': '#ff8c00'}
light_palette = {'Female': mcolors.to_rgba('#9b4d96', alpha=0.3), 'Male': mcolors.to_rgba('#ff8c00', alpha=0.3)}

# Create a stacked bar plot for each AU
for idx, au in enumerate(au_columns_c):
    # Get the counts for the current AU
    female_au = female_counts[female_counts['AU'] == au].iloc[0]
    male_au = male_counts[male_counts['AU'] == au].iloc[0]
    
    # Plotting presence (darker)
    plt.bar(au, female_au['Present'], color=palette['Female'], label='Female Present' if idx == 0 else "", zorder=10)
    plt.bar(au, male_au['Present'], color=palette['Male'], label='Male Present' if idx == 0 else "", zorder=10)
    
    # Plotting absence (lighter)
    plt.bar(au, female_au['Absent'], bottom=female_au['Present'], color=light_palette['Female'], label='Female Absent' if idx == 0 else "", zorder=5)
    plt.bar(au, male_au['Absent'], bottom=male_au['Present'], color=light_palette['Male'], label='Male Absent' if idx == 0 else "", zorder=5)

# Aesthetics
plt.title('Stacked Bar Plot of AU Presence vs. Absence for Depressed Females and Males')
plt.ylabel('Count')
plt.xlabel('Action Units')
plt.xticks(rotation=90)
plt.legend(title='Group')
plt.tight_layout()
plt.show()
