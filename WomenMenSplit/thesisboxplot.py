import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
female_file = os.path.join(script_dir, "female_participants.csv")
male_file = os.path.join(script_dir, "male_participants.csv")

# Load your data
female_data = pd.read_csv(female_file)
male_data = pd.read_csv(male_file)

# Select AU columns (those ending with "_r")
au_columns = [col for col in female_data.columns if col.endswith('_r')]

# Grouping by Participant_ID and calculating mean
female_aggregated = female_data.groupby('Participant_ID')[au_columns].mean()
male_aggregated = male_data.groupby('Participant_ID')[au_columns].mean()

# Add a "Group" column for gender
female_aggregated['Group'] = 'Female'
male_aggregated['Group'] = 'Male'

# Combine male and female aggregated data
combined_data = pd.concat([female_aggregated, male_aggregated])

# Melt the data for seaborn plotting
combined_data_melted = combined_data.reset_index().melt(
    id_vars=['Participant_ID', 'Group'], 
    value_vars=au_columns, 
    var_name='AU', 
    value_name='Value'
)

# Custom colors
darker_purple = "#9B4D96"
soft_orange = "#FF8C00"

# Plotting
plt.figure(figsize=(16, 6))
ax = sns.boxplot(
    data=combined_data_melted, 
    x='AU', y='Value', hue='Group',
    palette=[darker_purple, soft_orange], 
    showfliers=False
)

plt.title('Boxplot of AUs by Gender')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
