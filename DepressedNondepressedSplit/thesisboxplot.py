import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
depressed_file = os.path.join(script_dir, "depressed_participants.csv")
nondepressed_file = os.path.join(script_dir, "non_depressed_participants.csv")

# Load your data
depressed_data = pd.read_csv(depressed_file)
nondepressed_data = pd.read_csv(nondepressed_file)

# Select AU columns (those ending with "_r")
au_columns = [col for col in depressed_data.columns if col.endswith('_r')]

# Grouping by Participant_ID and calculating mean
depressed_aggregated = depressed_data.groupby('Participant_ID')[au_columns].mean()
nondepressed_aggregated = nondepressed_data.groupby('Participant_ID')[au_columns].mean()

# Add a "Group" column for depression status
depressed_aggregated['Group'] = 'Depressed'
nondepressed_aggregated['Group'] = 'Non-Depressed'

# Combine datasets
combined_data = pd.concat([depressed_aggregated, nondepressed_aggregated])

# Melt the data for seaborn plotting
combined_data_melted = combined_data.reset_index().melt(
    id_vars=['Participant_ID', 'Group'], 
    value_vars=au_columns, 
    var_name='AU', 
    value_name='Value'
)

# Custom colors: Blue for Non-Depressed, Yellow for Depressed
custom_palette = {"Non-Depressed": "#1f77b4", "Depressed": "#FFD700"}

# Plotting
plt.figure(figsize=(16, 6))
ax = sns.boxplot(
    data=combined_data_melted, 
    x='AU', y='Value', hue='Group',
    palette=custom_palette, 
    showfliers=False
)

plt.title('Boxplot of AUs by Depression Status')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
