import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load files
depressed_males = pd.read_csv(os.path.join(script_dir, "depressed_males.csv"))
non_depressed_males = pd.read_csv(os.path.join(script_dir, "non_depressed_males.csv"))

# Select AU columns (those ending with "_r")
au_columns = [col for col in depressed_males.columns if col.endswith('_r')]

# Grouping by Participant_ID and calculating mean and std
depressed_aggregated = depressed_males.groupby('Participant_ID')[au_columns].agg(['mean', 'std'])
non_depressed_aggregated = non_depressed_males.groupby('Participant_ID')[au_columns].agg(['mean', 'std'])

# Extract only the mean values
depressed_means = depressed_aggregated.xs('mean', axis=1, level=1)
non_depressed_means = non_depressed_aggregated.xs('mean', axis=1, level=1)

# Add Group column
depressed_means = depressed_means.copy()
non_depressed_means = non_depressed_means.copy()

depressed_means['Group'] = 'Depressed Male'
non_depressed_means['Group'] = 'Non-Depressed Male'

# Combine datasets
combined_data = pd.concat([depressed_means, non_depressed_means])

# Save combined data to CSV
combined_data.to_csv('depressed_vs_nondepressed_males_results.csv', index=False)

# Melt the data for plotting
combined_melted = combined_data.reset_index().melt(
    id_vars=['Participant_ID', 'Group'],
    value_vars=au_columns,
    var_name='AU',
    value_name='Value'
)

darker_purple = "#9B4D96"
soft_orange = "#FF8C00"

# Plotting
plt.figure(figsize=(16, 6))
ax = sns.boxplot(
    data=combined_melted,
    x='AU',
    y='Value',
    hue='Group',
    palette=[darker_purple, soft_orange],
    showfliers=False
)
plt.title('Boxplot of AUs: Depressed vs. Non-Depressed Males')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
