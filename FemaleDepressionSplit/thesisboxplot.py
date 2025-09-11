import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Get the dicrectory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load files
depressed_females = pd.read_csv(os.path.join(script_dir, "depressed_females.csv"))
non_depressed_females = pd.read_csv(os.path.join(script_dir, "non_depressed_females.csv"))

au_columns = [col for col in depressed_females.columns if col.endswith('_r')]

depressed_aggregated = depressed_females.groupby('Participant_ID')[au_columns].agg(['mean', 'std'])
non_depressed_aggregated = non_depressed_females.groupby('Participant_ID')[au_columns].agg(['mean', 'std'])

depressed_means = depressed_aggregated.xs('mean', axis=1, level=1)
non_depressed_means = non_depressed_aggregated.xs('mean', axis=1, level=1)

depressed_means = depressed_means.copy()
non_depressed_means = non_depressed_means.copy()

depressed_means['Group'] = 'Depressed Female'
non_depressed_means['Group'] = 'Non-Depressed Female'

combined_data = pd.concat([depressed_means, non_depressed_means])

combined_data.to_csv('depressed_vs_nondepressed_females_results.csv', index=False)

combined_melted = combined_data.reset_index().melt(
    id_vars=['Participant_ID', 'Group'],
    value_vars=au_columns,
    var_name='AU',
    value_name='Value'
)

darker_purple = "#9B4D96"
soft_orange = "#FF8C00"

plt.figure(figsize=(16, 6))
ax = sns.boxplot(
    data=combined_melted,
    x='AU',
    y='Value',
    hue='Group',
    palette=[darker_purple, soft_orange],
    showfliers=False
)
plt.title('Boxplot of AUs: Depressed vs. Non-Depressed Females')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()