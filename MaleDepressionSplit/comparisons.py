# import pandas as pd

# # Load data
# depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/MaleDepressionSplit/depressed_males.csv")
# non_depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/MaleDepressionSplit/non_depressed_males.csv")

# # Select AU columns
# au_columns = [col for col in depressed_males.columns if col.endswith('_r')]

# # Aggregate per participant (mean)
# depressed_aggregated = depressed_males.groupby('Participant_ID')[au_columns].mean()
# non_depressed_aggregated = non_depressed_males.groupby('Participant_ID')[au_columns].mean()

# # Calculate summary statistics
# depressed_mean = depressed_aggregated.mean()
# depressed_median = depressed_aggregated.median()
# depressed_range = depressed_aggregated.max() - depressed_aggregated.min()
# depressed_sd = depressed_aggregated.std()

# non_depressed_mean = non_depressed_aggregated.mean()
# non_depressed_median = non_depressed_aggregated.median()
# non_depressed_range = non_depressed_aggregated.max() - non_depressed_aggregated.min()
# non_depressed_sd = non_depressed_aggregated.std()

# # Create summary DataFrame
# summary_df = pd.DataFrame({
#     'Depressed_Male_Mean': depressed_mean,
#     'Depressed_Male_Median': depressed_median,
#     'Depressed_Male_Range': depressed_range,
#     'Depressed_Male_SD': depressed_sd,
#     'Non_Depressed_Male_Mean': non_depressed_mean,
#     'Non_Depressed_Male_Median': non_depressed_median,
#     'Non_Depressed_Male_Range': non_depressed_range,
#     'Non_Depressed_Male_SD': non_depressed_sd
# })

# # Save summary to CSV
# summary_df.to_csv('depressed_vs_nondepressed_males_summary.csv')

# # combined_melted = combined_data.reset_index().melt(
# #     id_vars=['Participant_ID', 'Group'],
# #     value_vars=au_columns,
# #     var_name='AU',
# #     value_name='Value'
# # )

# # darker_purple = "#9B4D96"
# # soft_orange = "#FF8C00"

# # plt.figure(figsize=(16, 6))
# # ax = sns.boxplot(
# #     data=combined_melted,
# #     x='AU',
# #     y='Value',
# #     hue='Group',
# #     palette=[darker_purple, soft_orange],
# #     showfliers=False
# # )
# # plt.title('Boxplot of AUs: Depressed vs. Non-Depressed Males')
# # plt.xticks(rotation=90)
# # plt.tight_layout()
# # plt.show()

import pandas as pd
from scipy.stats import mannwhitneyu
import os


# Get the dicrectory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load files
depressed_males = pd.read_csv(os.path.join(script_dir, "depressed_males.csv"))
non_depressed_males = pd.read_csv(os.path.join(script_dir, "non_depressed_males.csv"))

au_columns = [col for col in depressed_males.columns if col.endswith('_r')]

depressed_males_agg = depressed_males.groupby('Participant_ID').mean()
non_depressed_males_agg = non_depressed_males.groupby('Participant_ID').mean()

def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

p_values = []
au_names = []

for au in au_columns:
    female_data = depressed_males_agg[au]
    male_data = non_depressed_males_agg[au]

    U, p_value = mann_whitney_u_test(female_data, male_data)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values:")
print(results_df.sort_values(by='P-Value'))


