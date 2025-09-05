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


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# depressed_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepressedNondepressedSplit/depressed_participants.csv")
# non_depressed_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepressedNondepressedSplit/non_depressed_participants.csv")

# au_columns = [col for col in depressed_data.columns if col.endswith('_r')]

# depressed_aggregated = depressed_data.groupby('Participant_ID')[au_columns].mean()
# non_depressed_aggregated = non_depressed_data.groupby('Participant_ID')[au_columns].mean()

# depressed_mean = depressed_aggregated.mean()
# depressed_median = depressed_aggregated.median()
# depressed_range = depressed_aggregated.max() - depressed_aggregated.min()
# depressed_sd = depressed_aggregated.std()

# non_depressed_mean = non_depressed_aggregated.mean()
# non_depressed_median = non_depressed_aggregated.median()
# non_depressed_range = non_depressed_aggregated.max() - non_depressed_aggregated.min()
# non_depressed_sd = non_depressed_aggregated.std()

# summary_df = pd.DataFrame({
#     'Depressed_Mean': depressed_mean,
#     'Depressed_Median': depressed_median,
#     'Depressed_Range': depressed_range,
#     'Depressed_SD': depressed_sd,
#     'Non_Depressed_Mean': non_depressed_mean,
#     'Non_Depressed_Median': non_depressed_median,
#     'Non_Depressed_Range': non_depressed_range,
#     'Non_Depressed_SD': non_depressed_sd
# })

# depressed_aggregated['Group'] = 'Depressed'
# non_depressed_aggregated['Group'] = 'Non-Depressed'

# combined_data = pd.concat([depressed_aggregated, non_depressed_aggregated])

# summary_df.to_csv('depressed_vs_nondepressed_results.csv', index=False)

# # combined_data_melted = combined_data.reset_index().melt(id_vars=['Participant_ID', 'Group'], value_vars=au_columns, var_name='AU', value_name='Value')

# # blue = "#1f77b4"  
# # yellow = "#ffdd30"  

# # plt.figure(figsize=(16, 6))
# # ax = sns.boxplot(data=combined_data_melted, x='AU', y='Value', hue='Group', 
# #                  palette=[blue, yellow], showfliers=False)

# # plt.title('Boxplot of AUs by Depression Status')
# # plt.xticks(rotation=90)
# # plt.tight_layout()
# # plt.show()

# import pandas as pd
# from scipy.stats import mannwhitneyu


# depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/FemaleDepressionSplit/depressed_females.csv")
# non_depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/FemaleDepressionSplit/non_depressed_females.csv")

# au_columns = [col for col in depressed_females.columns if col.endswith('_r')]

# depressed_females_agg = depressed_females.groupby('Participant_ID').mean()
# non_depressed_females_agg = non_depressed_females.groupby('Participant_ID').mean()

# def mann_whitney_u_test(x, y):
#     U, p_value = mannwhitneyu(x, y, alternative='two-sided')
#     return U, p_value

# p_values = []
# au_names = []

# for au in au_columns:
#     female_data = depressed_females_agg[au]
#     male_data = non_depressed_females_agg[au]

#     U, p_value = mann_whitney_u_test(female_data, male_data)
    
#     p_values.append(round(p_value, 5))
#     au_names.append(au)

# results_df = pd.DataFrame({
#     'AU': au_names,
#     'P-Value': p_values
# })

# print("All AUs with their P-Values:")
# print(results_df.sort_values(by='P-Value'))
