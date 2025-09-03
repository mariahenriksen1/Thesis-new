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

import pandas as pd
from scipy.stats import mannwhitneyu

depressed_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepressedNondepressedSplit/depressed_participants.csv")
non_depressed_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepressedNondepressedSplit/non_depressed_participants.csv")

au_columns = [col for col in depressed_data.columns if col.endswith('_r')]

depressed_agg = depressed_data.groupby('Participant_ID').mean()
non_depressed_agg = non_depressed_data.groupby('Participant_ID').mean()

def mann_whitney_u_test(x, y):
    U, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return U, p_value

p_values = []
au_names = []

for au in au_columns:
    female_data = depressed_agg[au]
    male_data = non_depressed_agg[au]

    U, p_value = mann_whitney_u_test(female_data, male_data)
    
    p_values.append(round(p_value, 5))
    au_names.append(au)

results_df = pd.DataFrame({
    'AU': au_names,
    'P-Value': p_values
})

print("All AUs with their P-Values:")
print(results_df.sort_values(by='P-Value'))
