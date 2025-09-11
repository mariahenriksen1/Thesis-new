import os
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
female_file = os.path.join(script_dir, "female_participants.csv")
male_file = os.path.join(script_dir, "male_participants.csv")

# Load your data
female_data = pd.read_csv(female_file)
male_data = pd.read_csv(male_file)

au_columns = [col for col in female_data.columns if col.endswith('_r')]

# Grouping by Participant_ID and calculating mean
female_aggregated = female_data.groupby('Participant_ID')[au_columns].mean()
male_aggregated = male_data.groupby('Participant_ID')[au_columns].mean()

# Calculate statistics for females
female_mean = female_aggregated.mean()
female_median = female_aggregated.median()
female_range = female_aggregated.max() - female_aggregated.min()
female_sd = female_aggregated.std()

# Calculate statistics for males
male_mean = male_aggregated.mean()
male_median = male_aggregated.median()
male_range = male_aggregated.max() - male_aggregated.min()
male_sd = male_aggregated.std()

# Creating a summary dataframe to display all the statistics
summary_df = pd.DataFrame({
    'Female_Mean': female_mean,
    'Female_Median': female_median,
    'Female_Range': female_range,
    'Female_SD': female_sd,
    'Male_Mean': male_mean,
    'Male_Median': male_median,
    'Male_Range': male_range,
    'Male_SD': male_sd
})

# Printing the summary statistics
print(summary_df)

# combined_data_melted = combined_data.reset_index().melt(id_vars=['Participant_ID', 'Group'], value_vars=au_columns, var_name='AU', value_name='Value')

# darker_purple = "#9B4D96"
# soft_orange = "#FF8C00"

# plt.figure(figsize=(16, 6))
# ax = sns.boxplot(data=combined_data_melted, x='AU', y='Value', hue='Group', 
#                  palette=[darker_purple, soft_orange], showfliers=False)
# plt.title('Boxplot of AUs by Gender')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()

# import pandas as pd
# from scipy.stats import mannwhitneyu

# # Assuming your data is already loaded in female_data and male_data
# female_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/female_participants.csv")
# male_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/male_participants.csv")

# au_columns = [col for col in female_data.columns if col.endswith('_r')]

# females_agg = female_data.groupby('Participant_ID').mean()
# males_agg = male_data.groupby('Participant_ID').mean()

# def mann_whitney_u_test(x, y):
#     U, p_value = mannwhitneyu(x, y, alternative='two-sided')
#     return U, p_value

# p_values = []
# au_names = []

# for au in au_columns:
#     female_data = females_agg[au]
#     male_data = males_agg[au]

#     U, p_value = mann_whitney_u_test(female_data, male_data)
    
#     p_values.append(round(p_value, 5))
#     au_names.append(au)

# results_df = pd.DataFrame({
#     'AU': au_names,
#     'P-Value': p_values
# })

# print("All AUs with their P-Values:")
# print(results_df.sort_values(by='P-Value'))
