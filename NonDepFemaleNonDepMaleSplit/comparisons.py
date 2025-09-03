import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

non_depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
non_depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

au_columns = [col for col in non_depressed_females.columns if col.endswith('_r')]

female_min_max = pd.DataFrame({
    'Female_Min': non_depressed_females[au_columns].min(),
    'Female_Max': non_depressed_females[au_columns].max()
})

male_min_max = pd.DataFrame({
    'Male_Min': non_depressed_males[au_columns].min(),
    'Male_Max': non_depressed_males[au_columns].max()
})

female_means = pd.DataFrame({
    'Female_Mean': non_depressed_females[au_columns].mean()
})

male_means = pd.DataFrame({
    'Male_Mean': non_depressed_males[au_columns].mean()
})

# Compute medians for each AU column
female_medians = pd.DataFrame({
    'Female_Median': non_depressed_females[au_columns].median()
})

male_medians = pd.DataFrame({
    'Male_Median': non_depressed_males[au_columns].median()
})

# Combine with previous stats
au_stats_comparison = pd.concat([female_min_max, male_min_max, female_means, male_means, female_medians, male_medians], axis=1)

female_means = non_depressed_females.groupby('Participant_ID')[au_columns].mean()
female_means['Gender'] = 'Female'
male_means = non_depressed_males.groupby('Participant_ID')[au_columns].mean()
male_means['Gender'] = 'Male'

combined_df = pd.concat([female_means, male_means], axis=0).reset_index()

long_df = combined_df.melt(id_vars=['Participant_ID', 'Gender'], 
                            value_vars=au_columns, 
                            var_name='AU', 
                            value_name='Mean_Intensity')

darker_purple = "#9B4D96"  
soft_orange = "#FF8C00"  

plt.figure(figsize=(16, 6))
ax = sns.boxplot(data=long_df, x='AU', y='Mean_Intensity', hue='Gender', 
                 palette=[darker_purple, soft_orange], showfliers=False)

plt.title('AU Intensities Box Plot for Non-Depressed Females vs. Males')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # # Combine them together
# # df_comparison = pd.DataFrame({
# #     'Non_Depressed_Female_Mean': female_mean,
# #     'Non_Depressed_Female_Std': female_std,
# #     'Non_Depressed_Female_Median': female_median,
# #     'Non_Depressed_Male_Mean': male_mean,
# #     'Non_Depressed_Male_Std': male_std,
# #     'Non_Depressed_Male_Median': male_median
# # })

# # print(df_comparison)

# # non_depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
# # non_depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

# # au_columns = [col for col in non_depressed_females.columns if col.endswith('_r')]

# # def compute_stats(df, label):
# #     stats = pd.DataFrame({
# #         'Mean': df[au_columns].mean(),
# #         'Std': df[au_columns].std(),
# #         'Median': df[au_columns].median(),
# #         'Range': df[au_columns].max() - df[au_columns].min()
# #     })
# #     stats['Group'] = label
# #     stats.index.name = 'AU'
# #     return stats.reset_index()

# # female_stats = compute_stats(non_depressed_females, 'Non-Depressed Female')
# # male_stats = compute_stats(non_depressed_males, 'Non-Depressed Male')

# # combined_stats = pd.concat([female_stats, male_stats])
# # combined_stats.to_csv("non_depressed_gender_stats.csv", index=False)

# # print("Descriptive statistics saved to 'non_depressed_gender_stats.csv'")

# # # Load your datasets
# # non_depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
# # non_depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

# # # Get list of AU columns
# # au_columns = [col for col in non_depressed_females.columns if col.endswith('_r')]

# # # Function to compute zero proportions
# # def zero_proportions(df):
# #     return (df[au_columns] == 0).sum() / len(df)

# # # Calculate zero proportions for each group
# # female_zero_props = zero_proportions(non_depressed_females)
# # male_zero_props = zero_proportions(non_depressed_males)

# # # Combine into one DataFrame side-by-side
# # zero_comparison = pd.DataFrame({
# #     'Female_Zero_Proportion': female_zero_props,
# #     'Male_Zero_Proportion': male_zero_props
# # })

# # # Show the result
# # print(zero_comparison)

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load your data
# non_depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
# non_depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

# # Select AU columns
# au_columns = [col for col in non_depressed_females.columns if col.endswith('_r')]

# # Compute activation rates (proportion of frames where AU > 0)
# female_activation = (non_depressed_females[au_columns] > 0).sum() / len(non_depressed_females)
# male_activation = (non_depressed_males[au_columns] > 0).sum() / len(non_depressed_males)

# # Combine into DataFrame
# activation_df = pd.DataFrame({
#     'Female_Activation': female_activation,
#     'Male_Activation': male_activation
# }).reset_index().rename(columns={'index': 'AU'})

# # Plot
# plt.figure(figsize=(14, 6))
# bar_width = 0.35
# x = range(len(activation_df))

# plt.bar([i - bar_width/2 for i in x], activation_df['Female_Activation'], 
#         width=bar_width, label='Female', color='#9B4D96')
# plt.bar([i + bar_width/2 for i in x], activation_df['Male_Activation'], 
#         width=bar_width, label='Male', color='#FF8C00')

# plt.xticks(ticks=x, labels=activation_df['AU'], rotation=45)
# plt.ylabel("Activation Rate (Proportion of Frames > 0)")
# plt.title("AU Activation Rates: Non-Depressed Females vs. Males")
# plt.legend()
# plt.tight_layout()
# plt.show()
