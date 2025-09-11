import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
non_depressed_females = pd.read_csv(os.path.join(script_dir, "non_depressed_females.csv"))
non_depressed_males = pd.read_csv(os.path.join(script_dir, "non_depressed_males.csv"))

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

plt.title('AU Intensities Box Plot for Non-Depressed Females vs. Non-Depressed Males')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
