import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets for depressed males and females
depressed_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv")
depressed_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv")

# Specify the facial action units columns
au_columns = [col for col in depressed_females.columns if col.endswith('_r')]

# Compute the minimum and maximum values for both depressed females and males
female_min_max = pd.DataFrame({
    'Female_Min': depressed_females[au_columns].min(),
    'Female_Max': depressed_females[au_columns].max()
})

male_min_max = pd.DataFrame({
    'Male_Min': depressed_males[au_columns].min(),
    'Male_Max': depressed_males[au_columns].max()
})

# Compute the means for both depressed females and males
female_means = pd.DataFrame({
    'Female_Mean': depressed_females[au_columns].mean()
})

male_means = pd.DataFrame({
    'Male_Mean': depressed_males[au_columns].mean()
})

# Compute the medians for each AU column
female_medians = pd.DataFrame({
    'Female_Median': depressed_females[au_columns].median()
})

male_medians = pd.DataFrame({
    'Male_Median': depressed_males[au_columns].median()
})

# Combine the statistics into one DataFrame for comparison
au_stats_comparison = pd.concat([female_min_max, male_min_max, female_means, male_means, female_medians, male_medians], axis=1)

# Group by participant ID and compute the mean AU values for each participant
female_means = depressed_females.groupby('Participant_ID')[au_columns].mean()
female_means['Gender'] = 'Female'
male_means = depressed_males.groupby('Participant_ID')[au_columns].mean()
male_means['Gender'] = 'Male'

# Concatenate the dataframes for females and males
combined_df = pd.concat([female_means, male_means], axis=0).reset_index()

# Reshape the data to long format
long_df = combined_df.melt(id_vars=['Participant_ID', 'Gender'], 
                            value_vars=au_columns, 
                            var_name='AU', 
                            value_name='Mean_Intensity')

# Define the color palette
darker_purple = "#9B4D96"  
soft_orange = "#FF8C00"  

# Create the box plot for AU intensities
plt.figure(figsize=(16, 6))
ax = sns.boxplot(data=long_df, x='AU', y='Mean_Intensity', hue='Gender', 
                 palette=[darker_purple, soft_orange], showfliers=False)

# Title and adjustments for the plot
plt.title('AU Intensities Box Plot for Depressed Females vs. Depressed Males')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
