import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv")
males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv")

# Get the number of frames for each participant
female_lengths = females.groupby('Participant_ID')['frame'].nunique().reset_index(name='Total_Frames')
male_lengths = males.groupby('Participant_ID')['frame'].nunique().reset_index(name='Total_Frames')

# Get AU columns
au_c_cols = [col for col in females.columns if col.endswith('_c')]

# Function to compute flips and normalize them
def compute_flips_and_normalize(df, lengths_df, group_label, gender):
    df = df.merge(lengths_df[['Participant_ID', 'Total_Frames']], on='Participant_ID', how='left')
    
    df = df.sort_values(by=["Participant_ID", "frame"])
    flips = []

    for pid, group in df.groupby('Participant_ID'):
        group_flips = {}
        for col in au_c_cols:
            values = group[col].values
            flip_count = (values[1:] != values[:-1]).sum()
            norm_flips = (flip_count / group['Total_Frames'].iloc[0]) * 1000
            group_flips[col] = norm_flips

        group_flips['Participant_ID'] = pid
        group_flips['Group'] = group_label
        group_flips['Gender'] = gender  # Add gender directly here
        flips.append(group_flips)

    return pd.DataFrame(flips)

# Compute flips for both females and males
female_flips = compute_flips_and_normalize(females, female_lengths, "Depressed", "Female")
male_flips = compute_flips_and_normalize(males, male_lengths, "Depressed", "Male")

# Combine both into one DataFrame
all_flips = pd.concat([female_flips, male_flips])

# Filter out just the Depressed data
depressed_flips = all_flips[all_flips['Group'] == "Depressed"]

# Melt the data for easier analysis
depressed_melted = depressed_flips.melt(
    id_vars=["Participant_ID", "Group", "Gender"],
    var_name="AU",
    value_name="Normalized_Flips"
)

# Function to calculate 5-number summary + mean + range
summary_stats = depressed_melted.groupby(['Gender', 'AU'])['Normalized_Flips'].agg(
    min='min',
    q1=lambda x: x.quantile(0.25),
    median='median',
    q3=lambda x: x.quantile(0.75),
    max='max',
    mean='mean'
).reset_index()

# Calculate the range (max - min) and add to the summary statistics
summary_stats['Range'] = summary_stats['max'] - summary_stats['min']

#Print the summary statistics
print(summary_stats)

# Create box plot data from the raw data 'depressed_melted'
plt.figure(figsize=(12, 6))

# Create the boxplot
sns.boxplot(x='AU', y='Normalized_Flips', hue='Gender', data=depressed_melted, 
            palette="Set2", showfliers=False)

# Adding titles and labels
plt.title("Box Plot of Normalized Flips by AU and Gender")
plt.xlabel("Action Units (AU)")
plt.ylabel("Normalized Flip Count")
plt.xticks(rotation=45)  # Rotate AU names for better visibility
plt.legend(title="Gender")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

print("Depressed Females Count:", females['Participant_ID'].nunique())  
print("Depressed Males Count:", males['Participant_ID'].nunique()) 

