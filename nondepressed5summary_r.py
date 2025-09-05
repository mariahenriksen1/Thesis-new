import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level and into the NonDepFemaleNonDepMaleSplit folder
split_dir = os.path.join(script_dir, "..", "NonDepFemaleNonDepMaleSplit")

# Load the data
females = pd.read_csv(os.path.join(split_dir, "non_depressed_females.csv"))
males = pd.read_csv(os.path.join(split_dir, "non_depressed_males.csv"))

# Get AU columns for AU_r
au_r_cols = [col for col in females.columns if col.endswith('_r')]  # Identifying AU_r columns based on the '_r' suffix

# Function to calculate 5-number summary (min, q1, median, q3, max, mean, range)
def calculate_5_number_summary(df, group_label, gender):
    # Create a list to store the summary data
    summary_data = []
    
    for au in au_r_cols:
        # Get the summary statistics for each AU column
        min_value = df[au].min()
        q1 = df[au].quantile(0.25)
        median = df[au].median()
        q3 = df[au].quantile(0.75)
        max_value = df[au].max()
        mean_value = df[au].mean()
        range_value = max_value - min_value
        
        # Append the summary statistics to the list
        summary_data.append([group_label, gender, au, min_value, q1, median, q3, max_value, mean_value, range_value])
    
    # Return a DataFrame containing the summary statistics
    return pd.DataFrame(summary_data, columns=['Group', 'Gender', 'AU', 'min', 'q1', 'median', 'q3', 'max', 'mean', 'Range'])

# Calculate the 5-number summary for Non-Depressed Females and Males
female_summary = calculate_5_number_summary(females, 'Non-Depressed', 'Female')
male_summary = calculate_5_number_summary(males, 'Non-Depressed', 'Male')

# Combine the results into one DataFrame
final_summary = pd.concat([female_summary, male_summary])

# Print the final summary
print(final_summary)

# Create box plot data from the raw data
plt.figure(figsize=(12, 6))

# Create the boxplot
sns.boxplot(x='AU', y='mean', hue='Gender', data=final_summary, 
            palette="Set2", showfliers=False)

# Adding titles and labels
plt.title("Box Plot of AU Intensities by AU and Gender (Non-Depressed)")
plt.xlabel("Action Units (AU)")
plt.ylabel("Mean Intensity")
plt.xticks(rotation=45)  # Rotate AU names for better visibility
plt.legend(title="Gender")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# Print the count of non-depressed participants by gender
print("Non-Depressed Females Count:", females['Participant_ID'].nunique())  
print("Non-Depressed Males Count:", males['Participant_ID'].nunique())
