import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to input and output folders
women_men_split_dir = os.path.abspath(os.path.join(script_dir, "..", "WomenMenSplit"))
female_split_dir = os.path.abspath(os.path.join(script_dir, "..", "FemaleDepressionSplit"))


# Load female participants data
female_data_file = os.path.join(women_men_split_dir, "female_participants.csv")
female_data = pd.read_csv(female_data_file)

# Check required columns and split
if 'PHQ8_Binary' in female_data.columns and 'Participant_ID' in female_data.columns:
    depressed_ids = female_data[female_data['PHQ8_Binary'] == 1]['Participant_ID'].unique()
    non_depressed_ids = female_data[female_data['PHQ8_Binary'] == 0]['Participant_ID'].unique()

    # Filter original data to include all rows for those participants
    depressed_females = female_data[female_data['Participant_ID'].isin(depressed_ids)]
    non_depressed_females = female_data[female_data['Participant_ID'].isin(non_depressed_ids)]

    # Save to CSV in FemaleDepressionSplit folder
    depressed_females_file = os.path.join(female_split_dir, "depressed_females.csv")
    non_depressed_females_file = os.path.join(female_split_dir, "non_depressed_females.csv")

    depressed_females.to_csv(depressed_females_file, index=False)
    non_depressed_females.to_csv(non_depressed_females_file, index=False)

    # Print counts
    print(f"Depressed females: {len(depressed_ids)} unique participants")
    print(f"Non-depressed females: {len(non_depressed_ids)} unique participants")
else:
    print("Error: 'PHQ8_Binary' or 'Participant_ID' column not found in the dataset.")
