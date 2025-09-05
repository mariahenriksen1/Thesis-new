import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to input and output folders
women_men_split_dir = os.path.abspath(os.path.join(script_dir, "..", "WomenMenSplit"))
male_split_dir = os.path.abspath(os.path.join(script_dir, "..", "MaleDepressionSplit"))

# Load male participants data
male_data_file = os.path.join(women_men_split_dir, "male_participants.csv")
male_data = pd.read_csv(male_data_file)

# Check required columns and split
if 'PHQ8_Binary' in male_data.columns and 'Participant_ID' in male_data.columns:
    depressed_ids = male_data[male_data['PHQ8_Binary'] == 1]['Participant_ID'].unique()
    non_depressed_ids = male_data[male_data['PHQ8_Binary'] == 0]['Participant_ID'].unique()

    depressed_males = male_data[male_data['Participant_ID'].isin(depressed_ids)]
    non_depressed_males = male_data[male_data['Participant_ID'].isin(non_depressed_ids)]

    # Save to CSV in MaleDepressionSplit folder
    depressed_males_file = os.path.join(male_split_dir, "depressed_males.csv")
    non_depressed_males_file = os.path.join(male_split_dir, "non_depressed_males.csv")

    depressed_males.to_csv(depressed_males_file, index=False)
    non_depressed_males.to_csv(non_depressed_males_file, index=False)

    # Print counts
    print(f"Depressed males: {len(depressed_ids)} unique participants")
    print(f"Non-depressed males: {len(non_depressed_ids)} unique participants")
else:
    print("Error: 'PHQ8_Binary' or 'Participant_ID' column not found in the dataset.")
