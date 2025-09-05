import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the WomenMenSplit folder (input)
women_men_split_dir = os.path.abspath(os.path.join(script_dir, "..", "WomenMenSplit"))

# Path to the NonDepFemaleNonDepMaleSplit folder (output)
non_dep_split_dir = os.path.abspath(os.path.join(script_dir, "..", "NonDepFemaleNonDepMaleSplit"))

# Load input data
female_data = pd.read_csv(os.path.join(women_men_split_dir, "female_participants.csv"))
male_data = pd.read_csv(os.path.join(women_men_split_dir, "male_participants.csv"))

# Check required columns
if all(col in female_data.columns for col in ['PHQ8_Binary', 'Participant_ID']) and \
   all(col in male_data.columns for col in ['PHQ8_Binary', 'Participant_ID']):

    # Identify unique participant IDs for non-depressed individuals (PHQ8_Binary == 0)
    non_depressed_female_count = female_data[female_data['PHQ8_Binary'] == 0]['Participant_ID'].nunique()
    non_depressed_male_count = male_data[male_data['PHQ8_Binary'] == 0]['Participant_ID'].nunique()

    # Extract all rows associated with the identified participant IDs
    non_depressed_females = female_data[female_data['PHQ8_Binary'] == 0]
    non_depressed_males = male_data[male_data['PHQ8_Binary'] == 0]

    # Save the split datasets dynamically
    non_depressed_females.to_csv(os.path.join(non_dep_split_dir, "non_depressed_females.csv"), index=False)
    non_depressed_males.to_csv(os.path.join(non_dep_split_dir, "non_depressed_males.csv"), index=False)

    print(f"Non-depressed females: {non_depressed_female_count} participants")
    print(f"Non-depressed males: {non_depressed_male_count} participants")

else:
    print("Error: Required columns ('PHQ8_Binary' or 'Participant_ID') not found in one of the datasets.")
