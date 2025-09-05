import os
import pandas as pd


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
female_data = os.path.join(script_dir, "..", "WomenMenSplit", "female_participants.csv")
male_data   = os.path.join(script_dir, "..", "WomenMenSplit", "male_participants.csv")

# Resolve ".." to an absolute path
female_data = os.path.abspath(female_data)
male_data   = os.path.abspath(male_data)

# Load your gender dataset 
female_df = pd.read_csv(female_data)
male_df   = pd.read_csv(male_data)


# Ensure required columns exist
if {'PHQ8_Binary', 'Participant_ID'}.issubset(female_df.columns) and \
   {'PHQ8_Binary', 'Participant_ID'}.issubset(male_df.columns):

    # Identify unique participant IDs for *depressed* individuals (PHQ8_Binary == 1)
    depressed_female_count = female_df[female_df['PHQ8_Binary'] == 1]['Participant_ID'].nunique()
    depressed_male_count   = male_df[male_df['PHQ8_Binary'] == 1]['Participant_ID'].nunique()

    # Extract all rows associated with the identified participant IDs
    depressed_females = female_df[female_df['PHQ8_Binary'] == 1]
    depressed_males   = male_df[male_df['PHQ8_Binary'] == 1]


    # Save the split datasets
    output_females = os.path.join(script_dir, "depressed_females.csv")
    output_males   = os.path.join(script_dir, "depressed_males.csv")

    depressed_females.to_csv(output_females, index=False)
    depressed_males.to_csv(output_males, index=False)

    # Print the count of unique participants
    print(f"Depressed females: {depressed_female_count} participants")
    print(f"Depressed males: {depressed_male_count} participants")

else:
    print("Error: Required columns ('PHQ8_Binary' or 'Participant_ID') not found in one of the datasets.")
