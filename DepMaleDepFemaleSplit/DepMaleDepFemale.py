import pandas as pd

# Load gender-separated datasets
female_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/female_participants.csv")
male_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/male_participants.csv")

# Ensure required columns exist
if 'PHQ8_Binary' in female_data.columns and 'PHQ8_Binary' in male_data.columns and 'Participant_ID' in female_data.columns and 'Participant_ID' in male_data.columns:

    # Identify unique participant IDs for *depressed* individuals (PHQ8_Binary == 1)
    depressed_female_count = female_data[female_data['PHQ8_Binary'] == 1]['Participant_ID'].nunique()
    depressed_male_count = male_data[male_data['PHQ8_Binary'] == 1]['Participant_ID'].nunique()

    # Extract all rows associated with the identified participant IDs
    depressed_females = female_data[female_data['PHQ8_Binary'] == 1]
    depressed_males = male_data[male_data['PHQ8_Binary'] == 1]

    # Save the split datasets
    depressed_females.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv", index=False)
    depressed_males.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv", index=False)

    # Print the count of unique participants
    print(f"Depressed females: {depressed_female_count} participants")
    print(f"Depressed males: {depressed_male_count} participants")

else:
    print("Error: Required columns ('PHQ8_Binary' or 'Participant_ID') not found in one of the datasets.")
