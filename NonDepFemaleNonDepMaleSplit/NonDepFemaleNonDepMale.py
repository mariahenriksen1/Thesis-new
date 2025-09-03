import pandas as pd

female_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/female_participants.csv")
male_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/male_participants.csv")

if 'PHQ8_Binary' in female_data.columns and 'PHQ8_Binary' in male_data.columns and 'Participant_ID' in female_data.columns and 'Participant_ID' in male_data.columns:

    # Identify unique participant IDs for non-depressed individuals (PHQ8_Binary == 0)
    non_depressed_female_count = female_data[female_data['PHQ8_Binary'] == 0]['Participant_ID'].nunique()
    non_depressed_male_count = male_data[male_data['PHQ8_Binary'] == 0]['Participant_ID'].nunique()

    # Extract all rows associated with the identified participant IDs
    non_depressed_females = female_data[female_data['PHQ8_Binary'] == 0]
    non_depressed_males = male_data[male_data['PHQ8_Binary'] == 0]

    # Save the split datasets
    non_depressed_females.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv", index=False)
    non_depressed_males.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv", index=False)

    print(f"Non-depressed females: {non_depressed_female_count} participants")
    print(f"Non-depressed males: {non_depressed_male_count} participants")

else:
    print("Error: Required columns ('PHQ8_Binary' or 'Participant_ID') not found in one of the datasets.")
