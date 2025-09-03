import pandas as pd

female_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/female_participants.csv")

if 'PHQ8_Binary' in female_data.columns and 'Participant_ID' in female_data.columns:
    depressed_ids = female_data[female_data['PHQ8_Binary'] == 1]['Participant_ID'].unique()
    non_depressed_ids = female_data[female_data['PHQ8_Binary'] == 0]['Participant_ID'].unique()

    # Filter original data to include ALL rows from those participants
    depressed_females = female_data[female_data['Participant_ID'].isin(depressed_ids)]
    non_depressed_females = female_data[female_data['Participant_ID'].isin(non_depressed_ids)]

    # Save to CSV
    depressed_females.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/FemaleDepressionSplit/depressed_females.csv", index=False)
    non_depressed_females.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/FemaleDepressionSplit/non_depressed_females.csv", index=False)

    # Print unique participant counts
    print(f"Depressed females: {len(depressed_ids)} unique participants")
    print(f"Non-depressed females: {len(non_depressed_ids)} unique participants")
else:
    print("Error: 'PHQ8_Binary' or 'Participant_ID' column not found in the dataset.")
