import pandas as pd

male_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/WomenMenSplit/male_participants.csv")

if 'PHQ8_Binary' in male_data.columns and 'Participant_ID' in male_data.columns:
    depressed_ids = male_data[male_data['PHQ8_Binary'] == 1]['Participant_ID'].unique()
    non_depressed_ids = male_data[male_data['PHQ8_Binary'] == 0]['Participant_ID'].unique()

    depressed_males = male_data[male_data['Participant_ID'].isin(depressed_ids)]
    non_depressed_males = male_data[male_data['Participant_ID'].isin(non_depressed_ids)]

    depressed_males.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/MaleDepressionSplit/depressed_males.csv", index=False)
    non_depressed_males.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/MaleDepressionSplit/non_depressed_males.csv", index=False)

    print(f"Depressed males: {len(depressed_ids)} unique participants")
    print(f"Non-depressed males: {len(non_depressed_ids)} unique participants")
else:
    print("Error: 'PHQ8_Binary' or 'Participant_ID' column not found in the dataset.")
