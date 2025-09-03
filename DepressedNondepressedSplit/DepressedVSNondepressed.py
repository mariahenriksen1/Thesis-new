import pandas as pd
import os

data_directory = "/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/PreprocessedFiles"

all_data = []

for file_name in os.listdir(data_directory):
    if file_name.endswith("_merged_au_phq8_data.csv"):  
        file_path = os.path.join(data_directory, file_name)
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"File {file_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

df_combined = pd.concat(all_data, ignore_index=True)

df_combined.columns = df_combined.columns.str.strip()

if 'PHQ8_Binary' in df_combined.columns and 'Participant_ID' in df_combined.columns:
    
    depressed_ids = df_combined[df_combined['PHQ8_Binary'] == 1]['Participant_ID'].unique()
    non_depressed_ids = df_combined[df_combined['PHQ8_Binary'] == 0]['Participant_ID'].unique()

    depressed_data = df_combined[df_combined['Participant_ID'].isin(depressed_ids)]
    non_depressed_data = df_combined[df_combined['Participant_ID'].isin(non_depressed_ids)]

    print(f"Number of unique depressed participants: {len(depressed_ids)}")
    print(f"Number of unique non-depressed participants: {len(non_depressed_ids)}")

    depressed_data.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepressedNondepressedSplit/depressed_participants.csv", index=False)
    non_depressed_data.to_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepressedNondepressedSplit/non_depressed_participants.csv", index=False)

else:
    print("PHQ8_Binary or Participant_ID column not found in the dataframe. Please check the column names.")
