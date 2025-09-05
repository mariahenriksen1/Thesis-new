import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
data_directory = os.path.abspath(os.path.join(script_dir, "..", "PreprocessedFiles"))
output_directory = os.path.abspath(os.path.join(script_dir, "..", "DepressedNondepressedSplit"))


all_data = []

# Load all relevant CSV files
for file_name in os.listdir(data_directory):
    if file_name.endswith("_merged_au_phq8_data.csv"):
        file_path = os.path.join(data_directory, file_name)
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"File {file_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

# Combine all dataframes
df_combined = pd.concat(all_data, ignore_index=True)
df_combined.columns = df_combined.columns.str.strip()

# Check required columns and split
if 'PHQ8_Binary' in df_combined.columns and 'Participant_ID' in df_combined.columns:
    
    depressed_ids = df_combined[df_combined['PHQ8_Binary'] == 1]['Participant_ID'].unique()
    non_depressed_ids = df_combined[df_combined['PHQ8_Binary'] == 0]['Participant_ID'].unique()

    depressed_data = df_combined[df_combined['Participant_ID'].isin(depressed_ids)]
    non_depressed_data = df_combined[df_combined['Participant_ID'].isin(non_depressed_ids)]

    print(f"Number of unique depressed participants: {len(depressed_ids)}")
    print(f"Number of unique non-depressed participants: {len(non_depressed_ids)}")

    # Save split datasets to output directory
    depressed_file = os.path.join(output_directory, "depressed_participants.csv")
    non_depressed_file = os.path.join(output_directory, "non_depressed_participants.csv")

    depressed_data.to_csv(depressed_file, index=False)
    non_depressed_data.to_csv(non_depressed_file, index=False)
    print("Files saved successfully.")
else:
    print("PHQ8_Binary or Participant_ID column not found in the dataframe. Please check the column names.")
