import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output directories relative to the script location
data_directory = os.path.abspath(os.path.join(script_dir, "..", "PreprocessedFiles"))
output_directory = os.path.abspath(os.path.join(script_dir, "..", "WomenMenSplit"))

# Make sure the output directory exists
os.makedirs(output_directory, exist_ok=True)

all_data = []

# Load all files that match the pattern
for file_name in os.listdir(data_directory):
    if file_name.endswith("_merged_au_phq8_data.csv"):
        file_path = os.path.join(data_directory, file_name)
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"File {file_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

# Combine all data into one DataFrame
df_combined = pd.concat(all_data, ignore_index=True)

# Remove leading/trailing spaces from column names
df_combined.columns = df_combined.columns.str.strip()

print(f"Column names in the combined dataframe: {df_combined.columns.tolist()}")

# Split by gender and participant ID
if 'Gender' in df_combined.columns and 'Participant_ID' in df_combined.columns:
    male_data = df_combined[df_combined['Gender'] == 1]
    female_data = df_combined[df_combined['Gender'] == 0]

    unique_male_count = male_data['Participant_ID'].nunique()
    unique_female_count = female_data['Participant_ID'].nunique()

    print(f"Number of unique male participants: {unique_male_count}")
    print(f"Number of unique female participants: {unique_female_count}")

    # Save to CSV (keeping all rows)
    male_data.to_csv(os.path.join(output_directory, "male_participants.csv"), index=False)
    female_data.to_csv(os.path.join(output_directory, "female_participants.csv"), index=False)

else:
    print("Gender or Participant_ID column not found in the dataframe. Please check the column names.")

# Check missing participants
all_participant_ids = df_combined['Participant_ID'].unique()
male_participant_ids = male_data['Participant_ID'].unique()
female_participant_ids = female_data['Participant_ID'].unique()

missing_male_participants = set(all_participant_ids) - set(male_participant_ids)
missing_female_participants = set(all_participant_ids) - set(female_participant_ids)

print(f"Missing male participants (not in male dataset): {missing_male_participants}")
print(f"Missing female participants (not in female dataset): {missing_female_participants}")
