import pandas as pd

full_test_split_data = pd.read_csv('/Users/courtneymarshall/Desktop/DAIC-WOZ/full_test_split.csv')

missing_participants = [300, 301, 306, 308, 309, 311, 314, 323, 329]

for participant_id in missing_participants:
    participant_data = full_test_split_data[full_test_split_data['Participant_ID'] == participant_id]
    
    if not participant_data.empty:
        au_file_path = f"/Users/courtneymarshall/Desktop/DAIC-WOZ/{participant_id}_P/{participant_id}_CLNF_AUs.txt"
        try:
            au_data = pd.read_csv(au_file_path, delimiter=",")
            
            au_data['PHQ8_Binary'] = participant_data['PHQ_Binary'].values[0]
            au_data['PHQ8_Score'] = participant_data['PHQ_Score'].values[0]
            au_data['Gender'] = participant_data['Gender'].values[0]
            au_data['Participant_ID'] = participant_id
            
            columns = ['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score', 'Gender'] + [col for col in au_data.columns if col not in ['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score', 'Gender']]
            au_data = au_data[columns]
            
            output_file_path = f"{participant_id}_merged_au_phq8_data.csv"
            au_data.to_csv(output_file_path, index=False)
            
            print(f"Data for Participant {participant_id} successfully merged and saved to {output_file_path}")
        
        except Exception as e:
            print(f"Error loading or processing AU data for Participant {participant_id}: {e}")
    else:
        print(f"No PHQ8 data found for Participant {participant_id}")
