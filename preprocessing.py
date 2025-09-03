import pandas as pd

phq8_data = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/train_split_Depression_AVEC2017.csv")

for participant_id in range(300, 331):
    phq8_participant_data = phq8_data[phq8_data['Participant_ID'] == participant_id]
    
    if phq8_participant_data.empty:
        print(f"No PHQ8 data found for Participant {participant_id}")
    else:
        au_file_path = f"/Users/courtneymarshall/Desktop/DAIC-WOZ/{participant_id}_P/{participant_id}_CLNF_AUs.txt"
        
        try:
            au_data = pd.read_csv(au_file_path, delimiter=",")
            
            au_data['Participant_ID'] = participant_id
            au_data['PHQ8_Binary'] = phq8_participant_data['PHQ8_Binary'].values[0]
            au_data['PHQ8_Score'] = phq8_participant_data['PHQ8_Score'].values[0]
            au_data['Gender'] = phq8_participant_data['Gender'].values[0]

            columns = ['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score', 'Gender'] + [col for col in au_data.columns if col not in ['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score', 'Gender']]
            au_data = au_data[columns]

            output_file_path = f"{participant_id}_merged_au_phq8_data.csv"
            au_data.to_csv(output_file_path, index=False)

            print(f"Data for Participant {participant_id} successfully merged and saved to {output_file_path}")

        except Exception as e:
            print(f"Error loading or processing AU data for Participant {participant_id}: {e}")
