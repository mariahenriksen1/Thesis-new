import pandas as pd
import glob
import os
from scipy import stats

DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final/"
AU_COLUMNS = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r',
              'AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']

file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))

participant_stats = []

for path in file_paths:
    participant_id = os.path.basename(path).replace('_CLNF_AUs_final.csv', '')
    df = pd.read_csv(path)
    
    # Get label
    label_key = next((k for k in ['PHQ8_Binary','PHQ8','phq8','phq8_binary'] if k in df.columns), None)
    if label_key is None:
        continue
    label = int(df[label_key].iloc[0])
    
    # Get numeric gender code
    gender_val = df.iloc[0]['Gender'] if 'Gender' in df.columns else None
    if isinstance(gender_val, str):
        gender = 1 if gender_val.lower() in ['male','m','1'] else 0
    elif gender_val is not None:
        gender = int(gender_val)
    else:
        gender = -1  # unknown
    
    # Participant-level AU means
    au_means = df[AU_COLUMNS].mean()
    
    stats_dict = {'participant_id': participant_id, 'PHQ8_Binary': label, 'Gender': gender}
    for au in AU_COLUMNS:
        stats_dict[f'{au}_mean'] = au_means[au]
    
    participant_stats.append(stats_dict)

df_participant_stats = pd.DataFrame(participant_stats)
print(df_participant_stats.head())

df_male = df_participant_stats[df_participant_stats['Gender'] == 1]
df_female = df_participant_stats[df_participant_stats['Gender'] == 0]

gender_corrs = []
for au in AU_COLUMNS:
    corr_m, _ = stats.pointbiserialr(df_male['PHQ8_Binary'], df_male[f'{au}_mean'])
    corr_f, _ = stats.pointbiserialr(df_female['PHQ8_Binary'], df_female[f'{au}_mean'])
    gender_corrs.append({
        'AU': au,
        'Male_Corr': corr_m,
        'Female_Corr': corr_f,
        'Difference': abs(corr_m - corr_f)
    })

gender_corr_df = pd.DataFrame(gender_corrs).sort_values('Difference', ascending=False)
print("\nGender-specific AU-depression correlations:")
print(gender_corr_df.to_string(index=False))

