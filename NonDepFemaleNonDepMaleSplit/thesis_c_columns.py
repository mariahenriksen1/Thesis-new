import os
import pandas as pd

# Folder containing all participant CSVs
folder_path = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

# Read all CSV files and add Participant_ID and Gender/PHQ8_Binary
all_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith("_CLNF_AUs_final.csv"):
        participant_id = int(file_name.split("_")[0])
        df = pd.read_csv(os.path.join(folder_path, file_name))
        df['Participant_ID'] = participant_id
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Filter non-depressed participants
non_depressed_data = data[data['PHQ8_Binary'] == 0]

# Split by gender
females = non_depressed_data[non_depressed_data['Gender'] == 0]
males = non_depressed_data[non_depressed_data['Gender'] == 1]

# Select AU columns ending with "_c"
au_columns_c = [col for col in non_depressed_data.columns if col.endswith('_c')]

# Aggregate by participant: compute proportion of frames with AU present
female_agg = females.groupby('Participant_ID')[au_columns_c].mean()
male_agg = males.groupby('Participant_ID')[au_columns_c].mean()

# Compute mean, median per AU across participants
stats_df = pd.DataFrame(index=au_columns_c)
stats_df['Female_Mean'] = female_agg.mean()
stats_df['Female_Median'] = female_agg.median()
stats_df['Female_Prop_Frames'] = females[au_columns_c].sum() / len(females)

stats_df['Male_Mean'] = male_agg.mean()
stats_df['Male_Median'] = male_agg.median()
stats_df['Male_Prop_Frames'] = males[au_columns_c].sum() / len(males)

# Round for readability
stats_df = stats_df.round(3)

# Show table
print(stats_df)

#         Female_Mean  Female_Median  Female_Prop_Frames  Male_Mean  Male_Median  Male_Prop_Frames
# AU04_c        0.373          0.377               0.373      0.356        0.339             0.354
# AU12_c        0.450          0.452               0.448      0.317        0.232             0.323
# AU15_c        0.201          0.180               0.201      0.201        0.174             0.199
# AU23_c        0.701          0.779               0.695      0.788        0.875             0.782
# AU28_c        0.113          0.023               0.108      0.121        0.026             0.120
# AU45_c        0.403          0.396               0.406      0.380        0.388             0.386