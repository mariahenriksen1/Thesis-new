import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# load data
females = pd.read_csv(os.path.join(script_dir, "non_depressed_females.csv"))
males = pd.read_csv(os.path.join(script_dir, "non_depressed_males.csv"))


au_c_cols = [col for col in females.columns if col.endswith('_c')]

females[au_c_cols] = females[au_c_cols].replace(-100, pd.NA)
males[au_c_cols] = males[au_c_cols].replace(-100, pd.NA)

female_lengths = females.groupby('Participant_ID')['frame'].nunique().reset_index(name='Total_Frames')
male_lengths = males.groupby('Participant_ID')['frame'].nunique().reset_index(name='Total_Frames')

def compute_flips_and_normalize(df, lengths_df, group_label):
    df = df.merge(lengths_df[['Participant_ID', 'Total_Frames']], on='Participant_ID', how='left')
    df = df.sort_values(by=["Participant_ID", "frame"])
    flips = []

    for pid, group in df.groupby('Participant_ID'):
        group_flips = {}
        for col in au_c_cols:
            values = group[col].values
            clean_values = pd.Series(values).dropna().values
            if len(clean_values) > 1:
                flip_count = (clean_values[1:] != clean_values[:-1]).sum()
            else:
                flip_count = 0
            group_flips[col] = flip_count

        total_frames = group['Total_Frames'].iloc[0]
        for col in au_c_cols:
            group_flips[col] = (group_flips[col] / total_frames) * 1000

        group_flips['Participant_ID'] = pid
        group_flips['Group'] = group_label
        flips.append(group_flips)

    return pd.DataFrame(flips)

female_flips = compute_flips_and_normalize(females, female_lengths, "Female")
male_flips = compute_flips_and_normalize(males, male_lengths, "Male")
all_flips = pd.concat([female_flips, male_flips])

flips_melted = all_flips.melt(id_vars=["Participant_ID", "Group"], var_name="AU", value_name="Normalized_Flips")

plt.figure(figsize=(16, 6))
sns.boxplot(
    data=flips_melted,
    x="AU",
    y="Normalized_Flips",
    hue="Group",
    palette={"Female": "#9B4D96", "Male": "#FF8C00"}
)
plt.title("Flip Counts per 1000 Frames by AU (Non-Depressed)")
plt.ylabel("Flip Count per 1000 Frames")
plt.xlabel("Action Unit (AU)")
plt.xticks(rotation=45)
plt.legend(title="Group")
plt.tight_layout()
plt.show()





