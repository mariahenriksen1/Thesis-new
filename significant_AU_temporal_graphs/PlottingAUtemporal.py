import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import os

# -----------------------------
# CONFIG
# -----------------------------
ALL_AUS = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r',
           'AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']

DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
MIN_CONFIDENCE = 0.8
OUTPUT_DIR = "./significant_AU_temporal_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Significant AUs with corresponding groups
AU_GROUPS = {
    'AU01_r': ['NonDep_Women','Dep_Women'],
    'AU05_r': ['Men','Women'],             # Gender difference
    'AU15_r_gender': ['Men','Women'],      # Gender difference
    'AU15_r_dep_women': ['NonDep_Women','Dep_Women'], # Dep vs NonDep women
    'AU17_r': ['NonDep_Men','Dep_Men']
}

# -----------------------------
# LOAD FRAMES
# -----------------------------
file_paths = sorted(glob.glob(f"{DATA_DIR}/*_CLNF_AUs_final.csv"))
frame_level_data = []

for path in tqdm(file_paths, desc="Processing frames"):
    df = pd.read_csv(path)
    
    # Filter by confidence and success
    if 'confidence' in df.columns:
        df = df[df['confidence'] >= MIN_CONFIDENCE]
    if 'success' in df.columns:
        df = df[df['success'] == 1]
    if len(df) < 10:
        continue
    
    # Get label
    label_key = next((k for k in ['PHQ8_Binary','PHQ8','phq8','phq8_binary'] if k in df.columns), None)
    if label_key is None:
        continue
    label = int(df[label_key].iloc[0])
    
    # Get gender
    gender = None
    if 'Gender' in df.columns:
        val = df.iloc[0]['Gender']
        if isinstance(val, str):
            gender = 1 if val.lower() in ['male','m','1'] else 0
        else:
            gender = int(val)
    
    # Add frame-level info
    for idx, row in df.iterrows():
        frame_entry = {
            'participant_id': path.split('/')[-1].replace('_CLNF_AUs_final.csv',''),
            'PHQ8_Binary': label,
            'Gender': gender if gender is not None else -1,
            'frame': idx
        }
        for au in ALL_AUS:
            frame_entry[au] = row[au]
        frame_level_data.append(frame_entry)

df_frames = pd.DataFrame(frame_level_data)

# -----------------------------
# NORMALIZE FRAMES AND GROUPS
# -----------------------------
# Normalize frame number
df_frames['frame_norm'] = df_frames.groupby('participant_id')['frame'].transform(lambda x: x / x.max())

# Define combined depression+gender group
def get_group(row):
    if row['PHQ8_Binary']==0 and row['Gender']==1:
        return 'NonDep_Men'
    elif row['PHQ8_Binary']==0 and row['Gender']==0:
        return 'NonDep_Women'
    elif row['PHQ8_Binary']==1 and row['Gender']==1:
        return 'Dep_Men'
    elif row['PHQ8_Binary']==1 and row['Gender']==0:
        return 'Dep_Women'
    else:
        return 'Unknown'

df_frames['Group'] = df_frames.apply(get_group, axis=1)

# Define gender-only group for AU05 and AU15_gender
df_frames['Gender_Group'] = df_frames['Gender'].map({1:'Men', 0:'Women'})

# -----------------------------
# PLOTTING
# -----------------------------
sns.set(style="whitegrid")

for au_key, groups in AU_GROUPS.items():
    # Determine which column to use
    if au_key in ['AU05_r', 'AU15_r_gender']:
        use_col = 'Gender_Group'
        if au_key == 'AU05_r':
            plot_title = f"Temporal Dynamics of AU05_r (Gender difference)"
        else:
            plot_title = f"Temporal Dynamics of AU15_r (Gender difference)"
        au = au_key.replace('_gender','')
    else:
        use_col = 'Group'
        if au_key == 'AU15_r_dep_women':
            au = 'AU15_r'
            plot_title = f"Temporal Dynamics of AU15_r (Dep vs NonDep Women)"
        else:
            au = au_key
            plot_title = f"Temporal Dynamics of {au}"

    plt.figure(figsize=(10,5))
    
    for group, color in zip(groups, sns.color_palette("Set2", len(groups))):
        df_group = df_frames[df_frames[use_col]==group].copy()
        if len(df_group) == 0:
            continue
        df_group = df_group.sort_values('frame_norm')
        
        # Bin frames into 50 bins for clarity
        df_group['time_bin'] = pd.qcut(df_group['frame_norm'], 50, labels=False, duplicates='drop')
        mean_per_bin = df_group.groupby('time_bin')[au].mean()
        sem_per_bin = df_group.groupby('time_bin')[au].sem()
        bin_centers = mean_per_bin.index / max(mean_per_bin.index)  # normalized x-axis
        
        plt.plot(bin_centers, mean_per_bin.values, label=group, color=color)
        plt.fill_between(bin_centers, mean_per_bin - sem_per_bin, mean_per_bin + sem_per_bin,
                         color=color, alpha=0.2)
    
    plt.title(plot_title)
    plt.xlabel("Interview Progress (normalized)")
    plt.ylabel("AU Activation")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    safe_name = au_key.replace("_","")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_name}_temporal.png"))
    plt.close()
