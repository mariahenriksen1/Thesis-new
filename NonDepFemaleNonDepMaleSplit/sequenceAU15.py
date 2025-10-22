import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
data_folder = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"
au_column = "AU15_r"
target_frames = 25000  # number of frames to show
downsample_step = 500  # plot every 500 frames

# --- LOAD NON-DEPRESSED PARTICIPANTS ---
women_data = []
men_data = []

for file in os.listdir(data_folder):
    if file.endswith("_CLNF_AUs_final.csv"):
        df = pd.read_csv(os.path.join(data_folder, file))
        
        # Skip if required columns missing
        if "Gender" not in df.columns or "PHQ8_Binary" not in df.columns:
            continue
        
        # Only non-depressed participants
        if df["PHQ8_Binary"].iloc[0] != 0:
            continue
        
        au_values = df[au_column].values
        num_original_frames = len(au_values)
        
        # Interpolate or truncate to exactly 25,000 frames
        if num_original_frames >= target_frames:
            original_frames = np.arange(num_original_frames)
            target_frame_indices = np.linspace(0, num_original_frames-1, target_frames)
            f = interp1d(original_frames, au_values, kind='linear')
            aligned_values = f(target_frame_indices)
        else:
            aligned_values = np.pad(au_values, (0, target_frames - num_original_frames), 'edge')
        
        # Append to gender-specific list
        if df["Gender"].iloc[0] == 0:
            women_data.append(aligned_values)
        elif df["Gender"].iloc[0] == 1:
            men_data.append(aligned_values)

# --- CONVERT TO ARRAYS ---
women_data = np.array(women_data)
men_data = np.array(men_data)

# --- DOWNSAMPLE FOR PLOTTING ---
time_plot = np.arange(0, target_frames, downsample_step)
women_plot = women_data[:, ::downsample_step]
men_plot = men_data[:, ::downsample_step]

# --- PLOT ---
plt.figure(figsize=(16,6))

# Women sequences
for series in women_plot:
    plt.plot(time_plot, series, color='lightcoral', alpha=0.3)
plt.plot(time_plot, women_plot.mean(axis=0), color='red', linestyle='--', linewidth=2, label='Non-Depressed Women Mean')

# Men sequences
for series in men_plot:
    plt.plot(time_plot, series, color='skyblue', alpha=0.3)
plt.plot(time_plot, men_plot.mean(axis=0), color='blue', linestyle='--', linewidth=2, label='Non-Depressed Men Mean')

plt.xlabel(f"Frame (every {downsample_step} frames)")
plt.ylabel(f"{au_column} Intensity")
plt.title(f"{au_column} Sequences for Non-Depressed Men vs Women (Downsampled)")
plt.ylim(0, 5)
plt.legend()
plt.tight_layout()
plt.show()
