import os
import glob
import pandas as pd
import numpy as np

# -----------------------------
# LOAD LABELS ONLY
# -----------------------------
DATA_DIR = "/Users/raemarshall/Desktop/daicwoz/cleaned_participants_final"

labels = []
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_CLNF_AUs_final.csv")))

for path in file_paths:
    df = pd.read_csv(path)
    df = df[df["success"] == 1]
    labels.append(int(df["PHQ8_Binary"].iloc[0]))

y = np.array(labels)

# -----------------------------
# CHECK CLASS DISTRIBUTION
# -----------------------------
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("\nClass distribution (PHQ8_Binary):")
for label, count in class_distribution.items():
    print(f"  Class {label}: {count} participants ({count / len(y) * 100:.1f}%)")
