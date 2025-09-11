import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
depressed_females = pd.read_csv(os.path.join(script_dir, "depressed_females.csv"))
depressed_males   = pd.read_csv(os.path.join(script_dir, "depressed_males.csv"))

au_cols = ['AU06_r', 'AU10_r']

female_means = depressed_females.groupby('Participant_ID')[au_cols].mean()
male_means = depressed_males.groupby('Participant_ID')[au_cols].mean()

# Define a function to compute descriptive stats
def describe_au(data, label):
    for au in ['AU06_r', 'AU10_r']:
        print(f"\n{label} - {au}:")
        print(f"  Mean: {data[au].mean():.4f}")
        print(f"  Median: {data[au].median():.4f}")
        print(f"  Std Dev: {data[au].std():.4f}")
        print(f"  Min: {data[au].min():.4f}")
        print(f"  Max: {data[au].max():.4f}")
        print(f"  Range: {(data[au].max() - data[au].min()):.4f}")

# # Output stats
# describe_au(female_means, "Depressed Females")
# describe_au(male_means, "Depressed Males")

purple = '#9B4D96'   
orange = '#FF8C00'   

plt.figure(figsize=(12, 5))
for i, au in enumerate(au_cols):
    plt.subplot(1, 2, i+1)
    plt.hist(female_means[au], bins=20, alpha=0.6, label='Female', color=purple, density=True)
    plt.hist(male_means[au], bins=20, alpha=0.6, label='Male', color=orange, density=True)
    
    plt.title(f'{au} Histogram (Depressed)')
    plt.xlabel('Mean Intensity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

