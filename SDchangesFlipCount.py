import pandas as pd
import matplotlib.pyplot as plt

# Create long-form data
data = [
    # AU, Group, Gender, SD_NonDep, SD_Dep
    ['AU04_c', 'Female', 25.94, 36.46],
    ['AU04_c', 'Male', 23.92, 23.02],
    ['AU12_c', 'Female', 28.49, 19.84],
    ['AU12_c', 'Male', 21.61, 22.07],
    ['AU15_c', 'Female', 32.94, 14.46],
    ['AU15_c', 'Male', 48.32, 13.63],
    ['AU23_c', 'Female', 61.11, 59.61],
    ['AU23_c', 'Male', 51.36, 42.68],
    ['AU28_c', 'Female', 24.16, 28.19],
    ['AU28_c', 'Male', 33.33, 33.01],
    ['AU45_c', 'Female', 31.25, 9.55],
    ['AU45_c', 'Male', 14.64, 25.53],
]

df = pd.DataFrame(data, columns=['AU', 'Gender', 'SD_NonDep', 'SD_Dep'])

# Create a unique label for y-axis
df['Label'] = df['AU'] + ' (' + df['Gender'] + ')'

# Sort by AU then Gender (Female first)
df['GenderOrder'] = df['Gender'].map({'Female': 0, 'Male': 1})
df = df.sort_values(by=['AU', 'GenderOrder']).reset_index(drop=True)

# Set up the plot
plt.figure(figsize=(10, 6))
y_pos = range(len(df))

# Plot lines and dots
for i, row in df.iterrows():
    plt.plot([row['SD_NonDep'], row['SD_Dep']], [i, i], color='gray', lw=1)
    plt.scatter(row['SD_NonDep'], i, color='orange', label='Non-Depressed' if i == 0 else "")
    plt.scatter(row['SD_Dep'], i, color='green', label='Depressed' if i == 0 else "")

# Formatting
plt.yticks(ticks=y_pos, labels=df['Label'])
plt.xlabel('Standard Deviation of Flip Counts')
plt.title('Change in Flip Count Variability by Gender and Depression Status')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()

plt.show()
