import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_depressed = pd.DataFrame({
    'Action Unit': ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r'],
    'Depressed_Female_Mean': [0.446298, 0.295865, 0.452506, 0.129523, 0.130917, 0.287375, 0.157875, 0.219778, 0.359008, 0.128013, 0.108418, 0.309031, 0.876855, 0.065702],
    'Depressed_Male_Mean': [0.569394, 0.250925, 0.388471, 0.168596, 0.315756, 0.406532, 0.341814, 0.263017, 0.487247, 0.136241, 0.191032, 0.265057, 0.911228, 0.137661]
})

df_depressed_melted = df_depressed.melt(id_vars='Action Unit', value_vars=['Depressed_Female_Mean', 'Depressed_Male_Mean'],
                                         var_name='Group', value_name='Mean Intensity')

plt.figure(figsize=(12, 6))
sns.barplot(x='Action Unit', y='Mean Intensity', hue='Group', data=df_depressed_melted, 
            palette=["#7F4B9B", "#B5D9A4"])

for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

plt.title('Mean AU Intensities for Depressed Females vs. Males')
plt.ylabel('Mean Intensity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
