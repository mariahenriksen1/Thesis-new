import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Action Unit': ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r'],
    'Non_Depressed_Female_Mean': [0.620577, 0.359541, 0.647040, 0.292825, 0.182633, 0.452980, 0.265126, 0.328448, 0.470481, 0.231957, 0.259547, 0.348539, 1.385979, 0.072442],
    'Non_Depressed_Male_Mean': [0.569298, 0.358853, 0.449013, 0.155471, 0.166311, 0.391502, 0.217553, 0.249340, 0.360770, 0.151219, 0.183642, 0.318852, 1.078395, 0.058689]
})

df_melted = df.melt(id_vars='Action Unit', value_vars=['Non_Depressed_Female_Mean', 'Non_Depressed_Male_Mean'],
                    var_name='Group', value_name='Mean Intensity')

plt.figure(figsize=(12, 6))
sns.barplot(x='Action Unit', y='Mean Intensity', hue='Group', data=df_melted, 
            palette=["#7F4B9B", "#B5D9A4"])

for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

plt.title('Mean AU Intensities for Non-Depressed Females vs. Males')
plt.ylabel('Mean Intensity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
