import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for AU12 — Lip Corner Puller 
df = pd.DataFrame({
    'Condition': ['Non-Depressed', 'Non-Depressed', 'Depressed', 'Depressed'],
    'Group': ['Female', 'Male', 'Female', 'Male'],
    'Presence (%)': [30.64, 30.25, 24.21, 15.39],
    'Intensity': [0.328448 * 100, 0.249340 * 100, 0.219778 * 100, 0.263017 * 100]
})

df_melted = df.melt(id_vars=['Condition', 'Group'], var_name='Metric', value_name='Value')

df_melted['Label'] = df_melted['Condition'] + ' - ' + df_melted['Group']

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_melted, x='Label', y='Value', hue='Metric', palette=['#c7a0d7', '#b6d7b9'])

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

plt.title('AU12 (Lip Corner Puller) — Presence vs. Intensity by Group and Condition')
plt.ylabel('Value (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
