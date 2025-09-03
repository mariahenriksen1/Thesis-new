import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for AU_c ranges
data = {
    "AU": ["AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c"],
    "Female": [15.93, 73.22, 27.25, 78.44, 82.06, 19.76],
    "Male": [37.82, 87.22, 35.70, 88.54, 59.01, 25.69]
}

# Create DataFrame and melt it to long format
df = pd.DataFrame(data)
df_melted = df.melt(id_vars="AU", var_name="Gender", value_name="Range")

# Set plot style
sns.set(style="whitegrid")

# Create the dot plot
plt.figure(figsize=(10, 6))
sns.stripplot(x="Range", y="AU", hue="Gender", data=df_melted, 
              dodge=True, size=8, palette="Set2", orient='h')

# Titles and labels
plt.title("Range of AU_c Presence (Non-Depressed Group)", fontsize=14)
plt.xlabel("Range")
plt.ylabel("Action Unit")
plt.legend(title="Gender")
plt.tight_layout()

plt.show()
