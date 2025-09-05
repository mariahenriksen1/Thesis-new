import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the NonDepFemaleNonDepMaleSplit folder
nondep_split_dir = os.path.join(script_dir, "..", "NonDepFemaleNonDepMaleSplit")

# Load the CSV files dynamically
nondep_females = pd.read_csv(os.path.join(nondep_split_dir, "non_depressed_females.csv"))
nondep_males = pd.read_csv(os.path.join(nondep_split_dir, "non_depressed_males.csv"))


# Define threshold
threshold_value = 0.1  # Set the threshold value

# Define all action unit columns (replace with your actual list of AU columns)
all_aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r']

# Apply thresholding to all AU columns (both non-depressed and depressed groups)
for au in all_aus:
    nondep_females[au] = nondep_females[au].apply(lambda x: x if x >= threshold_value else 0)
    nondep_males[au] = nondep_males[au].apply(lambda x: x if x >= threshold_value else 0)

# Prepare data for PCA (you can choose nondep_females or nondep_males based on the group you want to analyze)
data_for_pca = nondep_females[all_aus]  # Non-depressed females data

# Normalize the data using StandardScaler (important for PCA)
scaler = StandardScaler()
data_for_pca_scaled = scaler.fit_transform(data_for_pca)

# Initialize PCA to reduce dimensions (adjust n_components as needed, e.g., 2 for 2D projection)
pca = PCA(n_components=2)

# Fit PCA to the data and transform it
pca_results = pca.fit_transform(data_for_pca_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])

# Plot the PCA results (PC1 vs. PC2)
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], color='blue', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Non-Depressed Females (After Thresholding)')
plt.show()

# Print the explained variance ratio to understand the importance of each principal component
print("Explained Variance Ratio for PCA (Non-Depressed Group):", pca.explained_variance_ratio_)

