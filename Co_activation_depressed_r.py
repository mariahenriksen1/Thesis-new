import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dep_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv")
dep_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv")

# Add gender column to each dataset
dep_females['Gender'] = 'Female'
dep_males['Gender'] = 'Male'

# List of selected AUs (action units) with '_r' suffix
selected_aus = ['AU06_r', 'AU12_r', 'AU01_r', 'AU04_r', 'AU15_r', 'AU02_r', 'AU05_r', 'AU26_r', 'AU20_r', 'AU09_r', 'AU17_r', 'AU14_r']

# Remove '_r' from the AU names for labeling purposes
base_aus = [au.replace('_r', '') for au in selected_aus]

# Replace -100 with NaN for all AU columns in both datasets
dep_females[selected_aus] = dep_females[selected_aus].replace(-100, pd.NA)
dep_males[selected_aus] = dep_males[selected_aus].replace(-100, pd.NA)

# Function to calculate the correlation matrix for selected AUs and plot the heatmap
def calculate_correlation_for_aus(data, gender):
    # Calculate the correlation matrix for the selected AUs
    correlation_matrix = data[selected_aus].corr()

    # Rename columns and index to remove the '_r' suffix for display
    correlation_matrix.columns = base_aus
    correlation_matrix.index = base_aus
    
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(f'Correlation Matrix for Selected AUs ({gender})')
    plt.show()

    return correlation_matrix

# Plot correlation matrices for Non-Depressed Females and Non-Depressed Males
calculate_correlation_for_aus(dep_females, 'Depressed Females')

calculate_correlation_for_aus(dep_males, 'Depressed Males')
