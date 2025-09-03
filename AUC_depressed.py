import pandas as pd
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

dep_females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_females.csv")
dep_males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/DepMaleDepFemaleSplit/depressed_males.csv")

au_r_cols = [col for col in dep_females.columns if col.endswith('_r')]

# Function to compute AUC
def compute_auc(df, participant_col='Participant_ID', frame_col='frame'):
    auc_list = []
    
    for pid, group in df.groupby(participant_col):
        participant_auc = {'Participant_ID': pid}
        
        group = group.sort_values(by=frame_col) 
        
        for au in au_r_cols:
            y = group[au].values
            x = group[frame_col].values  
            
            if len(x) > 1:
                auc = simpson(y=y, x=x)
            else:
                auc = 0 
                
            participant_auc[au] = auc
        
        auc_list.append(participant_auc)
    
    return pd.DataFrame(auc_list)

# Compute AUC for females and males
female_auc = compute_auc(dep_females)
male_auc = compute_auc(dep_males)

# Add gender column
female_auc['Gender'] = 'Female'
male_auc['Gender'] = 'Male'

# Combine data
auc_combined = pd.concat([female_auc, male_auc])

# Melt data to long format
auc_melted = auc_combined.melt(id_vars=['Participant_ID', 'Gender'], 
                               var_name='AU', 
                               value_name='AUC_Value')

# Normalize AUC values per AU
scaler = MinMaxScaler()
auc_melted['AUC_Value_Normalized'] = auc_melted.groupby('AU')['AUC_Value'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)

# Create Boxplot
plt.figure(figsize=(14, 6))
sns.boxplot(x="AU", y="AUC_Value_Normalized", hue="Gender", data=auc_melted, palette="Set2")
plt.title("Box Plot of Normalized AUC Values by Action Unit (Depressed Males vs Females)")
plt.xlabel("Action Unit (AU)")
plt.ylabel("Normalized AUC")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)

# Fix legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title="Gender")
plt.tight_layout()
plt.show()


