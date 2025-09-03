
import pandas as pd

# Load your pre-split non-depressed data
females = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_females.csv")
males = pd.read_csv("/Users/courtneymarshall/Desktop/DAIC-WOZ/ExploringActionUnits/NonDepFemaleNonDepMaleSplit/non_depressed_males.csv")

# Count frames per participant
female_lengths = females.groupby("Participant_ID")["frame"].count().reset_index(name="Total_Frames")
male_lengths = males.groupby("Participant_ID")["frame"].count().reset_index(name="Total_Frames")

# Add group label
female_lengths["Group"] = "Female"
male_lengths["Group"] = "Male"

# Combine
frame_lengths = pd.concat([female_lengths, male_lengths])
print(frame_lengths)
