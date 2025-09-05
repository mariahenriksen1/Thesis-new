
import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to NonDepFemaleNonDepMaleSplit folder (assuming it's one level up)
split_dir = os.path.join(script_dir, "..", "NonDepFemaleNonDepMaleSplit")

# Load the CSV files dynamically
females = pd.read_csv(os.path.join(split_dir, "non_depressed_females.csv"))
males = pd.read_csv(os.path.join(split_dir, "non_depressed_males.csv"))

# Count frames per participant
female_lengths = females.groupby("Participant_ID")["frame"].count().reset_index(name="Total_Frames")
male_lengths = males.groupby("Participant_ID")["frame"].count().reset_index(name="Total_Frames")

# Add group label
female_lengths["Group"] = "Female"
male_lengths["Group"] = "Male"

# Combine
frame_lengths = pd.concat([female_lengths, male_lengths])
print(frame_lengths)
