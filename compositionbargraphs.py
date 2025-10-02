# import matplotlib.pyplot as plt

# # Data
# categories = ["Men", "Women"]
# values = [102, 87]
# total = sum(values)
# percentages = [v / total * 100 for v in values]

# # Create bar graph
# plt.figure(figsize=(6,4))
# plt.bar(categories, values, color=['steelblue', 'lightcoral'])

# # Title with numbers included dynamically
# plt.title(f"Gender Composition of DAIC-WOZ Dataset Participants")

# plt.ylabel("Count")
# plt.xlabel("Gender")

# plt.tight_layout()
# plt.show()


# # import matplotlib.pyplot as plt

# # # Data
# # categories = ["Depressed", "Non-Depressed"]
# # values = [56, 133]
# # total = sum(values)
# # percentages = [v / total * 100 for v in values]

# # # Create bar graph
# # plt.figure(figsize=(6,4))
# # plt.bar(categories, values, color=['#1f77b4', '#ffdd57'])  # blue and yellow

# # # Title
# # plt.title(
# #     "Depression Status Composition of DAIC-WOZ Dataset Participants"
# # )

# # plt.ylabel("Count")
# # plt.xlabel("Depression Status")

# # plt.tight_layout()
# # plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Depressed", "Non-Depressed"]
men_values = [25, 77]
women_values = [31, 56]

x = np.arange(len(categories))  # positions for groups
width = 0.35  # width of bars

# Create figure
plt.figure(figsize=(7,5))

# Plot bars with muted colors
plt.bar(x - width/2, men_values, width, label='Men', color='#6CA0A3')      # muted teal
plt.bar(x + width/2, women_values, width, label='Women', color='#F5A78A')  # muted coral

# Title and labels
plt.title("Depression Status by Gender in DAIC-WOZ Dataset")
plt.ylabel("Count")
plt.xlabel("Depression Status")
plt.xticks(x, categories)
plt.legend()

plt.tight_layout()
plt.show()

