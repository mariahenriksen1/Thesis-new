import matplotlib.pyplot as plt
from matplotlib.table import Table

# Data for Depressed Women vs Depressed Men
AUs_dep_split = ["AU09_r","AU17_r","AU12_r","AU05_r","AU15_r","AU02_r","AU25_r",
                 "AU04_r","AU10_r","AU06_r","AU20_r","AU14_r","AU26_r","AU01_r"]
p_values_dep_split = [0.03633,0.09932,0.24862,0.28403,0.39142,0.46833,0.57522,
                      0.65632,0.65632,0.68031,0.70463,0.71691,0.86909,0.97370]
movements_dep_split = ["Nose Wrinkler","Chin Raiser","Lip Corner Puller","Upper Lid Raiser",
                       "Lip Corner Depressor","Outer Brow Raiser","Lips Part","Brow Lowerer",
                       "Upper Lip Raiser","Cheek Raiser","Lip Stretcher","Dimpler","Jaw Drop","Inner Brow Raiser"]

# Create figure
fig, ax = plt.subplots(figsize=(8,6))
ax.axis('off')

# Create table
tbl = Table(ax, bbox=[0,0,1,1])

# Column widths
col_widths = [0.2, 0.2, 0.6]
row_height = 0.05

# Header
headers = ["AU Code", "P-Value", "Facial Movement"]
for i, header in enumerate(headers):
    cell = tbl.add_cell(0, i, width=col_widths[i], height=row_height, text=header, loc='center', facecolor='lightgray')
    cell.get_text().set_weight('bold')

# Add data rows
for j in range(len(AUs_dep_split)):
    bold = p_values_dep_split[j] < 0.05  # highlight significant AU
    cell1 = tbl.add_cell(j+1, 0, width=col_widths[0], height=row_height, text=AUs_dep_split[j], loc='center', facecolor='white')
    cell2 = tbl.add_cell(j+1, 1, width=col_widths[1], height=row_height, text=f"{p_values_dep_split[j]:.5f}", loc='center', facecolor='white')
    cell3 = tbl.add_cell(j+1, 2, width=col_widths[2], height=row_height, text=movements_dep_split[j], loc='center', facecolor='white')
    
    if bold:
        cell1.get_text().set_weight('bold')
        cell2.get_text().set_weight('bold')
        cell3.get_text().set_weight('bold')

# Add table to axes
ax.add_table(tbl)
plt.title("Facial Action Units: P-Values", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
