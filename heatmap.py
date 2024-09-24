import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'mean_iterations_minus_five.csv'
data = pd.read_csv(file_path)

# Pivot the data to create a matrix with 'batch_size' as rows and 'initial_samples' as columns
heatmap_data = data.pivot(index='batch_size', columns='initial_samples', values='mean_iterations')

# Sort the rows (batch_size) and columns (initial_samples) in increasing order
heatmap_data = heatmap_data.sort_index(ascending=True).sort_index(axis=1, ascending=True)

# Create a heatmap with batch_size on the y-axis and initial_samples on the x-axis
plt.figure(figsize=(10, 8))
ax= sns.heatmap(heatmap_data, annot=False, cmap='magma', cbar=True, fmt='.1f',linewidths=0.2,linecolor='black')

ax.invert_yaxis()

ax.tick_params(axis='both', which='major', labelsize=14) 

# Adjust the font size of the colorbar tick labels
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)  # Adjust 'labelsize' as needed
# Add labels and title
plt.title('Mean number of iterations to reach maximum',fontsize=14)
plt.xlabel('Initial Samples',fontsize=14)
plt.ylabel('Batch Size',fontsize=14)
plt.tight_layout()

plt.savefig("branin_heatmap.png",dpi=600, transparent=True,bbox_inches="tight")

