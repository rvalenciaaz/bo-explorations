import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_melted=pd.read_csv("instances.csv")
# Using seaborn to plot with confidence intervals
plt.figure(figsize=(10, 6))

# Use seaborn lineplot to automatically handle mean and confidence intervals
sns.lineplot(data=df_melted, x='Iteration', y='Value', estimator='mean', errorbar=('ci', 95))

plt.xticks(range(1, max(df_melted['Iteration'])+1))

plt.grid(True)  # Enable grid

plt.title('Value vs. Iteration with 95% Confidence Interval')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig("instances.png",bbox_inches="tight",dpi=600)
