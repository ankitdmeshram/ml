import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import matplotlib.pyplot as plt

# Create a simulated shopping trends dataset
data = pd.DataFrame({
    "Age": [25, 35, 45, 55, 22, 38, 41, 52, 28, 31],
    "Annual Income (USD)": [45000, 60000, 75000, 90000, 38000, 72000, 65000, 88000, 42000, 51000],
    "Average Purchase Value (USD)": [50, 75, 100, 125, 40, 80, 90, 110, 55, 60],
    "Purchase Frequency (per month)": [5, 3, 2, 1, 6, 4, 3, 2, 7, 5]
})

# Preprocess the data (replace missing values, if any)
data = data.fillna(data.mean())

# Select relevant features for clustering
features = ["Age", "Annual Income (USD)", "Average Purchase Value (USD)", "Purchase Frequency (per month)"]
data_subset = data[features]

# Create distance matrix
distance_matrix = linkage(data_subset, method='ward')

# Generate dendrogram
plt.figure(figsize=(15, 10))
dendrogram(distance_matrix)
plt.title("Dendrogram of Simulated Shopping Trends Dataset")
plt.show()

# Choose number of clusters (based on dendrogram and domain knowledge)
n_clusters = 3  # Adjust as needed

# Cut the dendrogram to obtain clusters
clusters = cut_tree(distance_matrix, n_clusters=n_clusters)

# Assign cluster labels to data points
data["cluster_label"] = clusters

# Further analysis and interpretation based on the simulated dataset
