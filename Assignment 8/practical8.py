import numpy as np
from sklearn_extra.cluster import KMedoids  # Correct import statement
import matplotlib.pyplot as plt

# Prepare the data
data = np.array([
    [0.1, 0.6],
    [0.15, 0.71],
    [0.08, 0.9],
    [0.16, 0.85],
    [0.2, 0.3],
    [0.25, 0.5],
    [0.24, 0.1],
    [0.3, 0.2]
])

# Choose the number of clusters (adjust as needed)
k = 2

# Create and fit the KMedoids model
kmedoids = KMedoids(n_clusters=k, random_state=0).fit(data)

# Access cluster information
print("Medoids:", kmedoids.cluster_centers_)
print("Cluster labels:", kmedoids.labels_)
print("Total cost:", kmedoids.inertia_)

# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=kmedoids.labels_)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], marker='x', color='red', s=200)
plt.title("K-Medoids Clustering Results")
plt.show()
