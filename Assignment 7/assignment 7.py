import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame({'X': [0.1,0.15,0.08,0.16,0.02,0.25,0.24,0.3],
                   'y': [0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]})
f1 = df['X'].values
f2 = df['y'].values
X = np.array(list(zip(f1,f2)))
print(X)

#centroid points
c_x = np.array([0.1,0.3])
c_y = np.array([0.6,0.2])

colmap = {1: 'r', 2: 'b'}
plt.scatter(f1,f2, color='k')
plt.scatter(c_x[0], c_y[0], marker='*', s=200, c='r')
plt.scatter(c_x[1], c_y[1], marker='*', s=200, c='b')
plt.show()

#centroid
C = np.array([[c_x[0],c_y[0]], [c_x[1], c_y[1]]])
print("Centroid Points are:\n",C)

model = KMeans(n_clusters=2,init=C,n_init=1)
model.fit(X)
labels = model.labels_
print("Clusters are:\n",labels)

print("P6 belongs to cluster", model.labels_[5])

#using labels find population around centroid
count=0
for i in range(len(labels)):
        if(labels[i]==1):
            count = count+1
print("Number of population around cluster 2 is: ",count)

#updated values of centroids
new_centroids = model.cluster_centers_
print("Previous values of centroids m1 and m2 are:\n", C)
print("Updated  values of centroids m1 and m2 are:\n", new_centroids[0], new_centroids[1] )


