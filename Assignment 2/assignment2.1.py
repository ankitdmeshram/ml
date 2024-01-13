import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
x = iris.data
y = iris.target
print(iris.feature_names)
print(iris)

#Step 1: Standardize Data
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

#Step 2: Finding covariance matrix
covariance_matrix = np.cov(x_std, rowvar=False)
print(covariance_matrix)

#Step 3: Findind eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print(eigenvalues)

#Step 4: Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
print(sorted_indices)
sorted_eigenvalues = eigenvalues[sorted_indices]
print(sorted_eigenvalues)
sorted_eigenvectors = eigenvectors[:,sorted_indices]
print(sorted_eigenvectors)

#Step 5: Retain top eigenvalues to capture 95% variance
total_variance = sum(sorted_eigenvalues)
variance_ratio = sorted_eigenvalues/total_variance
print(variance_ratio)
cumulative_variance_ratio = np.cumsum(variance_ratio)
print(cumulative_variance_ratio)
num_components = np.argmax(cumulative_variance_ratio>=0.95)+1
selected_eigenvectors = sorted_eigenvectors[:,:num_components]

#Step 6: Trand=sform data to lower_diemsional space
x_pca = x_std.dot(selected_eigenvectors)

#Step 7: Plot the transformed data
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()
