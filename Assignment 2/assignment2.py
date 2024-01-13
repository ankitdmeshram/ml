import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
print(iris)

df = pd.DataFrame(iris['data'],columns = iris['feature_names'])
print(df)

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
print(scaled_data)

pca = PCA(0.95)
pca.fit(scaled_data)

pca_X = pca.transform(scaled_data)
print(scaled_data.shape)
print(pca_X.shape)

y = iris.target
plt.scatter(pca_X[:,0], pca_X[:,1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()
