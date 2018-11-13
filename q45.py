from scipy.cluster.hierarchy import dendrogram, linkage, ward, fcluster
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import numpy as np

expec = np.array([0.88,0.90,0.90,0.87,0.93,0.89,0.88,0.81,0.82,0.85,0.77,0.71,0.75,0.70,0.44,0.47,0.23,0.34,0.31,0.24,0.76])
educ = np.array([0.99,0.99,0.98,0.98,0.93,0.97,0.87,0.92,0.92,0.90,0.85,0.83,0.83,0.62,0.58,0.37,0.33,0.36,0.35,0.37,0.80])
pib = np.array([0.91,0.93,0.94,0.97,0.93,0.92,0.91,0.80,0.75,0.64,0.69,0.72,0.63,0.60,0.37,0.45,0.27,0.51,0.32,0.36,0.61])
polit = np.array([1.10,1.26,1.24,1.18,1.20,1.04,1.41,0.55,1.05,0.07,-1.36,0.47,-0.87,0.21,-1.36,-0.68,-1.26,-1.98,-0.55,0.20,0.39])

X = []
for i,j,k,l in zip(expec,educ,pib,polit):
    X.append([i,j,k,l])
X = np.array(X)

Z = linkage(X, 'ward')
clusters = fcluster(Z, 0.5, criterion='distance')
fig = plt.figure(figsize=(25, 10))

dn = dendrogram(Z)


Z = linkage(X, 'single')

kmeans = fcluster(Z, 0.5, criterion='distance')
fig = plt.figure(figsize=(25, 10))

dn = dendrogram(Z)
plt.show()

print(clusters)
print(kmeans)
