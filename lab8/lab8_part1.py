"""
Part 1
"""

# Imports
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. (0 points) Load the dataset (DataLab8.csv)
moviesDataset = pd.read_csv('DataLab8.csv')

# 2. (5 points) To perform a k-means analysis on the dataset, extract only the numerical attributes: remove the "user" attribute 
data = moviesDataset.drop('user', axis=1)

## Suppose you want to determine the number of clusters k in the initial data 'data' ##
# 3. (5 points) Create an empty list to store the SSE of each value of k (so that, eventually, we will be able to compute the optimum number of clusters k)
sseL = []

# 4. (15 points) Apply k-means with a varying number of clusters k and compute the corresponding sum of squared errors (SSE) 
# Hint1: use a loop to try different values of k. Think about the reasonable range of values k can take (for example, 0 is probably not a good idea).
# Hint2: research about cluster.KMeans and more specifically 'inertia_'
# Hint3: If you get an AttributeError: 'NoneType' object has no attribute 'split', consider downgrading numpy to 1.21.4 this way: pip install --upgrade numpy==1.21.4
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sseL.append(kmeans.inertia_)

#  5. (10 points) Plot to find the SSE vs the Number of Cluster to visually find the "elbow" that estimates the number of clusters. (read online about the "elbow method" for clustering)
plt.figure(figsize=(12, 8))
plt.plot(range(1, 7), sseL, marker='o')
plt.title('Elbow Method for Determining the Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# 6. (5 points) Look at the plot and determine the number of clusters k (read online about the "elbow method" for clustering)
k = 2

# 7. (10 points) Using the optimized value for k, apply k-means on the data to partition the data, then store the labels in a variable named 'labels'
# Hint1: research about cluster.KMeans and more specifically 'labels_'
kmeans_optimized = KMeans(n_clusters=k)
kmeans_optimized.fit(data)
labels = kmeans_optimized.labels_

# 8. Display the assignments of each users to a cluster 
clusters = pd.DataFrame(labels, index=moviesDataset.user, columns=['Cluster ID'])
print('Patrick Lay')