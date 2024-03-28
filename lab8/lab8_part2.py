# -*- coding: utf-8 -*-
"""
Part 2
"""

import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.cluster.hierarchy import dendrogram, linkage

# (0 point) Import the vertebrate.csv data
data = pd.read_csv('vertebrate.csv')

# (5 points) Pre-process data: create a new variable and bind it with all the numerical attributes (i.e. all except the 'Name' and 'Class')
NumericalAttributes = data.drop(['Name', 'Class'], axis=1)

### (10 points) Single link (MIN) analysis + plot associated dendrogram ###
min_analysis = linkage(NumericalAttributes, method='single')

# (5 points) Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
plt.figure(figsize=(12, 8))
dendrogram(min_analysis, labels=data['Name'].tolist(), orientation='right')
plt.title('Single Linkage Dendrogram')
plt.show()

### (10 points) Complete Link (MAX) analysis + plot associated dendrogram ###
max_analysis = linkage(NumericalAttributes, method='complete')

# (5 points) Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
plt.figure(figsize=(12, 8))
dendrogram(max_analysis, labels=data['Name'].tolist(), orientation='right')
plt.title('Complete Linkage Dendrogram')
plt.show()

### (10 points) Group Average analysis ###
average_analysis = linkage(NumericalAttributes, method='average')

# (5 points) Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
plt.figure(figsize=(10, 7))
dendrogram(average_analysis, labels=data['Name'].tolist(), orientation='right')
plt.title('Average Linkage Dendrogram')
plt.show()
print("Patrick Lay")