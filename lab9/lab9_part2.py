# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 1) (10 points) Load the data from the file 'dataOutliers.npy'
data = np.load('dataOutliers.npy')

# 2) (10 points) Create a scatter plot to visualize the data (This is just a FYI, make sure to comment the below line after you visualized the data)
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Data')
plt.show()

# 3) (20 points) Anomaly detection: Density-based
# Fit the LocalOutlierFactor model for outlier detection
# Then predict the outlier detection labels of the data points
lof = LocalOutlierFactor()
outlier_labels = lof.fit_predict(data)


# 4) (10 points) Plot results: make sure all plots/images are closed before running the below commands
# Create a scatter plot of the data (exact same as in 2) )
# Then, indicate which points are outliers by plotting circles around the outliers
outliers = data[outlier_labels == -1]
plt.scatter(data[:, 0], data[:, 1], c=outlier_labels, cmap='viridis')
plt.scatter(outliers[:, 0], outliers[:, 1], edgecolors='r', facecolors='none', s=100, label='Outliers')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Data with Outliers Detected')
plt.legend()
plt.show()
print('Patrick Lay')