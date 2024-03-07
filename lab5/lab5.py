import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix

# you may need to close psyder, open it as an administrator, and
# downgrade your matplotlib version:
# pip install matplotlib==3.2.0

# 2) Read the dataset located here 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
# 3) Assign new headers to the DataFrame
data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']

# 4) Drop the 'Sample code number' attribute 
data = data.drop(['Sample code number'],axis=1)

### Missing Values ###
# 5)Convert the '?' to NaN
data = data.replace('?',np.NaN)

# 6) Count the number of missing values in each attribute of the data.
print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))
    
# 7) Discard the data points that contain missing values
data = data.dropna()

### Outliers ### 
# 8)Draw a boxplot to identify the columns in the table that contain outliers 
# The 'Bare Nuclei' attribute is a string. Convert it to a numerical attribute.
data.iloc[:,5]=pd.to_numeric(data.iloc[:,5])
# you may need to close psyder, open it as an administrator, and
# downgrade your matplotlib version:
# pip install matplotlib==3.2.0
data.boxplot(figsize=(20,3))

### Duplicate Data ### 
# 9) Check for duplicate instances.
dups = data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))

# 10) Drop row duplicates
print('Number of rows before discarding duplicates = %d' % (data.shape[0]))
data = data.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % (data.shape[0]))

### Discretization ### 
# 11) Plot a 10-bin histogram of the attribute values 'Clump Thickness' distribution
data['Clump Thickness'].hist(bins=10)

# 12)Discretize the Clump Thickness' attribute into 4 bins of equal width.
data['Clump Thickness'] = pd.cut(data['Clump Thickness'], 4)
data['Clump Thickness'].value_counts(sort=False)

### Sampling ### 
# 13) Randomly select 1% of the data without replacement. The random_state argument of the function specifies the seed value of the random number generator.
sample = data.sample(frac=0.01, replace=False, random_state=1)
print(sample)

X = data.drop(columns=['Class'])
y = data['Class']

X['Clump Thickness'] = X['Clump Thickness'].cat.codes

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = y.map({2: 0, 4: 1})

print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train, y_train)

accuracy = knn_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)


cv_accuracy = cross_val_score(knn_classifier, X_scaled, y, cv=10, scoring='accuracy').mean()
cv_precision = cross_val_score(knn_classifier, X_scaled, y, cv=10, scoring=make_scorer(precision_score)).mean()
cv_recall = cross_val_score(knn_classifier, X_scaled, y, cv=10, scoring=make_scorer(recall_score)).mean()
cv_f1_score = cross_val_score(knn_classifier, X_scaled, y, cv=10, scoring=make_scorer(f1_score)).mean()

print("Average Accuracy:", cv_accuracy)
print("Average Precision:", cv_precision)
print("Average Recall:", cv_recall)
print("Average F1 Score:", cv_f1_score)

y_pred = knn_classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()




























