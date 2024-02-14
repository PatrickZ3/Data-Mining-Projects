# Import the packages
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1) (5 points) Read the vertebrate.csv data
data = pd.read_csv('vertebrate.csv')
print("=============== Read Data ===============")
display(data)

# 2) (15 points) The number of records is limited. Convert the data into a binary classification: mammals versus non-mammals
# Hint: ['fishes','birds','amphibians','reptiles'] are considered 'non-mammals'
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')
data
print("=============== Convert to binary classification ===============")
display(data)

# 3) (15 points) We want to classify animals based on the attributes: Warm-blooded,Gives Birth,Aquatic Creature,Aerial Creature,Has Legs,Hibernates
# For training, keep only the attributes of interest, and seperate the target class from the class attributes
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1:]

# 4) (10 points) Create a decision tree classifier object. The impurity measure should be based on entropy. Constrain the generated tree with a maximum depth of 3
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
# 5) (10 points) Train the classifier
clf.fit(x, y)
print("=============== Classifier ===============")
display(x)
display(y)

# 6) (25 points) Suppose we have the following data
testData = [['lizard',0,0,0,0,1,1,'non-mammals'],
           ['monotreme',1,0,0,0,1,1,'mammals'],
           ['dove',1,0,0,1,1,0,'non-mammals'],
           ['whale',1,1,1,0,0,0,'mammals']]
testData = pd.DataFrame(testData, columns=data.columns)

# Prepare the test data and apply the decision tree to classify the test records.
# Extract the class attributes and target class from 'testData'
yTest = testData.iloc[:, -1:]
xTest = testData.iloc[:, 1:-1]
predY = clf.predict(xTest)
results = pd.DataFrame()
results['Name'] = testData['Name']
results['Actual'] = yTest
results['Predicted'] = predY
print("=============== testData ===============")
display(results)
# Hint: The classifier should correctly label the vertabrae of 'testData' except for the monotreme

# 7) (10 points) Compute and print out the accuracy of the classifier on 'testData'
print("=============== Accuract of testData ===============")
print("The accuracy of the classifier is:", accuracy_score(yTest, predY))

# 8) (10 points) Plot your decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=list(x.columns), class_names=clf.classes_, filled=True)
plt.show()