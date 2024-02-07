# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:04:30 2024

@author: patze
"""
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Question 2"""
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
df = pd.read_csv(url, header=None) 

"""Question 3"""
df.columns=["Sample Code Number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
display(df)

"""Question 4"""
df=df.drop(["Sample Code Number"],axis=1)
print("=============== Removed Sample Code Number ===============")
display(df)

"""Question 5"""
df.replace('?',np.nan, inplace=True)
print("=============== Replaced '?' to Numpy Nan ===============")
display(df)
missing_values_count = df.isna().sum()

"""Question 6"""
print("=============== Number of missing values for each attribute: ===============")
print(missing_values_count)

"""Question 7"""
df.dropna(inplace=True)
print("=============== Removed Missing Values ===============")
display(df)

"""Question 8"""
fig = plt.figure(figsize =(20, 3))
df.boxplot()
"""
Attributes that has outliers: 
- Marginal Adhesion
- Single Epithelial Cell Size
- Bland Chromatin
- Normal Nucleoli
- Mitoses
"""

"""Question 9"""
print("=============== Finding Duplicates ===============")
print("Amount of Records That Were Duplicates:", df.duplicated().sum())

"""Question 10"""
print("=============== Removing Duplicates ===============")
df.drop_duplicates(inplace=True)
display(df)

"""Question 11"""
df["Clump Thickness"].hist(bins=10)
bins = pd.cut(df['Clump Thickness'],4)
print("=============== Discretize into 4 bins ===============")
print("Range of values and number of records for each category:")
print(bins.value_counts())


"""Question 12"""
print("=============== Random 1% Sample ===============")
sample = df.sample(frac=0.01, random_state=1)
display(sample)




















