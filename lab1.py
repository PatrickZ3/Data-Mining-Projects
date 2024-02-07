# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:08:24 2024

@author: patze
"""

import pandas as pd
import numpy as np
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)   
df.columns = ["Sepal Length", "Sepal Width", "Petal Length", "petal Width", "Class"] 
columnsNames = ["Sepal Length", "Sepal Width", "Petal Length", "petal Width", "Class"] 

"""
For each quantitative attribute, calculate its average, standard deviation, minimum, and
maximum values. Use the methods associated with the DataFrame; e.g. the average can be
computed using the method described here:
    """

statistics_df = df[columnsNames].describe().transpose()
statistics_df = statistics_df.drop(["25%", "50%", "75%"], axis=1)
display(statistics_df)

print("========================================================")

classCount = df["Class"].value_counts().reset_index()
classCount.columns = ["Class", "Frequency"] 
display (classCount)