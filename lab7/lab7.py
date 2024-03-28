# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:08:09 2024

@author: patze
"""

import pandas as pd
import numpy as np
from apyori import apriori

data = pd.read_csv('weather.csv')

temperature_bins = pd.cut(data['temperature'], bins=3, labels=['co8ol', 'mild', 'hot'])
data['temperature'] = temperature_bins

humidity_bins = pd.cut(data['humidity'], bins=2, labels=['normal', 'high'])
data['humidity'] = humidity_bins

data['windy'] = data['windy'].map({True: 'True', False: 'False'})

data_list = data.values.tolist()

print(data_list)

results = list(apriori(data_list, min_support=0.28, min_confidence=0.5))

for itemset in results:
    for rule_index in range(len(itemset.ordered_statistics)):
        print(list(itemset.ordered_statistics[rule_index].items_base),
              '->',
              list(itemset.ordered_statistics[rule_index].items_add),
              ', Support:', itemset.support,
              ', Confidence:', itemset.ordered_statistics[rule_index].confidence)

print('Patrick Lay')