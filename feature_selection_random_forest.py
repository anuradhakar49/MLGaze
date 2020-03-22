# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:35:33 2020

@author: Anuradha Kar, March 2020
"""

###### feature_selection_random_forest.py ################
## Python program to observe significance of gaze error features using
## a Random forest classifier. The output CSV file from the create_features.py is
#used as input here.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('features_data.csv')

X = df.drop('Target', 1)  
y = df[["Target"]]
X, y = shuffle(X, y)

########################### random forest based feature significance detector
x_st = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(x_st, y, test_size = 0.3, random_state=42, stratify=y)
clf_rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0)
clf_rf.fit(X_train, y_train.values.ravel())
##  Feature importance score 
print clf_rf.score(X_test, y_test)
importances = clf_rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(np.arange(0, 20, step=1))
plt.ylabel("Feature number")
