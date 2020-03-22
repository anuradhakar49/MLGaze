# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:14:17 2020

@author:Anuradha Kar, March 2020
"""
###### train_knn.py ################
## Python program to observe train a KNN classifier using gaze error features
## The output CSV file from the create_features.py is used as input here.
## The train, test and cross validation accuracy are outputs along with
## confusion matrix and precision/recall scores. 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import learning_curve

df = pd.read_csv('features_data.csv')

X = df.drop('Target', 1)  
y = df[["Target"]]

X, y = shuffle(X, y)
x_st = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

############################### KNN training ####################

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train.values.ravel())
y_pred = knn.predict(X_test)
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)
y_score = knn.predict_proba(X_test)
# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print knn.score(X_test, y_test)

print "train acc" , train_accuracy
print "test acc", test_accuracy

########confusion matrix

cnf_knn= confusion_matrix(y_test, y_pred)

########cross validation scores
cv_scores_knn = cross_val_score(knn, x_st, y.values.ravel(), cv=10, scoring='accuracy')

print "Cross validation scores" , cv_scores_knn
print "Mean cross validation score", np.mean(cv_scores_knn)

FP = cnf_knn.sum(axis=0) - np.diag(cnf_knn)  
FN = cnf_knn.sum(axis=1) - np.diag(cnf_knn)
TP = np.diag(cnf_knn)
TN = cnf_knn.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print "Detection scores", np.mean(TPR), np.mean(FPR), np.mean(TNR), np.mean(FNR) 
################# Plotting the confusion matrix
fig= plt.figure(1)

yticklabels = ['UD50', 'UD60', 'UD70', 'UD80', 'Roll', 'Pitch', 'Yaw']
xticklabels = ['UD50', 'UD60', 'UD70', 'UD80', 'Roll', 'Pitch', 'Yaw']

sns.heatmap(cnf_knn,cmap='viridis',linewidths=0.25, annot=True,yticklabels=yticklabels, xticklabels=xticklabels)
plt.title('Confusion matrix KNN')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

############# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve( KNeighborsClassifier(n_neighbors=3), 
                                                        X, 
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv=5,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
fig= plt.figure(2)
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="lightpink")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="lightpink")

# Create plot
plt.title("Learning Curve-KNN")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.grid()
plt.show()