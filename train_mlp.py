# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:44:00 2020

@author: 14233242
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 05 11:56:07 2019

@author: 14233242
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier  
import seaborn as sns
from sklearn.model_selection import learning_curve

df = pd.read_csv('features_data.csv')

X = df.drop('Target', 1) 
y = df[["Target"]]

X, y = shuffle(X, y)
x_st = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

############ training the MLP
mlp = MLPClassifier(max_iter=1000)
mlp = MLPClassifier(hidden_layer_sizes=(10,10),alpha=0.0001, max_iter=2000)  
mlp.fit(X_train, y_train.values.ravel())  
y_pred = mlp.predict(X_test)
train_accuracy = mlp.score(X_train, y_train)
test_accuracy = mlp.score(X_test, y_test)
acc= metrics.accuracy_score(y_test, y_pred)

# Generate the confusion matrix and classification report
cnf_mlp= confusion_matrix(y_test, y_pred)

print cnf_mlp
print(classification_report(y_test, y_pred))
print mlp.score(X_test, y_test)
print "train acc" , train_accuracy
print "test acc", acc

#plt.plot(mlp.loss_curve_)

fig= plt.figure(1)

plt.title('Confusion matrix (MLP)')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

yticklabels = ['UD50', 'UD60', 'UD70', 'UD80', 'Roll', 'Pitch', 'Yaw']
xticklabels = ['UD50', 'UD60', 'UD70', 'UD80', 'Roll', 'Pitch', 'Yaw']
sns.heatmap(cnf_mlp,cmap='coolwarm_r',linewidths=0.25, annot=True)#,yticklabels=yticklabels, xticklabels=xticklabels)
plt.show()

FP = cnf_mlp.sum(axis=0) - np.diag(cnf_mlp)  
FN = cnf_mlp.sum(axis=1) - np.diag(cnf_mlp)
TP = np.diag(cnf_mlp)
TN = cnf_mlp.sum() - (FP + FN + TP)
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

cv_scores_mlp= cross_val_score(mlp, X, y.values.ravel(), cv=10, scoring='accuracy')
print "Cross validation scores" , cv_scores_mlp
print "Mean cross validation score", np.mean(cv_scores_mlp)

############# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(MLPClassifier(hidden_layer_sizes=(10,10)), 
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
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="lightgreen")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="lightgreen")

# Create plot
plt.title("Learning Curve-MLP")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.grid()
plt.show()