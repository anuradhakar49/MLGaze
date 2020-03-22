# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:39:47 2020

@author:Anuradha Kar, March 2020
"""
###### train_SVM.py ################
## Python program to observe train six regression models classifier using gaze 
## angles and error values. 
## The regression model parameters and RMSE scores of model fit are outputs
## data available from: https://data.mendeley.com/datasets/cfm4d9y7bh/1


import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import signal
from sklearn import metrics 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
############### load data

gaze_ang=[]
pit_ang=[]
yaw_ang=[]
err=[]

with open('C:/Users/Documents/Python Scripts/ml_gaze/us01_typ20_tab.csv','r') as csvfile:
    datavals = csv.reader(csvfile, delimiter=',')
    datavals.next()
    for r1 in datavals:
       

        gaze_ang.append(float(r1[14]))   #14th column is gaze frontal angle , 12th is pitch, 10th is yaw
        pit_ang.append(float(r1[12]))
        yaw_ang.append(float(r1[10]))  
        err.append(float(r1[15]))


gaze_ang= gaze_ang[1:2491]   
pit_ang= pit_ang[1:2491] 
yaw_ang= yaw_ang[1:2491]  
err= err[1:2491] 
#### outlier removal and error calculation
gaze_f= signal.medfilt(gaze_ang,41)
yaw_f= signal.medfilt(yaw_ang,41)
pit_f= signal.medfilt(pit_ang,41)

err_f= signal.medfilt(err,41)   


dataX = {'GAZE ANG':gaze_f,'YAW':yaw_f, 'PITCH':pit_f}
dataY = {'ERROR':err_f}
X = pd.DataFrame (dataX)
y = pd.DataFrame (dataY)

X_st = StandardScaler().fit_transform(X)
y_st = StandardScaler().fit_transform(y)

####################fitting the regression models
regr = linear_model.LinearRegression()  
regr.fit(X_st, y_st)

ridgereg = Ridge(alpha=0.001,normalize=True)
ridgereg.fit(X_st, y_st)

lassoreg = Lasso(alpha=0.001,normalize=True, max_iter=1e5)
lassoreg.fit(X_st, y_st)

elast = ElasticNet(alpha=0.2,random_state=0)
elast.fit(X_st, y_st)
   
nn = MLPRegressor(
    hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(X_st, y_st)


############# poly fit
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X_st)
model = LinearRegression()
model.fit(x_poly, y_st)

###test data####
test_data= pd.read_csv('C:/Users/14233242/Documents/Python Scripts/ml_gaze/us20_typ20_tab.csv')
testX= pd.DataFrame()


x1= signal.medfilt(test_data['GAZE ANG'].values,41)
x2= signal.medfilt(test_data['YAW DATA'].values,41)
x3= signal.medfilt(test_data['PITCH DATA'].values,41)


testX['GAZE ANG'] = x1[60:2491]
testX['YAW'] = x2[60:2491]
testX['PITCH'] = x3[60:2491]
testX = StandardScaler().fit_transform(testX)
########## test Y
yact = test_data['DIFF GZ'].values
yact1=signal.medfilt(yact,41)
yact1=yact1[60:2491]


############## predicted values
test_poly_x = polynomial_features.fit_transform(testX)
y_poly_pred = model.predict(test_poly_x)
ypred= regr.predict(testX)
yrid = ridgereg.predict(testX)
ylas = lassoreg.predict(testX)
yelast= elast.predict(testX)
ynn= nn.predict(testX)

plt.plot(abs(yact1), 'r', label= "Actual")
plt.plot(abs(ypred), 'g',label="Linear")
plt.plot(abs(y_poly_pred),'b',label= "Poly")
plt.plot(abs(yrid), 'k',label= "Ridge")
plt.plot(abs(ylas), 'c',label= "Lasso")
plt.plot(abs(yelast), 'm',label= "Elastic")
plt.plot(abs(ynn), 'y',label= "MLP")

#plt.xlim(500,1500)
plt.legend(loc='top left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.title("Actual and predicted gaze errors-Tablet Yaw")
plt.xlabel("Data samples")
plt.ylabel(" Gaze error (degrees)")
print "Mean value", np.mean(yact1)
print('RMSE-linear:', np.sqrt(metrics.mean_squared_error(yact1, ypred)))
print('RMSE-poly:', np.sqrt(metrics.mean_squared_error(yact1, y_poly_pred)))
print('RMSE-rid:', np.sqrt(metrics.mean_squared_error(yact1, yrid)))
print('RMSE-lasso:', np.sqrt(metrics.mean_squared_error(yact1, ylas)))
print('RMSE-elastic:', np.sqrt(metrics.mean_squared_error(yact1, yelast)))
print('RMSE-NN:', np.sqrt(metrics.mean_squared_error(yact1, ynn)))

print('Intercept: \n', elast.intercept_)
print('Coefficients: \n', elast.coef_)

# Create means and standard deviations of training set scores
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-.", linewidth=2, label="Training error")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation error")
    plt.legend(loc='upper right')
    plt.ylim([0.5, 1.5])

fig= plt.figure(2)
plt.title("Learning Curve-ElasticNet")
plt.xlabel("Training Set Size"), plt.ylabel("RMSE error"), plt.legend(loc="best")
plot_learning_curves(ElasticNet(), X_st, y_st)
plt.grid()
    