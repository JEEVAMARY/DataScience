# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:03:45 2019

@author: jeeva
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\jeeva\Downloads\abalone-dataset')
dataset=pd.read_csv('abalone.csv')
#Data Preprocessing
dataset.head()
description=dataset.describe()
dataset.dtypes
#Handling missig values
dataset.isnull().sum()
#Correlation analysis of numerical variables
dataset.corr()
sns.heatmap(dataset.corr(),cmap='Blues' , annot= True)
#Encoding
dataset['Sex'] = dataset['Sex'].map( {'M':1, 'F':2 , 'I':0} )
#Extract independant and response variables
X= dataset.drop(['Rings'], axis=1)
Xs=X
y = dataset['Rings'].reshape(-1,1)


#Normalise
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#Recursive Feature Elimination
from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import RFE
adj_R2 = []
feature_set = []
max_adj_R2_so_far = 0
n = len(X)
k = len(X[0])
for i in range(1,k+1):
    selector = RFE(LinearRegression(), i,verbose=1)
    selector = selector.fit(X, y)
    current_R2 = selector.score(X,y)
    current_adj_R2 = 1-(n-1)*(1-current_R2)/(n-i-1) 
    adj_R2.append(current_adj_R2)
    feature_set.append(selector.support_)
    if max_adj_R2_so_far < current_adj_R2:
        max_adj_R2_so_far = current_adj_R2
        selected_features = selector.support_
    print('End of iteration no. {}'.format(i))
print(selected_features)    
X_sub = X[:,selected_features]

#Spliting of dataset
from sklearn.cross_validation  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sub,y,random_state=0)
#train the model
model = LinearRegression()
model.fit(X_train,y_train)
model.coef_
#see performance score 
model.score(X_test,y_test)
#prediction
y_pred = model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred.reshape(len(y_pred),1)).reshape(len(y_pred))
y_test = sc_y.inverse_transform(y_test.reshape(len(y_test),1)).reshape(len(y_test))
plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
#see performance score
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
import statsmodels.api as sm
#OLS model
X_modified = sm.add_constant(X_train)
lin_reg = sm.OLS(y_train,X_modified)
result = lin_reg.fit()
print(result.summary())
#K-Fold model Score---4 Fold
scores = []
max_score = 0
from sklearn.model_selection import KFold
kf = KFold(n_splits=4,random_state=0,shuffle=True)
for train_index, test_index in kf.split(X_sub):
    X_train, X_test = X_sub[train_index], X_sub[test_index]
    y_train, y_test = y[train_index], y[test_index]
    current_model = LinearRegression()
    #train the model
    current_model.fit(X_train,y_train)
    #see performance score
    current_score = model.score(X_test,y_test)
    scores.append(current_score)
    if max_score < current_score:
        max_score = current_score
        best_model = current_model


best_model.intercept_
best_model.coef_
