# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:40:55 2019

@author: jeeva
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


#exploring source dataset------------------------------------------------------
pmsm_data=pd.read_csv('pmsm_temperature_data.csv')
pmsm_data.shape
print(pmsm_data.columns)
profile_idmax=pmsm_data.profile_id.value_counts().idxmax()
#Taking the profile id with most data recordings as the dataset
pmsm=pmsm_data.loc[pmsm_data.profile_id == profile_idmax ]
pmsm.drop('profile_id',1,inplace=True)
pmsm = pmsm.sample(frac=0.10)
#Data Preprocessing
pmsm.head()
pmsm.shape
pmsm.describe()
pmsm.isnull().sum()#Checking for NA values
#-------------------------------------------------------------------------------
#Correlation-------------------------------------------------------------------
pmsm.corr()
sns.heatmap(pmsm.corr(),cmap='Blues' , annot= True)
target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']
#Split features and Targets----------------------------------------------------
X=pmsm.drop(target_features,1)
#Y=pmsm[target_features]
y=pmsm['stator_winding']
#------------------------------------------------------------------------------
lr = LinearRegression()
#Recursive Feature Elimination-------------------------------------------------
nof_list=np.arange(1,8)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=nof)
rfe.fit(X,y)
print ("Features sorted by their rank:")
print(rfe.support_)
print(rfe.ranking_)
selected_features = rfe.support_
X_sub = X.loc[:,selected_features]
#------------------------------------------------------------------------------
#Split to train and test ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_sub,y,random_state=0)

"""Linear Regression"""
model = LinearRegression()
model.fit(X_train,y_train)
model.coef_
#see performance score 
model.score(X_test,y_test)
model.score(X_train,y_train)
score_d={}
score_d.update(LinearRegression = model.score(X_test,y_test))
#Kfold CrossValidation
kf = KFold(n_splits=10, shuffle=True, random_state=1)
result=cross_val_score(model,X,y,cv=kf)
print(result)
print("Accuracy: "+ str(result.mean()*100))

"""KNearest Neighbours"""

knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
knn.score(X_train,y_train)
score_d.update(KNeighborsRegressor = knn.score(X_test,y_test) )
#Grid Search
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
dicti=model.best_params_
dicti["n_neighbors"]
model.best_score_
#score----------------------------------------------------------------------
def evaluation(model,X_test,y_test,pred):
    mse = mean_squared_error(y_test,pred)
    r2=model.score(X_test,y_test)
    print("R2 score "+str(r2))
    train_score=model.score(X_train,y_train)
    print("Train score "+str(train_score))
    print("Mean Squared Error "+str(mse))
#model fit function------------------------------------------------------------
def model_run(model):
    model.fit(X_test,y_test)
    pred=model.predict(X_test)
    evaluation(model,X_test,y_test,pred)
    return model.score(X_test,y_test)

"""SVR"""
svm=(SVR(kernel='linear'))
svm_sc=model_run(svm)
score_d.update(SupportVectorRegressor = svm_sc)
#gridsearch
svr =SVR()
param_grid={'kernel': ('linear', 'rbf','poly'),'C': [0.1, 1,10]}
        
gsc = GridSearchCV(
        svr,param_grid,cv=4  
        )
gsc.fit(X_train,y_train)
model.best_params_
model.best_score_
train_scores = model.cv_results_['mean_train_score']
print("Mean training scores "+str(train_scores.mean()))
test_scores = model.cv_results_['mean_test_score']
print(test_scores.mean())
print("Mean test scores "+str(test_scores.mean()))
model.best_score_

"""Random Forest Regression"""
rfr=RandomForestRegressor(max_depth=4, random_state=2)
rfr_sc=model_run(rfr)
score_d.update( RandomForestRegressor = rfr_sc)
#gridsearch---
param_dict = {   'n_estimators':[10] ,           
                'max_depth':[1,5,10,15,25]
                            }
rfr= RandomForestRegressor()
model = GridSearchCV(rfr,param_grid=param_dict,cv=4) 
model.fit(X_train,y_train)
score=model.score(X_test,y_test)
model.best_params_
train_scores = model.cv_results_['mean_train_score']
print("Mean training scores "+str(train_scores.mean()))
test_scores = model.cv_results_['mean_test_score']
print(test_scores.mean())
print("Mean test scores "+str(test_scores.mean()))
model.best_score_

"""Adaboost Regressor"""
ada=AdaBoostRegressor(n_estimators=500,learning_rate=0.001,random_state=1)
ada_sc=model_run(ada)
score_d.update( AdaBoostRegressor = ada_sc)
parameters = {'n_estimators': (150,250,500,750),
               'learning_rate':(1,2,5)             }
adb = GridSearchCV( AdaBoostRegressor(),param_grid=parameters,cv=4)
adb.fit(X_train, y_train)
adb.best_params_
adb.best_score_
train_scores = adb.cv_results_['mean_train_score']
print("Mean training scores "+str(train_scores.mean()))
test_scores = adb.cv_results_['mean_test_score']
print(test_scores.mean())
print("Mean test scores "+str(test_scores.mean()))

"""OLS model"""
X_modified = sm.add_constant(X_train)
lin_reg = sm.OLS(y_train,X_modified)
result = lin_reg.fit()
print(result.summary())
