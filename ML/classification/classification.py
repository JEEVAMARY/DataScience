# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 07:42:33 2019

@author: jeeva
"""

import os
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
os.chdir(r'C:\Users\jeeva\Desktop\ML')
dataset=pd.read_csv('meningitis_dataset.csv')
dataset.shape
dataset = dataset.sample(frac=0.01)
dataset.shape
cols=dataset.columns.tolist()
dataset.isnull().sum()
print(cols)
"""serotypemeningitis sero type, null if sero type not certain or if disease==meningitis
NmANisseria Meningitidis Group A hot encoding
NmCNisseria Meningitidis Group C hot encoding
NmWNisseria Meningitidis Group W hot encoding"""
dat=dataset.drop(['id', 'surname', 'firstname', 'middlename', 'gender',  'gender_female', 'state', 'settlement', 'urban_settlement', 'report_date',  'age_str', 'date_of_birth','adult_group', 'cholera', 'diarrhoea', 'measles', 'viral_haemmorrhaphic_fever', 'meningitis', 'ebola', 'marburg_virus', 'yellow_fever', 'rubella_mars', 'malaria', 'serotype','NmW', 'health_status', 'dead', 'report_outcome', 'unconfirmed'],axis=1)
print(dat.columns)
dat['disease'] = dat['disease'].astype('category').cat.codes
#CORRELATION ANALYSIS
dat.corr()
sns.heatmap(dat.corr(),cmap='Blues' ,annot= True,annot_kws={"size": 5} )
##Extract independant and response variables
target=dat['disease']
X=dat.drop(['disease'], axis=1)
y=target
#RECURSIVE FEATURE ELIMINATION
nof_list=np.arange(1,10)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 0)
    model = LogisticRegression()
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
lr=LogisticRegression()
rfe = RFE(lr, n_features_to_select=nof)
rfe.fit(X,y)
print ("Features sorted by their rank:")
print(rfe.support_)
print(rfe.ranking_)
selected_features = rfe.support_
X_sub = X.loc[:,selected_features]

#Spliting of dataset

X_train, X_test, y_train, y_test = train_test_split(X_sub,y,random_state=0)
#train the model
acc_score={}
macro_p_r_s={}
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#train the model
model.fit(X_train,y_train) #training
#predict testcases
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)
#performance measures on the test set
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score 
def perf(y_test,y_pred):
    confusion_matrix(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    print("recall"+str(precision_recall_fscore_support(y_test, y_pred, average='macro')))
    print(classification_report(y_test,y_pred))    
    return( accuracy_score(y_test,y_pred))
acc_score.update( LogisticRegression = perf(y_test,y_pred))
#kfold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=10, shuffle=True, random_state=1)
result=cross_val_score(model,X,y,cv=kf)
print(result)
print("Accuracy: "+ str(result.mean()*100))
#KNN
from sklearn.neighbors import KNeighborsClassifier
knc= KNeighborsClassifier()
knc.fit(X_train,y_train)
y_pred = knc.predict(X_test)
y_pred_probs = knc.predict_proba(X_test)
accuracy_score(y_test,y_pred)
acc_score.update(KNeighborsClassifier = perf(y_test,y_pred))

param_dict = {
                'n_neighbors':[3,5,7,4,11], 
                'weights':['uniform','distance'] ,
                'metric':['euclidean','manhattan']
             }

model = GridSearchCV(KNeighborsClassifier(),param_grid=param_dict,cv=4) 
model.fit(X,y)
model.best_params_
train_scores = model.cv_results_['mean_train_score']
train_scores.mean()
test_scores = model.cv_results_['mean_test_score']
test_scores.mean()
model.best_score_
grid_score={}
grid_score.update(knn= (model.best_score_))
#ADABOOST
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
acc_score.update( AdaBoostClassifier= perf(y_test,y_pred))
param_dict = {'n_estimators':[50,30,20,10,45],
              'learning_rate':[1,2,3]
             }
# run grid search
model = GridSearchCV(AdaBoostClassifier(), param_grid=param_dict,cv=4)
model.fit(X,y)
model.best_params_
model.best_score_
train_scores = model.cv_results_['mean_train_score']
train_scores.mean()
test_scores = model.cv_results_['mean_test_score']
test_scores.mean()
grid_score.update(adaboost = model.best_score_)
#RANDOMFOREST

rf = RandomForestClassifier(n_estimators=15, max_depth=3)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
y_pred = rf.predict(X_test)
acc_score.update( RandomForestClassifier= perf(y_test,y_pred))
param_dict = {   'n_estimators':[10] ,           
                'max_depth':[1,5,10,15,25]
                            }
rf = RandomForestClassifier()
model = GridSearchCV(rf,param_grid=param_dict,cv=4) 
model.fit(X_train,y_train)
score=model.score(X_test,y_test)
model.best_params_
train_scores = model.cv_results_['mean_train_score']
test_scores = model.cv_results_['mean_test_score']
model.best_score_
train_scores.mean()
test_scores = model.cv_results_['mean_test_score']
test_scores.mean()
grid_score.update(randomforest = model.best_score_)

#SVM
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
acc_score.update(SupportVectorClassifier = perf(y_test,y_pred))
param_dict = {
                'C': [1.0,0.1,0.001], 
                'kernel':['linear','rbf','poly'] ,
                'degree':[2,3]
             }

model = GridSearchCV(clf,param_grid=param_dict,cv=4) 
model.fit(X_train,y_train)
model.best_params_
train_scores = model.cv_results_['mean_train_score']
test_scores = model.cv_results_['mean_test_score']
model.best_score_
train_scores.mean()
test_scores.mean()
