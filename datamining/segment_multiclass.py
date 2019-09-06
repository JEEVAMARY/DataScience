# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:14:14 2019

@author: jeeva
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir(r'C:\Users\jeeva\Downloads')
multi=pd.read_csv('segment_csv.csv')
#Preprocessing
desc=multi.describe()
multi.dtypes
#Encoding
c = multi['class'].astype('category')
d = dict(enumerate(c.cat.categories))
print (d)
multi['class'] = multi['class'].astype('category').cat.codes
#multi['back'] = multi['class'].map(d)
multi.dtypes
multi.isnull().sum()
multi['class'].value_counts()
#Correlation Analysis
multi.corr()
sns.heatmap(multi.corr(),cmap='Blues' ,annot= True,annot_kws={"size": 5} )
##Extract independant and response variables
target=multi['class']
X=multi.drop(['class'], axis=1)
y=target


#Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
accuracies = []
feature_set = []
max_accuracy_so_far = 0
for i in range(1,len(X.columns)+1):
    selector = RFE(LogisticRegression(), i,verbose=1)
    selector = selector.fit(X, y)
    current_accuracy = selector.score(X,y)
    accuracies.append(current_accuracy)
    feature_set.append(selector.support_)
    if max_accuracy_so_far < current_accuracy:
        max_accuracy_so_far = current_accuracy
        selected_features = selector.support_
    print('End of iteration no. {}'.format(i))
X_sub = X.loc[:,selected_features]
print  (selected_features)

#Spliting of dataset
from sklearn.cross_validation  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sub,y,random_state=0)
#train the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#train the model
model.fit(X_train,y_train) #training


#predict testcases
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)
#performance measures on the test set
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score 
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
j=classification_report(y_test,y_pred)
scores = []
max_score = 0
"""
from sklearn.model_selection import KFold
from sklearn.svm import SVR


best_svr = SVR(kernel='rbf')
cv = KFold(n_splits=5, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))
print(np.mean(scores)) """


