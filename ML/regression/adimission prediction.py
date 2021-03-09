# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:09:24 2021

@author: KBINM18203
"""

import os
os.getcwd()
os.chdir(r'C:\Users\kbinm18203\Downloads')

import pandas as pd
df=pd.read_csv('Admission_Predict.csv')
df.describe()
plox=df.head()
print(plox)
df.isnull().sum()
cols=df.columns
df.dtypes
numeric_var = [key for key in dict(df.dtypes)
                   if dict(df.dtypes)[key]
                       in ['float64','float32','int32','int64']] # Numeric Variable

cat_var = [key for key in dict(df.dtypes)
             if dict(df.dtypes)[key] in ['object'] ] # Categorical Varible
for i in cols:
    print(df[cols].dtype())
print(numeric_var)

#Seperate dependant/target variable and indepandant variable
x=df.iloc[:,:-1].values
print(x)
y=df['Chance of Admit '].values
print(y)

import seaborn as sns
sns.heatmap(df.corr())
type(df)

df.drop('Serial No.',axis=1)

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(x_test)

from sklearn import linear_model
model = linear_model.LinearRegression(normalize=False)

model.fit(x_train,y_train)
model.intercept_
model.score(x_train, y_train)
model.score(x_test,y_test)
