# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:57:58 2019

@author: jeeva
"""

@author: jeeva
"""
""""https://data.world/ahalps/social-influence-on-shopping/workspace/file?filename=WhatsgoodlyData-6.csv"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.getcwd()
os.chdir(r'C:\Users\jloui\Downloads')
"""What social platform has influenced your online shopping most?"""
influence= pd.read_csv('WhatsgoodlyData-6.csv' )
influence.drop(['Question'],axis=1,inplace=True)
Seg=influence['Segment Type'].unique()
type(Seg)
Seg.dtype
print(Seg)
for x in Seg:
    globals()['df%s' % x] = pd.DataFrame(data=(influence[influence['Segment Type']== x]))
app=influence['Answer'].dropna().unique()
appl=app.tolist()
for x in app:
    globals()['df%s' % x] = pd.DataFrame(data=(influence[influence['Answer']== x]))
ust=(influence['Segment Description'].unique().tolist())
#1-------------Global Analysis
pos = np.arange(len(app))
count=dfMobile.Count.tolist()
help(plt.bar) 
plt.bar(pos,count,color='blue',edgecolor='black',width=.21)
plt.xticks(pos,appl)
plt.xlabel('Application', fontsize=20)
plt.ylabel('Count', fontsize=16)
plt.title(' Count of application influenced in segment-Mobile',fontsize=18)
plt.show()

#2---------------Gender Analysis

Count_Male=dfGender[dfGender['Segment Description']=='Male voters'].Count.tolist()
P_male=dfGender[dfGender['Segment Description']=='Male voters'].Percentage.tolist()
P_fm=dfGender[dfGender['Segment Description']=='Female voters'].Percentage.tolist()
Count_Female=dfGender[dfGender['Segment Description']=='Female voters'].Count.tolist() 
female_male=dfGender['Segment Description'].unique()
pos = np.arange(len(appl))
bar_width = 0.35
Count_Male=dfGender[dfGender['Segment Description']=='Male voters'].Count.tolist()
P_male=dfGender[dfGender['Segment Description']=='Male voters'].Percentage.tolist()
P_fm=dfGender[dfGender['Segment Description']=='Female voters'].Percentage.tolist()
Count_Female=dfGender[dfGender['Segment Description']=='Female voters'].Count.tolist()
 
plt.bar(pos,Count_Male,bar_width,color='blue',edgecolor='black')
plt.bar(pos+bar_width,Count_Female,bar_width,color='pink',edgecolor='black')
plt.xticks(pos, appl)
plt.xlabel('Application', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Group Barchart - Count  By Gender',fontsize=18)
plt.legend(['Male','Female'],loc=2)
plt.show()  
#----------------
dfCustom.drop(['Segment Type'],axis=1,inplace=True)
Custom=dfCustom['Segment Description'].unique()
School=dfCustom[dfCustom['Segment Description'].str.match('or private school?')]
Class=dfCustom[dfCustom['Segment Description'].str.match('your parents make?')]
Engineer=dfCustom[dfCustom['Segment Description'].str.match("What's your major? ME/EE/other engineer")]
Loan=dfCustom[dfCustom['Segment Description'].str.match('student loan debt?')]
Zodiac=dfCustom[dfCustom['Segment Description'].str.match('your zodiac sign?')]
Leaning=dfCustom[dfCustom['Segment Description'].str.match("What's your leaning?")]  
#3--------SCHOOL GOING
"""Percentage of Influence for whole school going"""
TotalSc=School.groupby(School.Answer).sum()
print(TotalSc)
val=TotalSc.Percentage

colors = ['b', 'g', 'r', 'c', 'm', 'y']
labels = appl
explode = (0, 0, 0.2, 0, 0)
plt.pie(val, colors=colors, labels= val,explode=explode,
                counterclock=False, shadow=True)
plt.title('Whole School Going ')
plt.legend(labels,loc=3)
plt.show()

S=School['Segment Description'].unique()

for x in S:
       globals()[('%s' % x).replace("or private school?", "").replace(" ","")] = pd.DataFrame(data=(School[School['Segment Description']== x]))
Sc=['Noschool','Private','Public']  
schoolf= {
'Noschool Count':Noschool.Count.tolist(),'Noschool Percent':Noschool.Percentage.tolist(),
'Private Count':Private.Count.tolist(),'Private Percent':Private.Percentage.tolist(),'Public Percent ':Public.Percentage.tolist(),}
schoodf=pd.DataFrame(schoolf,index=appl)
for x in schoodf:
    if 'Percent' in x:
        j=x
        values=schoodf[j]
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        labels = appl
        explode = (0.2, 0, 0, 0, 0)
        plt.pie(values, colors=colors, labels= values,explode=explode,
                counterclock=False, shadow=True)
        plt.title(x)
        plt.legend(labels,loc=3)
        plt.show()
#4-----LOAN DEBT
L=Loan['Segment Description'].unique()
for x in L:
    if 'Yes' in x:
        j=x
        globals()['yes' ] = pd.DataFrame(data=(Loan[Loan['Segment Description']== j]))
    else:
        globals()['no' ] = pd.DataFrame(data=(Loan[Loan['Segment Description']== x]))
        
yes=yes.set_index('Answer')        
yes=yes.drop('Segment Description',axis=1)  
no=no.drop('Segment Description',axis=1) 
yesc= yes.Count.tolist() 
no=no.set_index('Answer')
noc=no.Count.tolist()    

pos=pos = np.arange(len(app))     
plt.bar(pos,yesc,bar_width,color='blue',edgecolor='black')
plt.bar(pos+bar_width,noc,bar_width,color='pink',edgecolor='black')
plt.xticks(pos, appl)
plt.xlabel('Application', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Group Barchart - Count  By People with and without debt',fontsize=18)
plt.legend(['Yes','No'],loc=2)
plt.show() 
#5------ZODIAC----
Z=Zodiac['Segment Description'].unique()
for x in Z:
    globals()['df%s' % x] = pd.DataFrame(data=(Zodiac[Zodiac['Segment Description']== x]))
#----I changed the variable names in the dataframes created from the varible explorer
z=['Capricorn%' ,'Sagittarius%','Scorpio%','Libra%','Virgo','Leo%','Cancer%','Gemini%','Taurus%','Aries%','Pisces%','Aquarius%']
Zdf={'Capricorn':Capricorn.Count.tolist(),'Capricorn%':Capricorn.Percentage.tolist(),
    'Sagittarius Count':Sagittarius.Count.tolist(),'Sagittarius%':Sagittarius.Percentage.tolist(),
    'Scorpio Count':Scorpio.Count.tolist(),'Scorpio%':Scorpio.Percentage.tolist(),
    'Libra Count':Libra.Count.tolist(),'Libra%':Libra.Percentage.tolist(),
    'Leo':Leo.Count.tolist(),'Leo%':Leo.Percentage.tolist(),
    'Cancer Count':Cancer.Count.tolist(),'Cancer%':Cancer.Percentage.tolist(),
    'Gemini Count':Gemini.Count.tolist(),'Gemini%':Gemini.Percentage.tolist(),
    'Taurus Count':Taurus.Count.tolist(),'Taurus%':Taurus.Percentage.tolist(),
    'Aquarius Count':Aquarius.Count.tolist(),'Aquarius%':Aquarius.Percentage.tolist(),
    'Aries Count':Aries.Count.tolist(),'Aries%':Aries.Percentage.tolist(),
    'Pisces Count' :Pisces.Count.tolist(),'Pisces%':Pisces.Percentage.tolist(),
     'Virgo Count' :Virgo.Count.tolist(),'Pisces%':Virgo.Percentage.tolist()
    }  
zdata=pd.DataFrame(Zdf,index=appl)
zdata.info
maxiumcount=zdata.idxmax(axis=1)
values=[]
print('Applications where most used by the zodiac group:\n' ,maxiumcount)
type(x)
type(values)
"""PIE CHART SHOWS HOW PEOPLE WITH DIFFERENT ZODIAC SIGNS GET INFLUENCED BY DIFFERENT APPLICATION"""
for x in zdata:
    if '%' in x:
        j=x
        values=zdata[j]
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        labels = appl
        explode = (0.2, 0, 0, 0, 0)
        plt.pie(values, colors=colors, labels= values,explode=explode,
                counterclock=False, shadow=True)
        plt.title(x)
        plt.legend(labels,loc=3)
        plt.show()     