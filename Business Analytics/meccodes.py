# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import numpy as np
os.getcwd()
os.chdir(r'C:\Users\louij\OneDrive\Documents\BA')
csv=pd.read_csv('MEC_Retail-SalesMarketing_-ProfitCost.csv')
csv.isnull().sum()
csv.columns
csv.drop(['Unnamed: 15','Unnamed: 14'],axis=1,inplace=True)
na=csv[csv[' Revenue '].isnull()]
na.columns
na['Order method type'].value_counts()
na['Order method type'].value_counts().plot(kind='pie',autopct='%.2f')
na['Retailer country'].value_counts().plot(kind='pie')
na['Product type'].value_counts().plot(kind='pie',fontsize=7)
na['Product line'].value_counts().plot(kind='pie')
na['Year'].value_counts().plot(kind='pie')
mec=csv.dropna()
mec=csv.dropna(subset=[' Revenue '])
mec.isnull().sum()
col=mec.columns.tolist()
mec.columns
mec['Product line'].unique()
mec[' Revenue ']=mec[' Revenue '].str.strip()
mec[' Revenue ']=mec[' Revenue '].str.replace(',','').replace('-','')
mec[' Revenue ']=pd.to_numeric(mec[' Revenue '])

mec[' Planned revenue ']=mec[' Planned revenue '].str.strip()
mec[' Planned revenue ']=mec[' Planned revenue '].str.replace(',','').replace('-','')
mec[' Planned revenue ']=pd.to_numeric(mec[' Planned revenue '])
    
mec[' Quantity ']=mec[' Quantity '].str.strip()
mec[' Quantity ']=mec[' Quantity '].str.replace(',','').replace('-','')
mec[' Quantity ']=pd.to_numeric(mec[' Quantity '])

mec[' Unit price ']=mec[' Unit price '].str.strip()
mec[' Unit price ']=mec[' Unit price '].str.replace(',','').replace('-','')
mec[' Unit price ']=pd.to_numeric(mec[' Unit price '])

mec[' Unit sale price ']=mec[' Unit sale price '].str.strip()
mec[' Unit sale price ']=mec[' Unit sale price '].str.replace(',','').replace('-','')
mec[' Unit sale price ']=pd.to_numeric(mec[' Unit sale price '])

mec[' Product cost ']=mec[' Product cost '].str.strip()
mec[' Product cost ']=mec[' Product cost '].str.replace(',','').replace('-','')
mec[' Product cost ']=pd.to_numeric(mec[' Product cost '])


#Size of pie
print("Total Revenue is for 2015-2018 "+str(mec[' Revenue '].sum()))

#By_year

year=mec.Year.unique().tolist()
yearR=[]
for item in year:
    filter=(mec['Year']== item)
    x=mec[filter]
    k=str(item) 
    yearR.append(x[' Revenue '].sum())
yeaD= dict(zip(year, yearR))
import matplotlib.pylab as plt
lists = sorted(yeaD.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.show()

p_l=mec['Product line'].unique().tolist()
p_ls=[]
p_d=[]

for item in p_l:
    filter=(mec['Product line']== item)
    x=mec[filter]
    j=(x[' Revenue '].sum())
    k=(x[' Planned revenue '].sum())
    p_ls.append(j)
    p_d.append(j-k)
d = {'list': p_l , 'diffrence':p_d} 
d=pd.DataFrame(d)
import matplotlib.pyplot as pyplot
import matplotlib.pylab as plt
pyplot.axis("equal")
pyplot.pie((p_ls),labels=(p_l),autopct=None)
pyplot.show()
#planned vs actual revenue 
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
g=sns.pointplot(x="list", y="diffrence",data=d)
g.set_xticklabels(labels=p_l,rotation=30)   
g.set_title("Planned vs actual revenue")




f=mec['Year']== 2018
y2018=mec[f]

f2=mec['Year']== 2017
y2017=mec[f2]

f3=mec['Year']== 2016
y2016=mec[f3]

f4=mec['Year']== 2015
y2015=mec[f4]

y2015.columns

#Ansoff


pt7=y2017['Product type'].unique().tolist()
pt8=y2018['Product type'].unique().tolist()
pt6=y2016['Product type'].unique().tolist()
pt5=y2015['Product type'].unique().tolist()
#product development:
pd6=set(pt6)-set(pt5)
print("product development in year 2016 :"+str(pd6))
pd7=set(pt7)-set(pt6)
pd8=set(pt8)-set(pt7)
#market development:
pc7=y2017['Retailer country'].unique().tolist()
pc8=y2018['Retailer country'].unique().tolist()
pc6=y2016['Retailer country'].unique().tolist()
pc5=y2015['Retailer country'].unique().tolist()
pdm6=set(pc6)-set(pc5)
print("market development in year 2016 :"+str(pdm6))
pdm7=set(pc7)-set(pc6)
pdm8=set(pc8)-set(pc7)#no market development
#market penetration
#change in order type method more shift towards web decline of fax,telephone,mail
y2015['Order method type'].value_counts().plot(kind='pie',autopct='%.2f',title="2015")
y2016['Order method type'].value_counts().plot(kind='pie',autopct='%.2f',title="2016")
y2017['Order method type'].value_counts().plot(kind='pie',autopct='%.2f',title="2017")
y2018['Order method type'].value_counts().plot(kind='pie',autopct='%.2f',title="2018")
#diversification
mkt= (y2016['Retailer country']==('Australia' or 'Switzerland'))
mkt16=y2016[mkt]
div=set(mkt16['Product type'].unique().tolist())
pd6-div
print("huge diversification")

#detailed analysis year 2018 and 2017
#product share:
p7=y2017['Product line'].unique().tolist()
pv7=[]
for item in p7:
    filter=(y2017['Product line']== item)
    x=y2017[filter]
    pv7.append(x[' Revenue '].sum())
y17= dict(zip(p7, pv7))
p8=y2018['Product line'].unique().tolist()
pv8=[]
for item in p8:
    filter=(y2018['Product line']== item)
    x=y2018[filter]
    pv8.append(x[' Revenue '].sum())
y18= dict(zip(p8, pv8))
pyplot.axis("equal")
pyplot.pie((pv8),labels=(p8),autopct=None)
pyplot.show()
pyplot.axis("equal")
pyplot.pie((pv7),labels=(p7),autopct=None)
pyplot.show()

#bcg
growth=[]
share=[]
len(p8)
len(p7)
import itertools  
for (a, b, c) in zip(p7, pv7, pv8): 
     print (a,b,c)
     growth.append(((c-b)/b)*100)
    
Revenue18=sum(pv8)
for (a,b) in zip (p8,pv8):
    share.append((b/Revenue18)*100)
    

bcg = {'Product Line': p7 , 'Growth':growth, 'Share':share} 
bcg=pd.DataFrame(bcg)
#BCG 
import seaborn as sns
sns.set(style="white")
sns.relplot(x="Growth", y="Share", hue="Product Line", size="Share",
            sizes=(40, 400), alpha=.5, height=6, data=bcg)


y2018.columns
print("Total Revenue is for 2018 "+str(y2018[' Revenue '].sum()))
import matplotlib.pylab as plt
#ptype and revenue
def revenue(column,meth):
    ptype8=y2018[column].unique().tolist()
    pshare8=[]
    ptype7=y2017[column].unique().tolist()
    pshare7=[]
    for item in ptype8:
        filter=(y2018[column]== item)
        x=y2018[filter]
        pshare8.append(x[' Revenue '].sum())
    for item in ptype7:
        filter=(y2017[column]== item)
        x=y2017[filter]
        pshare7.append(x[' Revenue '].sum())
        
   
    if (meth=='line'):
        plt.style.use('classic')
        fig, ax = plt.subplots()
            # unpack a list of pairs into two tuples
        ax.plot(ptype8, pshare8,label="2018")
        ax.plot(ptype7, pshare7,label="2017")
        leg = ax.legend();
        plt.xticks(rotation='vertical')
        plt.show()
    else:
        plt.scatter(ptype8, pshare8,label="2018",c='red',alpha=0.5)
        plt.scatter(ptype7, pshare7,label="2017",c='blue',alpha=0.9)
        plt.xticks(rotation='vertical')
        leg = plt.legend();
        plt.show()

        
        
        
revenue('Product type','line')#no big diffrence in share

revenue('Order method type','scatter')#reduction in web
revenue('Retailer country','scatter')#high fall in US markets

#market share by country


"""price and quantity"""
#US

ptype8=y2018['Product type'].unique().tolist()

pp8=[]
pq8=[]
ptype7=y2017['Product type'].unique().tolist()
pp7=[]
pq7=[]

for item in ptype8:
    filter=(y2018['Product type']== item)
    x=y2018[filter]
    pq8.append(x[' Quantity '].sum())
    pp8.append(x[' Unit price '].mean())
for item in ptype7:
    filter=(y2017['Product type']== item)
    x=y2017[filter]
    pq7.append(x[' Quantity '].sum())
    pp7.append(x[' Unit price '].mean())


f=y2017['Retailer country']=='United States'
u7=y2017[f]
f6=y2016['Retailer country']=='United States'
u6=y2016[f6]
f5=y2015['Retailer country']=='United States'
u5=y2015[f5]
f8=y2018['Retailer country']=='United States'
u8=y2018[f8]
u8.columns
#product type AND GROWTH
def revenueus(column,meth):
    ptype8=u8[column].unique().tolist()
    pshare8=[]
    ptype7=u7[column].unique().tolist()
    pshare7=[]
    for item in ptype8:
        filter=(u8[column]== item)
        x=u8[filter]
        pshare8.append(x[' Revenue '].sum())
    for item in ptype7:
        filter=(u7[column]== item)
        x=u7[filter]
        pshare7.append(x[' Revenue '].sum())
        
   
    if (meth=='line'):
        plt.style.use('classic')
        fig, ax = plt.subplots()
            # unpack a list of pairs into two tuples
        ax.plot(ptype8, pshare8,label="2018")
        ax.plot(ptype7, pshare7,label="2017")
        leg = ax.legend();
        plt.xticks(rotation='vertical')
        plt.show()
    else:
        plt.scatter(ptype8, pshare8,label="2018",c='red',alpha=0.5)
        plt.scatter(ptype7, pshare7,label="2017",c='blue',alpha=0.9)
        plt.xticks(rotation='vertical')
        leg = plt.legend();
        plt.show()
revenueus('Product type','line')#no big diffrence in share
revenueus('Order method type','scatter')#
pl=u7['Product line'].unique().tolist()
u7['lossturnout']=u7[' Planned revenue ']-u7[' Revenue ']
g = sns.PairGrid(u7,vars=["lossturnout", " Quantity "],hue="Product line")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();
g = sns.PairGrid(u8,vars=[" Revenue ", " Product cost "],hue="Product line")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();
g = sns.PairGrid(u8,vars=[" Unit price ", " Quantity "],hue="Product line")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();
plt.figure(figsize=(5,5))
#########
byp7=u7.groupby(['Product line']).mean()
byp7['pl'] = byp7.index
byp8=u8.groupby(['Product line']).mean()
byp8['pl'] = byp8.index
byp6=u6.groupby(['Product line']).mean()
byp6['pl'] = byp6.index
byp5=u5.groupby(['Product line']).mean()
byp5['pl'] = byp5.index
plt.style.use('classic')
fig, ax = plt.subplots()
plt.title('US Product Overview')         # unpack a list of pairs into two tuples
ax.plot(byp7['pl'], byp7[' Revenue '],label="2018")
ax.plot(byp8['pl'], byp8[' Revenue '],label="2017")
ax.plot(byp7['pl'], byp7[' Quantity ']*100,label="2017q",marker='o')
ax.plot(byp8['pl'], byp8[' Quantity ']*100,label="2018q",marker='o')
ax.plot(byp7['pl'], byp7[' Unit cost ']*10000,label="2018unitcost",marker='+')
ax.plot(byp8['pl'], byp8[' Unit cost ']*10000,label="2017unitcost",marker='+')
ax.plot(byp6['pl'], byp6[' Unit cost ']*10000,label="2016unitcost",marker='+')
ax.plot(byp5['pl'], byp5[' Unit cost ']*10000,label="2015unitcost",marker='+')
ax.plot(byp6['pl'], byp6[' Revenue '],label="2016")
ax.plot(byp5['pl'], byp5[' Revenue '],label="2015")

leg = ax.legend();
plt.xticks(rotation='vertical')
plt.show()
p_ls=[]
p_d=[]
pl8=byp8['pl'].tolist()
difference=byp8[' Planned revenue ']-byp8[' Revenue ']
difference=difference.tolist()
d = {'list':pl8  , 'diffrence':difference} 
d=pd.DataFrame(d)
sns.set(style="ticks", color_codes=True)
g=sns.pointplot(x="list", y="diffrence",data=d)
g.set_xticklabels(labels=p_l,rotation=30)   
g.set_title("Planned vs actual revenue 18")

pl7=byp7['pl'].tolist()
difference=byp7[' Planned revenue ']-byp7[' Revenue ']
difference=difference.tolist()
d = {'list':pl8  , 'diffrence':difference} 
d=pd.DataFrame(d)
sns.set(style="ticks", color_codes=True)
g=sns.pointplot(x="list", y="diffrence",data=d)
g.set_xticklabels(labels=p_l,rotation=30)   
g.set_title("Planned vs actual revenue 17")
sns.heatmap(byp7.corr());
sns.heatmap(byp8.corr());
sns.heatmap(u8.corr());
sns.heatmap(u6.corr());
sns.heatmap(u5.corr());
    check=[ ' Planned revenue ', ' Product cost ',
           ' Quantity ', ' Unit cost ', ' Unit price ', ' Gross profit ',
           ' Unit sale price ']
