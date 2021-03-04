# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 02:37:23 2021

@author: KBINM18203
"""

import requests
import json
import pandas as pd
import re
url = "https://flipkart-reviews.p.rapidapi.com/v1.0/reviews"

querystring = {"page":"1","url":"https://www.flipkart.com/samsung-galaxy-f41-fusion-blue-128-gb/p/itm4769d0667cdf9?pid=MOBFV5PWG5MGD4CF&lid=LSTMOBFV5PWG5MGD4CFZ8YQJZ&marketplace=FLIPKART&srno=b_1_1&otracker=CLP_Filters&fm=organic&iid=aa5bdf83-0b88-4dc1-882b-6cd209759dec.MOBFV5PWG5MGD4CF.SEARCH&ppt=clp&ppn=mobile-phones-store&ssid=q4tmpjpjgg0000001614771310160\""}

headers = {
            'x-rapidapi-key': "76483ef484mshc54f7f4dd2d6f63p13da0djsndab1827f4a8c",
            'x-rapidapi-host': "flipkart-reviews.p.rapidapi.com"
            }

def getdata(url,querystring,headers):
             
        response = requests.request("GET", url, headers=headers, params=querystring)
        data=response.json()
        return (data)
        # print(data)
        #print(data[2])
        # type(data[2]) -->dict
        # type(data) --->list
      
data=getdata(url,querystring,headers)            
col=[]
for i in data[2]:
                col.append(i)       
print (col)
df=pd.DataFrame(columns=col)   
def adddata(data,df):
    for j in range(2,len(data)):
            df=df.append(data[j],ignore_index=True)
    return(df)  

    
#print(data[1]['total_pages'])
n=max(re.findall('[0-9]+',data[1]['total_pages'].replace(',','')))
        
if (int(n)>=1000):
                x=999
else:
                x=n
k=1            
for k in range (x):
    querystring = {"page":str(k),"url":"https://www.flipkart.com/samsung-galaxy-f41-fusion-blue-128-gb/p/itm4769d0667cdf9?pid=MOBFV5PWG5MGD4CF&lid=LSTMOBFV5PWG5MGD4CFZ8YQJZ&marketplace=FLIPKART&srno=b_1_1&otracker=CLP_Filters&fm=organic&iid=aa5bdf83-0b88-4dc1-882b-6cd209759dec.MOBFV5PWG5MGD4CF.SEARCH&ppt=clp&ppn=mobile-phones-store&ssid=q4tmpjpjgg0000001614771310160\""}
    data=getdata(url,querystring,headers)
    df=df=adddata(data,df)
 






    
print(df)



    
print(df)
   
  

