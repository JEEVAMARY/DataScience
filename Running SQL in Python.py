# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 01:20:58 2021

@author: KBINM18203
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:43:52 2021

@author: KBINM18203
"""

import sys
import pandas as pd
import pymysql
from datetime import datetime
import os
import smtplib
from sqlalchemy import create_engine
from datetime import timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
import time
import logging

da_host_name = 'xx.xx.xx.xxx'
da_host_port = 3306
da_user_name = 'xx.aaa.a'
da_user_pass = '************'
da_db_name = 'xxxxxxxx'
def fetch(query):
        print('query is running')
        db_conn = pymysql.connect(host=da_host_name, port=da_host_port, user=da_user_name, passwd=da_user_pass,
                                  db=da_db_name)

        mydata = pd.read_sql(query, con=db_conn)
        db_conn.close()
        print("sql fetch size - {}".format(len(mydata.index)))
        print(mydata.head())
        return mydata
        
query=""" select * from db.table"""
data=fetch(query)
type(data)
data.columns


#----------------------------------------
# get data from excel file
#----------------------------------------
"""XLS_FILE = "C:\\Users\xyz\\jkl\\pqr.xlsx"
ROW_SPAN = (1, 12)
COL_SPAN = (1, 13)
app = Dispatch("Excel.Application")
app.Visible = True
ws = app.Workbooks.Open(XLS_FILE).Sheets(1)
xldata = [[ws.Cells(row, col).Value 
              for col in range(COL_SPAN[0], COL_SPAN[1])] 
             for row in range(ROW_SPAN[0], ROW_SPAN[1])]
#print xldata
    import numpy as np
a = np.asarray(list(xldata), dtype='object')
print (a)"""

#out file

"""
out=pd.read_excel('C:\\Users\xyz\\jkl\\pqr.xlsx',headers=0)
print(out)
out.columns
data.shape[0]
type(data['xyz'])
type(out['pqr'])
            
#filter
for i in range(data.shape[0]):
    data.loc[data['d_date'][i] >= '2020-04-01 00:00:00']
    """





























