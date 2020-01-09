#!/usr/bin/env python
# coding: utf-8

# In[313]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# # 1. Business understanding

# In[314]:


data=pd.read_csv('D:\As2\Store.csv')
data.head(10)


# In[315]:


print  (str(len(data.index)))
#total number of records


# # 2. Data understanding

# In[316]:


data.describe()


# In[317]:


data.info()
sns.countplot(x="Cus_Churn", data=data)
#dependent variable


# In[318]:


sns.countplot(x="Cus_Churn", hue="A7", data=data)
# A7 represents total sales.Comparision of customer churn wih total sales.


# In[319]:


sns.countplot(x="Cus_Churn", hue="B3", data=data)
# B3 represents average amount spend per visit.Comparision of customer churn wih average amount spend per visit.


# In[320]:


sns.countplot(x="Cus_Churn", hue="C2", data=data)
# C2 represents Average amount spend last year.Comparision of customer churn wih Average amount spend last year.  


# In[321]:


sns.countplot(x="Cus_Churn", hue="I2", data=data)
# I2 represents Average percent of product return.Comparision of customer churn wih Average percent of product return.  


# In[322]:


sns.countplot(x="Cus_Churn", hue="J12", data=data)
# J12 represents Amount spend by customer in three months.Comparision of customer churn wih Amount spend by customer in three months.  


# In[323]:


data["E"].plot.hist()
# E represents response rate.Analysis of each customer response rate.  


# In[324]:


data["D"].plot.hist()
# F represents Number of days customer purchase.Analysis of Number of days customer purchase.


# # 3. Data Preparation

# In[325]:


data.isnull()
# Finding null values in data


# In[326]:


data.isnull().sum()
# No null values found


# In[327]:


sns.boxplot(x='Cus_Churn', y='J12', data=data)


# In[328]:


sns.boxplot(x='Cus_Churn', y='A2', data=data)


# In[329]:


data.head(5)


# In[330]:


data.drop(['A1','A2','A3','A4','A5','A6','A8','B1','B2','C1','C3','I1','I3','J1','J2','J3','J4','J5','J6','J7','J8','J9','J10','J11','J13','J14','J15','J16','K'], axis=1, inplace=True)


# In[331]:


data.head(5)


# In[332]:


data.info()


# In[333]:


data.head(10)


# In[334]:


data.isnull().sum()


# # 4. Training and Testing Data

# In[335]:


X=data.drop("Cus_Churn", axis=1)
y = data["Cus_Churn"]


# In[336]:


from sklearn.model_selection import train_test_split


# In[337]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


# In[338]:


from sklearn.linear_model import LogisticRegression


# In[339]:


logmodel=LogisticRegression() 


# In[340]:


logmodel.fit(X_train, y_train)


# In[341]:


predictions=logmodel.predict(X_train)


# In[342]:


from sklearn.metrics import classification_report


# In[343]:


classification_report(y_test, predictions)


# In[344]:


X.shape


# In[345]:


y.shape


# In[346]:


from sklearn.metrics import confusion_matrix


# In[347]:


confusion_matrix(y_test,predictions)


# # 5. Finding Accuracy

# In[348]:


from sklearn.metrics import accuracy_score


# In[349]:


accuracy_score(y_test,predictions)


# # 6. Calculating Risk

# In[350]:


#Looking unique values
print(data.nunique())


# In[351]:


#Looking the data
print(data.head())


# In[352]:


df_good = data[data["Cus_Churn"] == '1']
df_bad = data[data["Cus_Churn"] == '0']


# In[353]:


import plotly.graph_objs as go


# In[354]:


import plotly.offline as py 
py.init_notebook_mode(connected=True)


# In[355]:


trace0 = go.Bar(
x = data[data["Cus_Churn"]== '1']["A7"].value_counts().index.values,
y = data[data["Cus_Churn"]== '1']["A7"].value_counts().values,
name='Good credit'
)


# In[356]:


trace1 = go.Bar(
x = data[data["Cus_Churn"]== '0']["A7"].value_counts().index.values,
y = data[data["Cus_Churn"]== '0']["A7"].value_counts().values,
name='Bad Credit'
)

data = [trace0, trace1]

layout = go.Layout(
title="Customer Churn"
)


# In[357]:


fig = go.Figure(data=data, layout=layout)


# 

# In[358]:


py.iplot(fig, filename='Customer Churn')

