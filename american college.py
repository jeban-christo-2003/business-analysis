#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as pl


# In[2]:


df=pd.read_csv("ForbesAmericasTopColleges2019.csv")
df


# In[3]:


df.tail(3)


# In[4]:


df.describe(include='all')


# In[3]:


df.info()


# In[6]:


dups=df.duplicated()
print("Number of duplicated rows = %d" %(dups.sum()))
df[dups]


# In[7]:


df.isnull().sum()


# In[8]:


df['Net Price'].fillna(df['Net Price'].mean(), inplace=True)
df['Average Grant Aid'].fillna(df['Average Grant Aid'].mean(),inplace=True)
df['Alumni Salary'].fillna(df['Alumni Salary'].mean(), inplace=True)
df['Acceptance Rate'].fillna(df['Acceptance Rate'].mean(), inplace=True)
df['SAT Lower'].fillna(df['SAT Lower'].mean(), inplace=True)
df['SAT Upper'].fillna(df['SAT Upper'].mean(), inplace=True)
df['ACT Lower'].fillna(df['ACT Lower'].mean(), inplace=True)
df['ACT Upper'].fillna(df['ACT Upper'].mean(), inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# In[11]:


df.drop('Website',axis=1)


# In[12]:


#EDA 
#outliers Treatment


# In[13]:


sns.boxplot(data=df,x='Undergraduate Population',color='red')


# In[14]:


sns.boxplot(data=df,x='Student Population',color='magenta')


# In[15]:


sns.boxplot(data=df,x='Net Price',color='blue')


# In[16]:


sns.boxplot(data=df,x='Average Grant Aid',color='grey')


# In[17]:


sns.boxplot(data=df,x='Total Annual Cost',color='yellow')


# In[18]:


sns.boxplot(data=df,x='Alumni Salary',color='green')


# In[19]:


sns.boxplot(data=df,x='Acceptance Rate',color='purple')


# In[20]:


sns.boxplot(data=df,x='SAT Lower',color='maroon')


# In[21]:


sns.boxplot(data=df,x='SAT Upper',color='brown')


# In[22]:


sns.boxplot(data=df,x='ACT Lower',color='maroon')


# In[23]:


sns.boxplot(data=df,x='ACT Upper',color='yellow')


# In[24]:


minPopulation,maxPopulation=df['Student Population'].quantile([0.05,0.95])
print(round(minPopulation,2))
print(round(maxPopulation,2))


# In[25]:


df['Student Population']=np.where(df['Student Population']< minPopulation,minPopulation,df['Student Population'])
df['Student Population']=np.where(df['Student Population'] > maxPopulation,maxPopulation,df['Student Population'])


# In[26]:


sns.boxplot(data=df,x='Student Population',color='red')


# In[27]:


minunderPopulation,maxunderPopulation=df['Undergraduate Population'].quantile([0.05,0.95])
print(round(minunderPopulation,2))
print(round(maxunderPopulation,2))


# In[28]:


df['Undergraduate Population']=np.where(df['Undergraduate Population']< minunderPopulation,minunderPopulation,df['Undergraduate Population'])
df['Undergraduate Population']=np.where(df['Undergraduate Population'] > maxunderPopulation,maxunderPopulation,df['Undergraduate Population'])


# In[29]:


sns.boxplot(data=df,x='Undergraduate Population',color='red')


# In[30]:


minNetprice,maxNetprice=df['Net Price'].quantile([0.05,0.95])
print(round(minNetprice,2))
print(round(maxNetprice,2))


# In[31]:


df['Net Price']=np.where(df['Net Price']< minNetprice,minNetprice,df['Net Price'])
df['Net Price']=np.where(df['Net Price'] > maxNetprice,maxNetprice,df['Net Price'])


# In[32]:


sns.boxplot(data=df,x='Net Price',color='red')


# In[33]:


minalumini,maxalumini=df['Alumni Salary'].quantile([0.05,0.95])
print(round(minalumini,2))
print(round(maxalumini,2))


# In[34]:


df['Alumni Salary']=np.where(df['Alumni Salary']< minalumini,minalumini,df['Alumni Salary'])
df['Alumni Salary']=np.where(df['Alumni Salary'] > maxalumini,maxalumini,df['Alumni Salary'])


# In[35]:


sns.boxplot(data=df,x='Alumni Salary',color='red')


# In[4]:


sns.displot(data=df,kde=True)


# In[6]:


sns.histplot(data=df,color='red')


# In[ ]:





# In[ ]:




