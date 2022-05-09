#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt


# In[1]:


cd D:\FT\python\Ml\Assignment


# Part-1: Data Exploration and Pre-processing

# In[3]:


# 1) load the given dataset
data=pd.read_excel("Python_Project_7_KNN.xlsx")


# In[4]:


# 2) Check for the null values
data.isnull().sum()


# In[6]:


# 3) Get basic information from data 
data.info()


# In[7]:


# 4) Describe the dataset
data.describe()


# In[9]:


# 5) Display scatterplot between age & Total work Experience
plt.scatter(data['Age'],data['Total Work Experience'])
plt.show()


# In[11]:


# 6) Display box plot for age 
plt.boxplot(data['Age'])
plt.show()


# In[13]:


# 7) Display box plot for Cibil score 
plt.boxplot(data['Cibil score'])
plt.show()


# In[14]:


# 8) Create target and features data where target is Total bounces past12months
x=data.drop('Total bounces past12months',axis=1)
y=data['Total bounces past12months']


# Part-2: Working with Mode

# In[17]:


# 1) Split data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[1]:


# 2) Create a KNN classifier between Features and target data
from sklearn.neighbors import KNeighborsClassifier


# In[19]:


model=KNeighborsClassifier()


# In[20]:


model.fit(x_train,y_train)


# In[23]:


# 3) Display the test score
model.score(x_test,y_test)


# In[24]:


# 4) Display the training score 
model.score(x_train,y_train)


# In[26]:


pred=model.predict(x_test)


# In[28]:


# 5) Print the accuracy score 
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[30]:


# 6) Try 1 to 14 k values for classifier
train_score=[]
test_score=[]
for i in range(1,15):
    model=KNeighborsClassifier(i)
    model.fit(x_train,y_train)
    
    train_score.append(model.score(x_train,y_train))
    test_score.append(model.score(x_test,y_test))


# In[31]:


# 7) Display training and testing score for all the 1 to 14 k values
train_score


# In[32]:


test_score

