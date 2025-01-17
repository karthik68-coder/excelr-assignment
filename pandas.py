#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 


# In[5]:


data =[10,20,30,40,12]
i = ['a','b','c','d','e',]
series=pd.Series(data,i)
print(series)


# In[8]:


data={'a':10,'b':20,'c':30,'d':40}
series=pd.Series(data)
print(series)


# In[11]:


import numpy as np
data=np.array([200,300,400,500])
series=pd.Series(data,index=['(i)','(ii)','(iii)','(iv)'])
print(series)


# In[13]:


data={'Name':['Alice','Bob','Marry'],'Age':[25,30,40],'Country':["USA",'UK','AUS']}
df=pd.DataFrame(data,index=[1,2,3])
print(df)


# In[15]:


import numpy as np
array=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)
df=pd.DataFrame(array,columns=['i','ii','iii'],index=[1,2,3])
print(df)


# In[16]:


import numpy as np
array=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)
df=pd.DataFrame(array)
print(df)


# In[ ]:


#Reading a csv file in to pandas for data analysis


# In[18]:


import numpy as np
data={'Year':['1','2','3','4'],'CGPA':[7.92,7.25,8.2,9.2]}
df=pd.DataFrame(data,index=[1,2,3,4])
print(df)


# In[20]:


iris_df = pd.read_csv("iris.csv")
print(iris_df)


# In[21]:


iris_df.info()


# In[23]:


iris_df.head(10)


# In[25]:


iris_df.tail(7)


# In[26]:


iris_df.describe()


# In[27]:


print(iris_df.shape)
print(iris_df.ndim)
print(iris_df.size)


# In[28]:


#convert dataframe to numpy array


# In[29]:


iris_array = np.array(iris_df)
iris_array


# In[30]:


iris_df


# In[31]:


#print specifies colums  with names


# In[33]:


iris_df[["sepal.length","petal.width"]]


# In[34]:


#print specified roes using iloc(
iris_df.iloc[20,3]


# In[35]:


iris_df.iloc[10:21,:]  


# In[38]:


iris_df.iloc[15:21,0::3]


# In[39]:


iris_df.loc[10:20,"sepal.length":"petal.length"]


# In[ ]:




