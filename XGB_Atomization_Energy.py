
# coding: utf-8

# In[1]:


import time 
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# In[2]:


start = time.time()
data_frame = pd.read_csv("Thesis_data_training.csv",header=None)
data =  data_frame.values
np.random.shuffle(data)


# In[3]:


x = data[:,:2500]
y = data[:,2500]
x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,test_size=0.2)
KF = KFold(n_splits=5,shuffle=True,random_state=1)


# In[4]:


xgb_reg = xgb.XGBRegressor(objective ='reg:linear', subsample = 0.9, colsample_bytree = 1, learning_rate = 0.01,max_depth = 7, alpha = 20, n_estimators = 10000)


# In[5]:


xgb_reg.fit(x_train_all,y_train_all)


# In[6]:


y_pred = xgb_reg.predict(x_test)


# In[ ]:


rmse   = np.sqrt(mean_squared_error(y_test,y_pred))
end    = time.time()


# In[7]:


with open('RMSE0.txt','w') as file100:
    print(rmse,file=file100)
with open('time0.txt','w') as file101:
    print(end-time,file=file101)

