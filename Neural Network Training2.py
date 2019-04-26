
# coding: utf-8

# In[1]:


import time 
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# In[ ]:


start = time.time()
data_frame = pd.read_csv("Thesis_data_training.csv",header=None)
data =  data_frame.values
np.random.shuffle(data)


# In[ ]:


x = data[:,:2500]
y = data[:,2500]
x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,test_size=0.2)
KF = KFold(n_splits=5,shuffle=True,random_state=1)


# In[ ]:


num_input  = 2500
hiddenlayer1 = 650
hiddenlayer2 = 150
num_output = 1
mean_init = 0
std_init  = 1/np.sqrt(num_input)
lr_init   = 0.001
epoch     = 1
minibatch_size = 10000


# In[ ]:


class Neural(nn.Module):
    def __init__(self):
        super(Neural,self).__init__()
        self.fc1 = nn.Linear(num_input,hiddenlayer1)
        self.fc2 = nn.Linear(hiddenlayer1,hiddenlayer2)
        self.fc3 = nn.Linear(hiddenlayer2,num_output)
        
    def forward(self,x0):
        x1 = F.sigmoid(self.fc1(x0))
        x2 = F.sigmoid(self.fc2(x1))
        x3 = self.fc3(x2)
        return(x3)

neu = Neural()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(neu.parameters(),lr=lr_init)


# In[ ]:


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal(m.weight,mean=mean_init,std=std_init)
        torch.nn.init.normal(m.bias,mean=mean_init,std=std_init)


# In[ ]:


def cross_validation(x_tr,y_tr,x_te,y_te,c,mb_size,epoch):
    if c==0:
        ff_txt = "Loss_all_training_data.txt"
        sf_txt = "Loss_all_test_data.txt"
        ff     = "file0"
        sf     = "file1"
                
    elif c==1:
        ff_txt = "Loss_of_training_1.txt"
        sf_txt = "Loss_of_validation_1.txt"
        ff     = "file2"
        sf     = "file3"
        
    elif c==2:
        ff_txt = "Loss_of_training_2.txt"
        sf_txt = "Loss_of_validation_2.txt"
        ff     = "file4"
        sf     = "file5"
                
    elif c==3:
        ff_txt = "Loss_of_training_3.txt"
        sf_txt = "Loss_of_validation_3.txt"
        ff     = "file6"
        sf     = "file7"
                
    elif c==4:
        ff_txt = "Loss_of_training_4.txt"
        sf_txt = "Loss_of_validation_4.txt"
        ff     = "file8"
        sf     = "file9"
                
    else:
        ff_txt = "Loss_of_training_5.txt"
        sf_txt = "Loss_of_validation_5.txt"
        ff     = "file10"
        sf     = "file11"
        
    mb_quant   = len(y_tr)//mb_size
    loss_train = np.zeros((epoch*mb_quant))
    loss_test  = np.zeros((epoch*mb_quant))
    with open(sf_txt,'w') as sf:
        with open(ff_txt,'w') as ff:
            for j in range(0,epoch):
                for k in range(0,mb_quant):    
                        
                    index_mini_low  = k*mb_size
                    index_mini_high = (k+1)*mb_size
                    loss_index_mini = j*mb_quant+k
                    x_train_mini    = x_tr[index_mini_low:index_mini_high]
                    y_train_mini    = y_tr[index_mini_low:index_mini_high]
                    X_nn            = Variable(torch.Tensor(x_train_mini).float())
                    Y_nn            = Variable(torch.Tensor(y_train_mini).float())
                    for g in optimizer.param_groups:
                        g['lr'] = lr_init*np.exp(-1*loss_index_mini*0.000125)
            
                    #feedforward + backpropagation
                    optimizer.zero_grad()
                    out  = neu(X_nn)
                    Y_nn = torch.unsqueeze(Y_nn,1)
                    loss = criterion(out,Y_nn)
                    
                    X_nn_test = Variable(torch.Tensor(x_te).float())
                    Y_nn_test = Variable(torch.Tensor(y_te).float())
                    out_test  = neu(X_nn_test)
                    Y_nn_test = torch.unsqueeze(Y_nn_test,1)
                    loss_test_variable         = criterion(out_test,Y_nn_test)
                    loss_test[loss_index_mini] = loss_test_variable
                    print(loss_test[loss_index_mini],file=sf)
                        
                    X_nn_train = Variable(torch.Tensor(x_tr).float())
                    Y_nn_train = Variable(torch.Tensor(y_tr).float())
                    out_train  = neu(X_nn_train)
                    Y_nn_train = torch.unsqueeze(Y_nn_train,1)
                    loss_train_variable                = criterion(out_train,Y_nn_train)
                    loss_train[loss_index_mini] = loss_train_variable
                    print(loss_train[loss_index_mini],file=ff)
                        
                    loss.backward()
                    optimizer.step()


# In[ ]:


count = 0
for train,validation in KF.split(x_train_all):
    
    count += 1
    neu.apply(init_weights)
    x_train, x_validation = x_train_all[train], x_train_all[validation]
    y_train, y_validation = y_train_all[train], y_train_all[validation]
    cross_validation(x_train,y_train,x_validation,y_validation,count,minibatch_size,epoch)


# In[ ]:


neu.apply(init_weights)
cross_validation(x_train_all,y_train_all,x_test,y_test,0,minibatch_size,epoch)


# In[ ]:


end = time.time()
with open('computation_time.txt','w') as file100:
    print(end-start,file=file100)

