#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
import pandas as pd
import numpy as np


# In[31]:


RSQ = pd.read_csv('RSQ.csv')
index = RSQ.iloc[:, 0]
cov = pd.read_csv('covariance_matrix_2022_with_PF.csv')
cov.drop(cov.columns[[0]], axis=1, inplace=True)
W = pd.read_csv('Weights.csv')
Weights=W[index]
TWeights= Weights.transpose()

Weights


# In[33]:


VarCov = ((TWeights@cov)@Weights.to_numpy())/10000
VarCov 


# In[35]:


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# In[37]:


Corr = correlation_from_covariance(VarCov)
Final = np.sqrt(RSQ.iloc[:,-1:])
X = Final@Final.transpose()
Output = pd.DataFrame(X.values*Corr.values, columns=index, index=index)
Output


# In[39]:


# Personal = 'C:\\Users\\ftan\\Documents\\Python\\GCorr\\Output\\'
# Originations = 'S:\\Track Record\\Modeling\\Originations\\IDB Impact Plus\\'
path='S:\\Track Record\\Modeling\\Originations\\GARC USD-2\\GCorr\\'
#test.csv
#IDB Impact Plus_corr_mapping.csv
Output.to_csv(path+'Unicredit - GARC USD-2_corr_mapping.csv')


# In[ ]:





# In[ ]:




