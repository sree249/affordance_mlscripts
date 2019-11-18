
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


device_list = ['Desktop_Computer','Laptop','Tablet','Smart_Watch','Smart_Phone','Smart_Speaker']


# In[11]:


def participant_analysis(big_table,tmp_list):
    means = []
    for i in tmp_list:
        tmp_value = big_table['actual_use_'+i] - big_table['Predictions_'+i]
        tmp_abs = tmp_value.abs()
        means.append(tmp_abs.mean())
        
    print(tmp_list)
    print(1-np.mean(means))
    print(tmp_value.size)


# In[12]:


counter = 0
tmp_list = []
for i in device_list:
    
    tmp_list.append(i)
    
    if i =='Desktop_Computer':
        big_table = pd.read_csv('predictions_'+str(i)+'.csv')
    else:
        tmp_table = pd.read_csv('predictions_'+str(i)+'.csv')
        big_table = pd.merge(tmp_table,big_table,on='ResponseId')
        
    counter += 1
    
    if counter >1:
        participant_analysis(big_table,tmp_list)
        
        
        
        
    

