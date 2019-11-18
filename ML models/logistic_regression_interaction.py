
# coding: utf-8

# In[1]:


from sklearn.preprocessing import PolynomialFeatures


# In[8]:


#X = [[2,2,3]]


# In[9]:


#poly = PolynomialFeatures(interaction_only=True)


# In[10]:


#poly.fit_transform(X)


# In[1]:


from patsy import dmatrices


# In[36]:


import pandas as pd


import sys


# In[55]:


device_list = ['Smart_Phone','Laptop',  'Desktop_Computer','Tablet','Smart_Speaker','Smart_Watch']

head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q4_feat3', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
'Q6_feat3', 'Q6_feat4', 'Q7_feat1', 'Q7_feat2', 'Q7_feat3', 'Q7_feat5', 'Q8_feat1', 'Q8_feat2', 'Q8_feat3', 'Q8_feat5', 'Q9_feat1', 'Q9_feat2',
'Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6', 'Q10_feat7', 'Q11_feat1', 'Q11_feat2', 
'Q11_feat3', 'Q11_feat5', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]

sys.setrecursionlimit(3000)

for device in device_list:
    
    print("current device dataset:", device)
    file = '~/Dropbox/SYR_GAship/afforadance_Study/Datasets/Encoded_files/encoded_Affordance_November4_alldata_kbest'+device+'_data.csv'
    data = pd.read_csv(file)

 
    training_head = []
    training_head.extend(head)
    for e in head:
        cur = device+"_"+e
        training_head.append(cur)
        


    #adding in scenario_variables:
    #scenario_heads = ['location_1','location_2','location_3','Relationship_1','Relationship_2','Relationship_3']
    scenario_heads = ['location_1','location_2','location_3','Relationship_1','Relationship_2','Relationship_3']
    
    count = 0
    
    #generate formula
    
    mystr = "actual_use ~ "
    for scenario in scenario_heads:
        for feat in training_head:
            mystr = mystr + scenario + ':' + feat +'+'
            mystr = mystr + scenario + '*' + feat +'+'
            count = count + 1
  
    training_head.extend(scenario_heads)


    x_data = data[training_head]

    # #x_data = x_data.values.astype(float)



    y_data = data["actual_use"]
    
            
    mystr = mystr[:len(mystr)-1]
    
    mydata = pd.concat([x_data,y_data],axis=1)
    
    # print(mystr)
    
    y, X = dmatrices(mystr,mydata)
    
    # break


# In[58]:


from sklearn.linear_model import LogisticRegression
#clf1 = LogisticRegression()


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[61]:


#clf1.fit(X_train,y_train)


# In[63]:


#clf1.score(X_test, y_test)


# In[54]:


#X_train.item

