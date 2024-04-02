#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
data = pd.read_csv(r"C:\Users\kumar\Downloads\Oasis infobyte internship\Fraud-Detection-Project3Level2\Online Payment Fraud Detection.csv",encoding='unicode-escape')


# In[2]:


data.shape


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.describe()


# In[6]:


data.info


# In[7]:


data.columns


# In[8]:


data.dtypes


# In[9]:


data.isna().sum()


# In[10]:


data.nameDest.unique()


# In[11]:


data.nameOrig.unique()


# In[12]:


data.nameOrig.value_counts()


# In[13]:


data.nameDest.value_counts()


# In[14]:


data.amount.max()


# In[16]:


labels = data['type'].astype('category').cat.categories.tolist()
counts = data['type'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) 
ax1.axis('equal')
plt.show()


# In[17]:


data.type.value_counts()


# In[18]:


top_ten = data.groupby('nameOrig').type.sum().sort_values(ascending=False)[:10]
top_ten


# In[19]:


data['amount'].mean()


# In[20]:


sns.boxplot(y=data.step)
plt.title('Time of Transaction Profile')
plt.ylim(0,100)
plt.show()


# In[21]:


sns.boxplot(y=data.amount)
plt.title('Amounts Transacted Profile')
plt.ylim(0,1000000)
plt.show()


# In[25]:


sns.boxplot(y=data.isFraud)
plt.title('Fraud Profile')
plt.ylim(-1,1)
plt.show()


# In[27]:


Online_Payment_layout = sns.PairGrid(data, vars = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], hue = 'isFraud')
Online_Payment_layout.map_diag(plt.hist, alpha = 0.6)
Online_Payment_layout.map_offdiag(plt.scatter, alpha = 0.5)
Online_Payment_layout.add_legend()


# In[28]:


sns.barplot(x='amount', y='type', hue= 'isFraud', data=data)
plt.show()


# In[31]:


labels = data['isFraud'].astype('category').cat.categories.tolist()
counts = data['isFraud'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) 
ax1.axis('equal')
plt.show()


# In[32]:


Fraudulent_Transaction = data[data.isFraud ==1]
Not_Fraudulent_Transaction = data[data.isFraud ==0]
print('Fraudulent Transaction: {}'.format(len(Fraudulent_Transaction)))
print('Not Fraudulent Transaction: {}'.format(len(Not_Fraudulent_Transaction)))


# In[33]:


Not_Fraudulent_Transaction.amount.describe()


# In[34]:


Fraudulent_Transaction.amount.describe()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\kumar\Downloads\Oasis infobyte internship\Fraud-Detection-Project3Level2\Online Payment Fraud Detection.csv",encoding='unicode-escape')
sns.catplot(data=data,kind='box')
plt.ylim(0,2000000)


# In[13]:


Fraudulent_Transaction = data[data.isFraud ==1]
Not_Fraudulent_Transaction = data[data.isFraud ==0]
print('Fraudulent Transaction: {}'.format(len(Fraudulent_Transaction)))
print('Not Fraudulent Transaction: {}'.format(len(Not_Fraudulent_Transaction)))


# In[14]:


Not_Fraudulent_Transaction.amount.describe()


# In[15]:


Fraudulent_Transaction.amount.describe()


# In[6]:


import pandas as pd
data = pd.read_csv(r"C:\Users\kumar\Downloads\Oasis infobyte internship\Fraud-Detection-Project3Level2\Online Payment Fraud Detection.csv",encoding='unicode-escape')
Fraudulent_Transaction = data[data.isFraud ==1]
Not_Fraudulent_Transaction = data[data.isFraud ==0]
Non_Fraudulent_Sample = Not_Fraudulent_Transaction.sample(n=1142)
new_dataset = pd.concat([Non_Fraudulent_Sample, Fraudulent_Transaction], axis=0)
new_dataset.head()


# In[7]:


new_dataset.tail()


# In[8]:


new_dataset['isFraud'].value_counts()


# In[9]:


new_dataset.shape


# In[11]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, drop=None,)
encoder_df =  pd.get_dummies(new_dataset, columns=['type','nameOrig','nameDest'], prefix=['type','nameOrig','nameDest'])
encoder_df


# In[12]:


encoder_df.shape


# In[13]:


encoder_df.head()


# In[14]:


encoder_df.tail()


# In[15]:


Y = encoder_df['isFraud']
features = encoder_df.drop('isFraud', axis=1)
X = features
Y.head()


# In[16]:


X.head()


# In[17]:





# In[ ]:




