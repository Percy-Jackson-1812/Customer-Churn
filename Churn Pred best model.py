#!/usr/bin/env python
# coding: utf-8

# <font size = 6>Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# <font size = 6>Importing dataset

# In[2]:


data = pd.read_excel('customer_churn_large_dataset.xlsx')
data.head()


# <font size = 6>Exploratory Data Analysis

# In[3]:


data.isnull().sum()


# <font size = 3>We dont have any null values

# In[4]:


data.nunique()


# <font size = 3>Since customer id and name are totally unique we will drop these features

# In[5]:


data = data.drop(['CustomerID','Name'], axis =1)


# In[6]:


data.info()


# <font size = 3>There are 2 categorical(Gender and Location) and 5 numerical values (since 'Churn' has numerical eventhough categorical values 0, 1)

# In[7]:


data.describe(include = 'O')


# In[8]:


data.describe()


# <font size = 4>Observation<br>
#     <font size = 2.75>Most people are from Houston (20.157%)
#     <br>Slightly more than 50% of people churned
#     <br>The age of employees ranges from 18 to 70

# <font size = 6>Data Visualization

# <font size = 5>Correlation

# In[9]:


corr = data.corr()
sns.heatmap(corr, annot = True)


# <font size = 3>Observation<br>
#     <font size = 2> All the columns of aur dataset are eually important for prediction

# <font size = 5>Density Distribution

# In[10]:


def density(col,t,l):
    plt.figure(figsize=(6,6))
    sns.distplot(data[col], rug = True)
    plt.title(f'{t}', fontsize = 16)
    plt.xlabel(l,fontsize = 16)
    plt.show()
x = data.select_dtypes(include = 'number')
x = list(x.columns)
for i in x:
    density(i,i,i)


# <font size = 3>Observation<br>
#     <font size = 2>Despite having unique values and even the 'churn' variable, the columns are pretty well balanced

# <font size = 5>Count Plot

# In[11]:


sns.countplot(data=data, x='Gender', hue='Churn', color = 'cyan')


# In[12]:


sns.countplot(data=data, x='Location', hue='Churn', color = 'cyan')


# <font size = 6>Feature Engineering
#     <br><font size = 5>Encoding Categorical Values
#     <br><font size = 3>Gender and Location are categorical features where location has 5 different values and gender has 2 ( Male -> 1, Female -> 0)
#     <br>Label Encoding Gender and One Hot Encoding Location

# In[13]:


x = data.drop(['Churn'], axis = 1)
y = data['Churn']


# In[14]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
onc = OneHotEncoder()
en = LabelEncoder()
x.Gender = en.fit_transform(data.Gender)
x['Location'].replace({'Los Angeles': 'Los Angeles', 'New York': 'New York', 'Miami': 'Miami', 'Chicago': 'Chicago', 'Houston':'Houston'}, inplace = True)
x = pd.get_dummies(x, columns=["Location"], prefix=["Loc."])
x.head()


# <font size = 5>Feature Scaling

# In[15]:


from sklearn.preprocessing import MinMaxScaler
sc_col = ['Age','Subscription_Length_Months','Total_Usage_GB','Monthly_Bill']
sc = MinMaxScaler()
x = pd.DataFrame(sc.fit_transform(x.values), columns=x.columns, index=x.index)


# In[16]:


x.head()


# In[17]:


x.describe()


# <font size = 2>The columns are nomalized with min value being 0 and max value being 1

# <font size = 6>DataPreprocessing<br>
#     <font size = 5>Train - Test Split

# In[18]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)


# <font size = 6>Model Selection

# In[19]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# <font size = 5>AdaBoost Classifier

# In[20]:


abc = AdaBoostClassifier( n_estimators=25, learning_rate=0.5, algorithm='SAMME', random_state=42)
abc.fit(x_train,y_train)
a_pred = abc.predict(x_test)
print("AdaBoost Classifier Acc. : ", metrics.accuracy_score(y_test, a_pred))
print(classification_report(y_test, a_pred))


# In[21]:


sns.heatmap(confusion_matrix(y_test,a_pred),annot = True)
print('Precision: %.3f' % precision_score(y_test,a_pred))
print('Recall: %.3f' % recall_score(y_test,a_pred))
print('Accuracy: %.3f' % accuracy_score(y_test,a_pred))
print('F1 Score: %.3f' % f1_score(y_test,a_pred))


# <font size = 6> Saving the trained model

# In[34]:


import pickle
filen = 'best.sav'
pickle.dump(abc, open(filen, 'wb'))


# In[36]:


load_model = pickle.load(open('best.sav', 'rb'))


# In[ ]:




