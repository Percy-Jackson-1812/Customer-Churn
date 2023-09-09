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

# In[14]:


x = data.drop(['Churn'], axis = 1)
y = data['Churn']


# In[15]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
onc = OneHotEncoder()
en = LabelEncoder()
x.Gender = en.fit_transform(data.Gender)
x['Location'].replace({'Los Angeles': 'Los Angeles', 'New York': 'New York', 'Miami': 'Miami', 'Chicago': 'Chicago', 'Houston':'Houston'}, inplace = True)
x = pd.get_dummies(x, columns=["Location"], prefix=["Loc."])
x.head()


# <font size = 5>Feature Scaling

# In[16]:


from sklearn.preprocessing import MinMaxScaler
sc_col = ['Age','Subscription_Length_Months','Total_Usage_GB','Monthly_Bill']
sc = MinMaxScaler()
x = pd.DataFrame(sc.fit_transform(x.values), columns=x.columns, index=x.index)


# In[17]:


x.head()


# In[18]:


x.describe()


# <font size = 2>The columns are nomalized with min value being 0 and max value being 1

# <font size = 6>DataPreprocessing<br>
#     <font size = 5>Train - Test Split

# In[55]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)


# <font size = 6>Model Selection

# In[150]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# <font size = 5>Logistic Regression

# In[57]:


lr = LogisticRegression(penalty='l2', solver='newton-cg', C=0.001, random_state = 42)
lr.fit(x_train,y_train)
aclr = lr.score(x_test,y_test)
print("Logistic Regresion Acc. : ",aclr)

pred_lr = lr.predict(x_test)
rep = classification_report(y_test,pred_lr)
print(rep)


# In[61]:


sns.heatmap(confusion_matrix(y_test,pred_lr),annot = True)
print('Precision: %.3f' % precision_score(y_test, pred_lr))
print('Recall: %.3f' % recall_score(y_test, pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_test, pred_lr))
print('F1 Score: %.3f' % f1_score(y_test, pred_lr))


# <font size = 5>AdaBoost Classifier

# In[92]:


abc = AdaBoostClassifier( n_estimators=25, learning_rate=0.5, algorithm='SAMME', random_state=42)
abc.fit(x_train,y_train)
a_pred = abc.predict(x_test)
print("AdaBoost Classifier Acc. : ", metrics.accuracy_score(y_test, a_pred))
print(classification_report(y_test, a_pred))


# In[93]:


sns.heatmap(confusion_matrix(y_test,a_pred),annot = True)
print('Precision: %.3f' % precision_score(y_test,a_pred))
print('Recall: %.3f' % recall_score(y_test,a_pred))
print('Accuracy: %.3f' % accuracy_score(y_test,a_pred))
print('F1 Score: %.3f' % f1_score(y_test,a_pred))


# <font size =5>SVM

# In[112]:


svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
svm.fit(x_train, y_train)
spred = svm.predict(x_test)
print("SVC Acc. : ", metrics.accuracy_score(y_test, spred))
print(classification_report(y_test, spred))


# In[113]:


sns.heatmap(confusion_matrix(y_test,spred),annot = True)
print('Precision: %.3f' % precision_score(y_test,spred))
print('Recall: %.3f' % recall_score(y_test,spred))
print('Accuracy: %.3f' % accuracy_score(y_test,spred))
print('F1 Score: %.3f' % f1_score(y_test,spred))


# <font size = 5>Decision Tree

# In[117]:


dc = DecisionTreeClassifier()
dc.fit(x_train,y_train)
dcpred = dc.predict(x_test)
dtac = dc.score(x_test,y_test)
print("Decision Tree Acc. : ",dtac)
print(classification_report(y_test,dcpred))


# In[118]:


sns.heatmap(confusion_matrix(y_test,dcpred),annot = True)
print('Precision: %.3f' % precision_score(y_test,dcpred))
print('Recall: %.3f' % recall_score(y_test,dcpred))
print('Accuracy: %.3f' % accuracy_score(y_test,dcpred))
print('F1 Score: %.3f' % f1_score(y_test,dcpred))


# <font size = 5>Neural Network

# In[122]:


import tensorflow as tf

m = tf.keras.models.Sequential()
m.add(tf.keras.layers.Dense(units = 10, activation = 'relu'))
m.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))
m.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
m.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ["accuracy"])

m.fit(x_train, y_train, validation_data=(x_test,y_test), epochs = 10, batch_size = 10)


# In[123]:


history = m.history
loss = history.history['loss']
train = history.history['accuracy']
val_loss = history.history['val_loss']
test = history.history['val_accuracy']
plt.rcParams.update({'font.size': 10})

plt.figure(figsize=(10,10))
plt.plot(range(len(train)), train)

plt.plot(range(len(test)), test)
plt.legend(['train', 'test'], fontsize = 12)


# In[124]:


plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(5,5))

plt.plot(range(len(loss)), loss)
plt.plot(range(len(val_loss)), val_loss)
plt.legend(['loss', 'val_loss'], fontsize = 12)


# <font size = 5>Random Forest Classifier

# In[126]:


rfc = RandomForestClassifier(n_estimators = 500, oob_score = True, n_jobs = -1, random_state = 42, max_features = "auto", max_leaf_nodes = 30)
rfc.fit(x_train,y_train)
rpred = rfc.predict(x_test)
print("Random Forest Classifier Acc. : ",metrics.accuracy_score(y_test,rpred))
print(classification_report(y_test,rpred))


# In[127]:


sns.heatmap(confusion_matrix(y_test,rpred),annot = True)
print('Precision: %.3f' % precision_score(y_test,rpred))
print('Recall: %.3f' % recall_score(y_test,rpred))
print('Accuracy: %.3f' % accuracy_score(y_test,rpred))
print('F1 Score: %.3f' % f1_score(y_test,rpred))


# <font size = 6>Ensemble Learning<br>
#     <font size = 5>Voting Classifier

# In[133]:


vote = VotingClassifier(estimators=[('SVM',svm ), ('RFC', rfc), ('ABC',abc)], voting='hard')
vote.fit(x_train,y_train)
vote.score(x_test,y_test)


# <font size = 5>Bagging Classifier

# In[184]:


from sklearn.ensemble import BaggingClassifier

estimator_range = [2,4,6,8,10,12,14,16]

models = []
scores = []

for n in estimator_range:

    clf = BaggingClassifier(n_estimators = n, random_state = 22)
    clf.fit(x_train, y_train)
    models.append(clf)
    scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(x_test)))

plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

plt.show()


# <font size = 3>Thus nearly all the different techniques gives the similar accuracy

# In[ ]:




