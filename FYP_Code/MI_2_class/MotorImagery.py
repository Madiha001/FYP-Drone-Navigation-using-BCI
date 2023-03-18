#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install wyrm


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# In[ ]:


# Load the Preprocessed Data
df = pd.read_csv('preprocessed_MI.csv')


# Print Pandas Dataframe
print(df)
print(df.shape)


# In[ ]:


df_eeg_data = df[['1','2','3','4','5','6','7', '8']]

# Split the data into train and test set
train, test = train_test_split(df_eeg_data, test_size=0.2, random_state=42, shuffle=False)

# print(type(train))
# print(test)


# In[ ]:


# Epoch Data into 10 Second Windows
# 60, 2500, 8

# Convert Test and Training Data into Numpy Array
train = train.to_numpy()
test = test.to_numpy()

# Length of Windows (s)
window_length = 10

# Epoching Training Data
epoched_corrected = []

for filtered in train.T:
    array_epochs = []
    i = 0
    window_size_hz = int(window_length*250) # 10 Seconds

    while(i  < len(train) ):
        array_epochs.append(train[i:i + window_size_hz])
        i = i + window_size_hz 
    
    epoch = array_epochs
    data = np.array(array_epochs) # epoched_train

print(data.shape)


# Epoching Test Data
epoched_corrected = []

for filtered in test.T:
    array_epochs = []
    i = 0
    window_size_hz = int(window_length*250) # 10 Seconds

    while(i  < len(test) ):
        array_epochs.append(test[i:i + window_size_hz])
        i = i + window_size_hz 
    
    epoch = array_epochs
    epoched_test = np.array(array_epochs) # epoched_test

print(epoched_test.shape)


# In[ ]:


# Label Test
df_test = pd.read_csv('Labels/Label_Test.csv')


# Print Pandas Dataframe
df_test = ((df_test).to_numpy()).flatten()

# Shape of the Testing Labels
print(df_test.shape)

# Label Train
df_train = pd.read_csv('Labels/Label_Train.csv')

# Print Pandas Dataframe
df_train = ((df_train).to_numpy()).flatten()

# Shape of the Testing Labels
print(df_train.shape)


# In[ ]:


# Convert the train data into wyrm Data Format

from wyrm import processing as proc


from wyrm.types import Data

# Wyrm Data Attributes: 1.axes 2.names 3.units
# Initialize the First Attribute (Axes: describes the number of dimension of data)
axes = [np.arange(i) for i in data.shape]

# Assign 48 Labels to axes[0]
axes[0] = df_train
axes[2] = [str(i) for i in range(data.shape[2])]

# Initialize the 2nd and 3rd attribute(- Name: Describe the name of each dimension of data - Units: The units of the dimensions)
names = ['Class', 'Time', 'Channel']
units = ['#', 'ms', '#']


dat_train = Data(data=data, axes=axes, names=names, units=units)

dat_train.fs = 250

# Classes (2)
dat_train.class_names = ['hands', 'feet']

# Converting 0s and 1s for binary classification
dat_train.axes[0][dat_train.axes[0] == 2] = 0



print(dat_train)


# In[ ]:


# Get the data var up and running
data = epoched_test

print(data.shape)


# In[ ]:


# Convert the test data into wyrm data format
axes = [np.arange(i) for i in data.shape]
axes[2] = [str(i) for i in range(data.shape[2])]


# Assign 12 Labels to axes[0]
axes[0] = df_test

names = ['Class', 'Time', 'Channel']
units = ['#', 'ms', '#']

dat_test = Data(data=data, axes = axes, names = names, units = units)

dat_test.fs = 250

# Classes (2)
dat_test.class_names = ['hands' , 'feet']


# Converting 0s and 1s for binary classification
dat_test.axes[0][dat_test.axes[0] == 2] = 0



print(dat_test)


# In[ ]:


import matplotlib.pyplot as plt
from wyrm import plot as pt
pt.plot_timeinterval(dat_train)
plt.title("Time interval plot for train data before processing")
plt.savefig('png/Train_before_Process.png', dpi=900, format= "png", bbox_inches="tight")
pt.plot_timeinterval(dat_test)
plt.title("Time interval plot for test data before processing")  
plt.savefig('png/Test_before_Process.png', dpi=900, format= "png", bbox_inches="tight")


# In[ ]:


# Class Average for each channel
pt.plot_channels(dat_train,2)
plt.savefig('png/train_Class_average.png', dpi=600, format= "png", bbox_inches="tight")
pt.plot_channels(dat_test,2)
plt.savefig('png/Test_class_average.png', dpi=600, format= "png", bbox_inches="tight")


# In[ ]:


import matplotlib.pyplot as plt
from wyrm import plot as pt

def prepoc(dat_train, dat_test):
  fsm=dat_train.fs/2
  # filtering the data with 0.9 Hz high and 15 Hz low filter to reduce noise
  #Applying butterworth filters accompanied low pass and high pass filter
  c,a=proc.signal.butter(8,[15/fsm],btype='low')
  dat_train=proc.lfilter(dat_train,c,a)
  c,a=proc.signal.butter(8,0.9/fsm,btype='high')
  dat_train=proc.lfilter(dat_train,c,a)
  c,a=proc.signal.butter(8,[15/fsm],btype='low')
  dat_test=proc.lfilter(dat_test,c,a)
  c,a=proc.signal.butter(8,0.9/fsm,btype='high')
  dat_test=proc.lfilter(dat_test,c,a)
  
  
  # dat_train = proc.subsample(dat_train, 15.625,1)
  # dat_test = proc.subsample(dat_test, 15.625,1)

  pt.plot_timeinterval(dat_train)
  plt.title("Time interval plot for train data after filtering and subsampling")
  plt.savefig('png/Train_After_Process.png', dpi=900, format= "png", bbox_inches="tight")

  pt.plot_timeinterval(dat_test)
  plt.title("Time interval plot for test data after filtering and subsampling")
  plt.savefig('png/Test_After_Process.png', dpi=900, format= "png", bbox_inches="tight")
  
  #applying common spatial pattern
  filt, pattern, _ = proc.calculate_csp(dat_train)
  dat_train = proc.apply_csp(dat_train, filt)
  dat_test = proc.apply_csp(dat_test, filt)
  dat_train = proc.variance(dat_train,1)
  dat_train = proc.logarithm(dat_train)
  
  dat_test = proc.variance(dat_test,1)
  dat_test = proc.logarithm(dat_test)
  dat_train = proc.rectify_channels(dat_train)
  dat_test = proc.rectify_channels(dat_test)
  
  dat_train = proc.square(dat_train)
  dat_test = proc.square(dat_test)
  
  pt.plot_timeinterval(dat_train)
  plt.title("Time interval plot for train data after CSP filter")
  pt.plot_timeinterval(dat_test)
  plt.title("Time interval plot for test data after CSP filter")

  return dat_train,dat_test


# In[ ]:


fvtr, fvte = prepoc(dat_train, dat_test)


# In[ ]:


plt.plot(fvtr.data) #CSP for label 0 and 1 in train data
#plt.show
plt.title("Time interval plot for train data after CSP filter")
plt.savefig('png/Train_After_CSP.png', dpi=600, format= "png", bbox_inches="tight")


# In[ ]:


plt.plot(fvte.data) #CSP for label 0 and 1 in test data
plt.title("Time interval plot for test data after CSP filter")
plt.savefig('png/Test_After_CSP.png', dpi=600, format= "png", bbox_inches="tight")


# ## **LDA Classification**

# In[ ]:


from wyrm import processing as proc

cfy = proc.lda_train(fvtr)
result=proc.lda_apply(fvte,cfy)
result1=(np.sign(result)+1)/2


# In[ ]:


sum=0.0
for i in range(len(result1)):
	if result1[i]==df_test[i]:
		sum=sum+1
lda_acc=sum/(len(result))
print  (lda_acc)


# In[ ]:


# Creating a function to report confusion metrics
def confusion_metrics (conf_matrix):
# save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TP / float(TP + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')


# In[ ]:


from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(df_test, result1)


# In[ ]:


confusion_metrics(cm)


# In[ ]:


from sklearn.metrics import matthews_corrcoef
lda_mcc= matthews_corrcoef(df_test,result1)
lda_mcc


# ## **Train Labels**

# In[ ]:


y_tr=df_train
y_tr[y_tr == 2] = 0


# ## **Random Forest Classifier**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param={
'max_depth':[5,10,20,50,100],

'n_estimators':[2,10,25,50,100]
}

first_xgb = RandomForestClassifier()
clf =GridSearchCV(first_xgb,param, cv=5,verbose=2)
clf.fit(fvtr.data,y_tr)
print(clf.best_params_)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score, precision_score, recall_score
rf = RandomForestClassifier(max_depth = 5, n_estimators=50,n_jobs=1)
rf.fit(fvtr.data, y_tr)
rf_predict= rf.predict(fvte.data)
#rf_probs=rf.predict_proba(xte12)[:,1]
rf_acc=accuracy_score(df_test,rf_predict)
rf_mcc= matthews_corrcoef(df_test,rf_predict)
rf_mcc


# In[ ]:


from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(df_test, rf_predict)


# In[ ]:


confusion_metrics(cm)


# ## **XGBoost Classifier**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

param={
'min_child_weight':[5,10,15],
'max_depth':[5,10,20,50,100],
'learning_rate':[0.001,0.05,0.5,1],
'subsample':[0.8,0.5,0.2],
'n_estimators':[2,10,25,50,100]
}

first_xgb = xgb.XGBClassifier()
clf =GridSearchCV(first_xgb,param, cv=5,verbose=2)
clf.fit(fvtr.data,y_tr)
print(clf.best_params_)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score, precision_score, recall_score
xg = xgb.XGBClassifier(max_depth = 5, n_estimators=25,n_jobs=1,learning_rate=0.05,min_child_weight=5,subsample=0.8)
xg.fit(fvtr.data, y_tr)
xg_predict= xg.predict(fvte.data)
xg_acc=accuracy_score(df_test,xg_predict)
xg_mcc= matthews_corrcoef(df_test,xg_predict)
xg_mcc


# In[ ]:


from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(df_test, xg_predict)


# In[ ]:


# cmdf = pd.DataFrame(cm, 
#             columns = ['Predicted Negative', 'Predicted Positive'],
#             index = ['Actual Negative', 'Actual Positive'])
# cmdf


# In[ ]:


confusion_metrics(cm)


# ## **Support Vector Machine**

# In[ ]:


from sklearn.svm import SVC

param={
'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]
}

svm = SVC()
clf =GridSearchCV(svm,param, cv=5,verbose=2)
clf.fit(fvtr.data,y_tr)
print(clf.best_params_)


# In[ ]:


svm = SVC(C=0.0001, probability=True)
svm.fit(fvtr.data, y_tr)
svm_predict= svm.predict(fvte.data)
svm_acc=accuracy_score(df_test,svm_predict)
svm_mcc= matthews_corrcoef(df_test,svm_predict)
svm_mcc


# In[ ]:


confusion_metrics(cm)


# ## **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression

param={
'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]
}

lr = LogisticRegression()
clf =GridSearchCV(lr,param, cv=5,verbose=2)
clf.fit(fvtr.data,y_tr)
print(clf.best_params_)


# In[ ]:


lr = LogisticRegression(C=0.00001)
lr.fit(fvtr.data, y_tr)
lr_predict= lr.predict(fvte.data)
lr_acc=accuracy_score(df_test,lr_predict)
lr_mcc= matthews_corrcoef(df_test,lr_predict)
lr_mcc


# In[ ]:


confusion_metrics(cm)


# ## **K Nearest Neighbors Classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
param={
'n_neighbors':[2,6,10]
}

kn = KNeighborsClassifier()
clf =GridSearchCV(kn,param, cv=5,verbose=2)
clf.fit(fvtr.data,y_tr)
print(clf.best_params_)


# In[ ]:


kn = KNeighborsClassifier(n_neighbors=10)
kn.fit(fvtr.data, y_tr)
kn_predict= kn.predict(fvte.data)
k_acc=accuracy_score(df_test,kn_predict)
kn_mcc= matthews_corrcoef(df_test,kn_predict)
kn_mcc


# In[ ]:


from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(df_test, kn_predict)

confusion_metrics(cm)


# ## **Naive Bayes Classifier**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(fvtr.data, y_tr)
nb_predict= nb.predict(fvte.data)
nb_acc=accuracy_score(df_test,nb_predict)
nb_mcc= matthews_corrcoef(df_test,nb_predict)
nb_mcc


# In[ ]:


from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(df_test, nb_predict)

confusion_metrics(cm)


# ## **Decision Tree Classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
param={
'max_depth':[2,6,10,40,70,100],
'min_samples_split':[5,10,100,500]
}

dt = DecisionTreeClassifier()
clf =GridSearchCV(dt,param, cv=5,verbose=2)
clf.fit(fvtr.data,y_tr)
print(clf.best_params_)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
dt = DecisionTreeClassifier(max_depth=2,min_samples_split=5)
dt.fit(fvtr.data, y_tr)
dt_predict= dt.predict(fvte.data)
dt_acc=accuracy_score(df_test,dt_predict)
dt_mcc= matthews_corrcoef(df_test,dt_predict)
dt_mcc


# In[ ]:


from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(df_test, dt_predict)

confusion_metrics(cm)


# ## Model Comparison

# In[ ]:


X_bar = ['Random Forest','KNN','SVM','XGboost','Logistic Regression','Naive Bayes','Decision Tree','LDA']
Y_bar= [rf_acc*100,k_acc*100,svm_acc*100,xg_acc*100,lr_acc*100,nb_acc*100,dt_acc*100,lda_acc*100]
import matplotlib.pyplot as plt
plt.barh(X_bar, Y_bar, align='center', color=('#C4EE73','#EEA773', '#73EED9', '#1CB4E3', '#A081DC', '#BE20E7', '#F54B48','#DC93DD'))
plt.xlabel("Performance Accuracy in Percentage")
#plt.show()
plt.savefig('png/Comparison.png', dpi=600, format= "png", bbox_inches="tight")
                      

