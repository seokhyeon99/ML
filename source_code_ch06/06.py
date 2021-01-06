
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

df = pd.read_csv('D:/data/dataset_0914/PimaIndiansDiabetes.csv')

df_X = df.loc[:, df.columns != 'diabetes']
df_y = df[['diabetes']]

df_y[df_y['diabetes'] == 'pos'] = 1
df_y[df_y['diabetes'] == 'neg'] = 0

kf = KFold(n_splits=10, random_state=123, shuffle=True)

param_n_estimators = [100, 200, 300, 400, 500]
param_max_features = [1, 2, 3, 4, 5]
accuracy = []

for j in range(len(param_n_estimators)):
    for k in range(len(param_max_features)):
        model = RandomForestClassifier(n_estimators=param_n_estimators[j], max_features = param_max_features[k], random_state=1234)
        
        acc = np.zeros(10)
        i = 0
        
        for train_index, test_index in kf.split(df_X):
            train_X, test_X = df_X.iloc[train_index].astype('int'), df_X.iloc[test_index].astype('int')
            train_y, test_y = df_y.iloc[train_index].astype('int'), df_y.iloc[test_index].astype('int')
            
            model.fit(train_X, train_y)
            
            pred_y = model.predict(test_X)
            
            acc[i] = accuracy_score(test_y, pred_y)
            i += 1
        accuracy.append(np.mean(acc))

l = 0
for j in range(len(param_n_estimators)):
    for k in range(len(param_max_features)):
        print('n_estimators:', param_n_estimators[j], 'max_features', param_max_features[k], 'accuracy:', accuracy[l])
        l += 1
print('max:', max(accuracy))

'''
# Support Vector Machine

'''
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

df = pd.read_csv('D:/data/dataset_0914/PimaIndiansDiabetes.csv')

df_X = df.loc[:, df.columns != 'diabetes']
df_y = df[['diabetes']]

df_y[df_y['diabetes'] == 'pos'] = 1
df_y[df_y['diabetes'] == 'neg'] = 0

kf = KFold(n_splits=10, random_state=123, shuffle=True)
param_kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
accuracy = []
for j in range(len(param_kernel)):
    print('j', j)
    model = svm.SVC(kernel=param_kernel[j])

    acc = np.zeros(10)
    i = 0

    for train_index, test_index in kf.split(df_X):
        print('i', i)
        train_X, test_X = df_X.iloc[train_index].astype('int'), df_X.iloc[test_index].astype('int')
        train_y, test_y = df_y.iloc[train_index].astype('int'), df_y.iloc[test_index].astype('int')

        model.fit(train_X, train_y)

        pred_y = model.predict(test_X)

        acc[i] = accuracy_score(test_y, pred_y)
        i += 1
        
    accuracy.append(np.mean(acc))

for k in range(len(accuracy)):
    print(param_kernel[i],'accuracy',accuracy[k])
    '''
    '''
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

df = pd.read_csv('D:/data/dataset_0914/PimaIndiansDiabetes.csv')

df_X = df.loc[:, df.columns != 'diabetes']
df_y = df[['diabetes']]

df_y[df_y['diabetes'] == 'pos'] = 1
df_y[df_y['diabetes'] == 'neg'] = 0

kf = KFold(n_splits=10, random_state=123, shuffle=True)

model = svm.SVC(kernel='poly', degree=2)

acc = np.zeros(10)
i = 0

for train_index, test_index in kf.split(df_X):
    train_X, test_X = df_X.iloc[train_index].astype('int'), df_X.iloc[test_index].astype('int')
    train_y, test_y = df_y.iloc[train_index].astype('int'), df_y.iloc[test_index].astype('int')
    
    model.fit(train_X, train_y)
    
    pred_y = model.predict(test_X)
    
    acc[i] = accuracy_score(test_y, pred_y)
    i += 1

print('accuracy:', np.mean(acc))
'''

# DecisionTree

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import pydot

df = pd.read_csv('D:/data/dataset_0914/PimaIndiansDiabetes.csv')

df_X = df.loc[:, df.columns != 'diabetes']
df_y = df[['diabetes']]

df_y[df_y['diabetes'] == 'pos'] = 1
df_y[df_y['diabetes'] == 'neg'] = 0

kf = KFold(n_splits=10, random_state=123, shuffle=True)

model =  DecisionTreeClassifier(random_state=1234)

acc = np.zeros(10)
i = 0

for train_index, test_index in kf.split(df_X):
    train_X, test_X = df_X.iloc[train_index].astype('int'), df_X.iloc[test_index].astype('int')
    train_y, test_y = df_y.iloc[train_index].astype('int'), df_y.iloc[test_index].astype('int')
    
    model.fit(train_X, train_y)
    
    pred_y = model.predict(test_X)
    
    acc[i] = accuracy_score(test_y, pred_y)
    i += 1

print('accuracy:', np.mean(acc))