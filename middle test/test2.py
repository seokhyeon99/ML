import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('D:/data/middle_test/trainset.csv')
df_X = df.loc[:, df.columns != 'label']
df_y = df['label']

# Backward elimination (Recursive Feature Elimination) 17
######################################################################
from sklearn.feature_selection import RFE

model = LogisticRegression(solver='lbfgs', max_iter=500)
rfe = RFE(model, n_features_to_select=17)
fit = rfe.fit(df_X, df_y)
print("Num Features: %d" % fit.n_features_)
fs = df_X.columns[fit.support_].tolist()   # selected features

#==========================================Prediction============================================
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
'''
df_train_X = df_X[fs]
df_train_y = df_y

model = KNeighborsClassifier(n_neighbors=7)

for train_index, test_index in kf.split(df_train_X]):
    train_X, test_X = df_train_X[train_index], df_train_X[test_index]
    train_y, test_y = df_train_y[train_index], df_train_y[test_index]
    
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)

'''
train_X = df_X[fs]
train_y = df_y

model = KNeighborsClassifier(n_neighbors=7)
model.fit(train_X, train_y)

df = pd.read_csv('D:/data/middle_test/testset.csv')
df_test_X = df.loc[:,:]
df_test_X = df_test_X[fs]
len(df)
len(df_test_X)
pred_y = model.predict(df_test_X)

df_pred = pd.DataFrame(pred_y)
print(df_pred)
df_pred.to_csv('32183164_이석현.csv', header = None, index=None)
