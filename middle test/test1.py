import pandas as pd
#import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score

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
'''
print("Selected Features: %s" % fs)
#print("Feature Ranking: %s" % fit.ranking_)

scores = cross_val_score(model, df_X[fs], df_y, cv=5)
print("Acc: "+str(scores.mean()))
'''
#==========================================Algorithm============================================
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
import pandas as pd

# Load dataset
train_X = df_X[fs]
train_y = df_y

# Define learning model  
params = {   'bootstrap': True,
    'max_depth': 80,
    'max_features': 3,
    'min_samples_leaf': 3,
    'min_samples_split': 8,
    'n_estimators': 200}
help(RandomForestClassifier)
model = RandomForestClassifier(n_estimators = 800, max_depth = 90, max_features = 'sqrt',
                               min_samples_split = 5, bootstrap = 'False', random_state=1234)
# Train the model using the training sets
model.fit(train_X, train_y)
help(RandomForestClassifier)

# prepare the dataset
df = pd.read_csv('D:/data/middle_test/testset.csv')

df_test_X = df.loc[:,:]
df_test_X = df_test_X[fs]

pred_y = model.predict(df_test_X)
print(pred_y)
len(pred_y)


df_pred = pd.DataFrame(pred_y)
print(df_pred)
df_pred.to_csv('32183164_이석현.csv', header = None, index=None)

params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

'''
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
print(dataset.data)
print(dataset.target)
'''

