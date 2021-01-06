'''
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

Boston = pd.read_csv('d:/data/dataset_0914/BostonHousing.csv')
BH = Boston[['indus', 'dis', 'medv']]

BH = BH.loc[0:499]
kmeans = KMeans(n_clusters=5, random_state=123).fit(BH)

BH.insert(3, 'label', kmeans.labels_.reshape(-1, 1))
BH.head(10)
#print(BH)

#for i in kmeans.cluster_centers_:
    #print(i)


BH_test = Boston[['indus', 'dis', 'medv']]
BH_test = BH_test.loc[500:]
# predict new data
kmeans.predict(BH_test)
BH_test.insert(3, 'label', kmeans.predict(BH_test).reshape(-1, 1))
print(BH_test)

for i in range(5):
    print(BH[BH['label'] == i][['indus', 'dis', 'medv']].mean())

'''

'''
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

pid = pd.read_csv('d:/data/dataset_0914/PimaIndiansDiabetes.csv')
#print(pid)

pid_x = pid[['pregnant','glucose','pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age']]
pid_y = pid[['diabetes']]
pid_y.head(50)
pid_y.loc[pid_y['diabetes'] == 'pos'] = 1
pid_y.loc[pid_y['diabetes'] == 'neg'] = 0
scaler = StandardScaler()
scaler.fit(pid_x)
pid_x_scaled = scaler.transform(pid_x)

print(pid_x)
train_X, test_X, train_y, test_y  = train_test_split(pid_x, pid_y, test_size=0.3, random_state=123)
print(test_y)
train_y = train_y.astype('int')
test_y = test_y.astype('int')

frame1 = [train_X, test_X]
all_x = pd.concat(frame1)
frame2 = [train_y, test_y]
all_y = pd.concat(frame2)

type(pid_y)
type(train_y)
print(list(train_y['diabetes']))
model =  KNeighborsClassifier(n_neighbors=5)
model.fit(train_X, train_y)
pred_y = model.predict(train_X)
help(cross_val_score)

print(pid_x)

scores = cross_val_score(model, train_X, train_y, cv=10)
print('fold acc', scores)
print('mean acc', np.mean(scores))
'''
'''
pred_y = model.predict(train_X)
training_acc = accuracy_score(train_y, pred_y)
print('training accuracy:',training_acc)

pred_y = model.predict(test_X)
test_acc = accuracy_score(test_y, pred_y)
print('test accuracy:',test_acc)
'''

'''
tp = 0
fn = 0
tn = 0
fp = 0
a = list(pred_y)
j = 0
for i in test_y['diabetes']:
    if a[j] == 1 and i == 1:
        tp += 1
    elif a[j] == 0 and i == 1:
        fn += 1
    elif a[j] == 1 and i == 0:
        fp += 1
    else:
        tn += 1
    j += 1

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
precision = tp/(tp+fp)
f1_score = 2*sensitivity*specificity/(sensitivity+specificity)
print('f1 score:',f1_score)
print('precision:',precision)
print('recall:', sensitivity)
'''
'''
test_accuracy=[0]*10
for i in range(10):
    model =  KNeighborsClassifier(n_neighbors=(i+1))
    model.fit(train_X, train_y)
    
    pred_y = model.predict(test_X)
    test_accuracy[i] = accuracy_score(test_y, pred_y)
print('최적k값:',test_accuracy.index(max(test_accuracy)))
'''
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

pid = pd.read_csv('d:/data/dataset_0914/PimaIndiansDiabetes.csv')

pid_x = pid[['pregnant','glucose','pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age']]
pid_y = pid[['diabetes']]

pid_y.loc[pid_y['diabetes'] == 'pos'] = 1
pid_y.loc[pid_y['diabetes'] == 'neg'] = 0

scaler = StandardScaler()
scaler.fit(pid_x)
pid_x_scaled = scaler.transform(pid_x)

acc1=[0]*9
for j in range(1,10):
    kf = KFold(n_splits=j+1, random_state=123)      
    
    model =  KNeighborsClassifier(n_neighbors=5)
    
    acc = np.zeros((j+1,))
    
    i = 0
    
    for train_index, test_index in kf.split(pid_x_scaled):
    
        train_X, test_X = pid_x.iloc[train_index].astype('int'), pid_x.iloc[test_index].astype('int')
        train_y, test_y =  pid_y.iloc[train_index].astype('int'), pid_y.iloc[test_index].astype('int')
    
        model.fit(train_X, train_y)
    
        pred_y = model.predict(test_X)
    
        acc[i] = accuracy_score(test_y, pred_y)
        i += 1
    
    #print("10 fold :", acc)
    acc1[j-1] = np.mean(acc)
print(acc1)
print(acc1.index(max(acc1))+2)    

'''
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


pid = pd.read_csv('d:/data/dataset_0914/PimaIndiansDiabetes.csv')
#print(pid)

pid_x = pid[['pregnant','glucose','pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age']]
pid_y = pid[['diabetes']]
pid_y.head(50)
pid_y.loc[pid_y['diabetes'] == 'pos'] = 1
pid_y.loc[pid_y['diabetes'] == 'neg'] = 0
scaler = StandardScaler()
scaler.fit(pid_x)
pid_x_scaled = scaler.transform(pid_x)

kf = KFold(n_splits=10, random_state=123, shuffle=True)      

# Define learning model
model =  KNeighborsClassifier(n_neighbors=5)

acc = np.zeros((10,))     
i = 0                 

for train_index, test_index in kf.split(pid_x_scaled):
    #print("fold:", i)
    print(train_index)
    print(test_index)
    train_X, test_X = pid_x.iloc[train_index], pid_x.iloc[test_index]
    train_y, test_y =  pid_y.iloc[train_index], pid_y.iloc[test_index]

    # Train the model using the training sets
    model.fit(train_X, train_y)

    # Make predictions using the testing set
    pred_y = model.predict(test_X)
    #print(pred_y)

    # model evaluation: accuracy #############
    acc[i] = accuracy_score(test_y, pred_y)
    #print(accuracy_score(test_y, pred_y))
    #print('Accuracy : {0:3f}'.format(acc[i]))
    i += 1

print("10 fold :", acc)
print("mean accuracy :", np.mean(acc))
'''


'''
kf = KFold(n_splits=10, random_state=123, shuffle=True)       # 5 fold

model =  KNeighborsClassifier(n_neighbors=10)

scores = cross_val_score(model, pid_x, pid_y, cv=10)
print('fold acc', scores)
print('mean acc', np.mean(scores))
'''