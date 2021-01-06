'''
import numpy as np

def update_w(x, w):
    v = np.dot(x,w)
    e = 1 - v
    print('e:',e)
    print('delta_w:', 0.5*e*x)
    w = w + 0.5 * e * x
    print(w)
    return w

x = np.array([0.5, 0.8, 0.2])
w = np.array([0.4, 0.7, 0.8])
print('initial w:', w)

for i in range(10):
    print('%d번째' %(i+1), end='')
    w = update_w(x,w)
'''
'''
import numpy as np

def SIGMOID(x):
    return 1/(1 + np.exp(-x))

def SOFTMAX(x):
    e_x = np.exp(x)
    return e_x / e_X.sum(axis=0)

x = np.array([0.5,0.8,0.2])
w = np.array([0.4, 0.7, 0.8])
d = 1
alpha = 0.5

for i in range(50):
    v = np.sum(w*x)
    y = SIGMOID(v)
    e = d - y
    print('error', i, e)
    w = w + alpha*y*(1-y)*e*x
'''
'''
from sklearn import datasets
import random
import numpy as np

def SIGMOID(x):
    return 1/(1 + np.exp(-x))

def SLP_SGD(tr_X, tr_y, alpha, rep):
    n = tr_X.shape[1] * tr_y.shape[1]
    random.seed = 123
    w = random.sample(range(1,100), n)
    w = (np.array(w) - 50)/100
    w = w.reshape(tr_X.shape[1], -1)
    
    for i in range(rep):
        for k in range(tr_X.shape[0]):
            x = tr_X[k,:]
            v = np.matmul(x,w)
            y = SIGMOID(v)
            print('y',y)
            e = tr_y[k,:] - y
            temp = np.transpose(np.mat(x))*np.mat(e)
            print('temp',temp)
            w = w + alpha*y*(1-y)*np.array(temp)
        print('error',i,np.mean(e))
    return w

iris = datasets.load_iris()
X = iris.data
target = iris.target

num = np.unique(target, axis=0)
num = num.shape[0]
y = np.eye(num)[target]


##Train
W = SLP_SGD(X, y, alpha=0.01, rep = 600)

##Test
pred = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    v = np.matmul(X[i,:],W)
    y = SIGMOID(v)
    
    pred[i] = np.argmax(y)
    print("target, predict", target[i], pred[i])
print('repeat time = 600')
print("accuracy:", np.mean(pred==target))
'''

from sklearn import datasets
import random
import numpy as np
from sklearn.model_selection import train_test_split

def SIGMOID(x):
    return 1/(1 + np.exp(-x))

def getMatrixMean(matrix):
    sum = [[0]*matrix.shape[2] ]*matrix.shape[1]
    for k in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[2]):
                sum[i][j] += matrix[k][i][j]
    return sum            

def mini_batch(tr_X, tr_y, alpha, epoch, batch_size):
    n = tr_X.shape[1] * tr_y.shape[1]
    random.seed = 123
    w = random.sample(range(1,100), n)
    w = (np.array(w) - 50)/100
    w = w.reshape(tr_X.shape[1], -1)       
    for i in range(epoch):
        w_updated = []
        for k in range(tr_X.shape[0]):
            x = tr_X[k,:]
            v = np.matmul(x,w)
            y = SIGMOID(v)
            #print('y',y)
            e = tr_y[k,:] - y
            temp = np.transpose(np.mat(x))*np.mat(e)
            #print('temp',temp)
            w_updated.append(w + alpha*y*(1-y)*np.array(temp))
        w = getMatrixMean(np.array(w_updated))
        print('error',i,np.mean(e)) 
    return w

def getTarget(test_y):
    target = np.zeros(test_y.shape[0])
    for i in range(len(test_y)):
        for j in range(test_y.shape[1]):
            if test_y[i][j] == 1:
                target[i] = j
    return target

iris = datasets.load_iris()
X = iris.data
#type(iris.target)
target = iris.target
#print(target)
num = np.unique(target, axis=0)
num = num.shape[0]
y = np.eye(num)[target]

train_X, test_X, train_y, test_y = \
    train_test_split(X, y, test_size=0.3,\
                     random_state=1234) 
##Train
W = mini_batch(train_X, train_y, alpha=0.01, epoch = 5, batch_size = 10)
pred = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    v = np.matmul(X[i,:],W)
    y = SIGMOID(v)
    
    pred[i] = np.argmax(y)
    #print("target, predict", target[i], pred[i])
print("training accuracy:", np.mean(pred==target))
##Test
test_target = getTarget(test_y)
pred = np.zeros(test_X.shape[0])
for i in range(test_X.shape[0]):
    v = np.matmul(test_X[i,:],W)
    y = SIGMOID(v)
    
    pred[i] = np.argmax(test_y)
    #print("target, predict", test_target[i], pred[i])
print("test accuracy:", np.mean(pred==test_target))