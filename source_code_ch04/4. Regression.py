'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

Boston = pd.read_csv('d:/data/dataset_0914/BostonHousing.csv')
#print(Boston)

lstat = Boston['lstat']
medv  = Boston['medv']

lstat = np.array(lstat).reshape(506,1)
medv  = np.array(medv).reshape(506,1)

model =  LinearRegression()
model.fit(lstat, medv)

pred_y = model.predict(lstat)
#print(pred_y)

print('Coefficients: {0:.2f}, Intercept {1:.3f}'
      .format(model.coef_[0][0], model.intercept_[0]))

# The mean squared error
print('Mean squared error: {0:.2f}'.format(mean_squared_error(medv, pred_y))) 

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(medv, pred_y))

# Plot outputs
plt.scatter(lstat, medv,  color='black')
plt.plot(lstat, medv, color='blue', linewidth=0.1)

plt.xlabel('lstat')
plt.ylabel('medv')

plt.show()

print(model.predict([[2.0]]))
print(model.predict([[3.0]]))
print(model.predict([[4.0]]))
print(model.predict([[5.0]]))
'''
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

Boston = pd.read_csv('d:/data/dataset_0914/BostonHousing.csv')
#print(Boston)

ptratio = Boston['ptratio']
tax = Boston['tax']
rad = Boston['rad']
lstat = Boston['lstat']
medv  = Boston['medv']

Boston_X = Boston[['ptratio','tax','rad','lstat']]
Boston_y = Boston['medv']

model =  LinearRegression()
model.fit(Boston_X, Boston_y)
pred_y = model.predict(Boston_X)
#print(pred_y)

print('Coefficients: {0:.2f},{1:.2f},{2:.2f},{3:.2f} Intercept {4:.3f}'\
      .format(model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3],\
              model.intercept_))

print('Mean squared error: {0:.2f}'.\
      format(mean_squared_error(Boston_y, pred_y))) 

print('Coefficient of determination: %.2f'
      % r2_score(Boston_y, pred_y))

print(model.predict([[14,296,1,2.0]]))
print(model.predict([[15,222,2,3.0]]))
print(model.predict([[15,250,3,4.0]]))
'''
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ucla_admit = pd.read_csv('d:/data/dataset_0914/ucla_admit.csv')
#print(ucla_admit)

ucla_admit_X = ucla_admit[['gre','gpa','rank']]
ucla_admit_y = ucla_admit['admit']

train_X, test_X, train_y, test_y = \
    train_test_split(ucla_admit_X, ucla_admit_y, test_size=0.3,\
                     random_state=1234) 

# Define learning model
model =  LogisticRegression()

# Train the model using the training sets
model.fit(train_X, train_y)




# Make predictions using the testing set
pred_y = model.predict(test_X)
#print(pred_y)


test_acc = accuracy_score(test_y, pred_y)
print('Test Accuracy : {0:3f}'.format(test_acc))

pred_y = model.predict(train_X)
training_acc = accuracy_score(train_y, pred_y)
print('Training Accuracy : {0:3f}'.format(training_acc))

print(model.predict([[400,3.5,5]]))
print(model.predict([[550,3.8,2]]))
print(model.predict([[700,4.0,2]]))
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ucla_admit = pd.read_csv('d:/data/dataset_0914/ucla_admit.csv')
#print(ucla_admit)

ucla_admit_X = ucla_admit[['gre','gpa']]
ucla_admit_y = ucla_admit['admit']

train_X, test_X, train_y, test_y = \
    train_test_split(ucla_admit_X, ucla_admit_y, test_size=0.3,\
                     random_state=1234) 


# Define learning model
model =  LogisticRegression()

# Train the model using the training sets
model.fit(train_X, train_y)




# Make predictions using the testing set
pred_y = model.predict(test_X)
#print(pred_y)

test_acc = accuracy_score(test_y, pred_y)
print('Test Accuracy : {0:3f}'.format(test_acc))

pred_y = model.predict(train_X)
training_acc = accuracy_score(train_y, pred_y)
print('Training Accuracy : {0:3f}'.format(training_acc))
