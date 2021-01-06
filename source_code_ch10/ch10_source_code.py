'''
# load required modules
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt
import numpy as np

# load dataset
dataframe = pandas.read_csv ("D:/data/dataset_0914/iris.csv")
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
                             
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
                             
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
                             
# Divide train, test
train_X, test_X , train_y , test_y = train_test_split (X, dummy_y , test_size =0.4, random_state =321)

# define model (DNN structure)
epochs = 50
batch_size = 10
                             
model = Sequential()
model.add(Dense(10, input_dim =4 , activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.summary() # show model structure
                             
# Compile model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                             
# model fitting (learning)
disp = model.fit(train_X , train_y ,batch_size = batch_size, epochs=epochs, verbose=1, validation_data = (test_X , test_y))
                             
# Test model
pred = model.predict(test_X)
print(pred)
y_classes = [np.argmax(y, axis=None, out=None) for y in pred]
print(y_classes) # result of prediction
                             
# model performance
score = model.evaluate(test_X , test_y , verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
                             
# summarize history for accuracy
plt.plot(disp.history ['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

'''
# load required modules
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt
import numpy as np

# load dataset
dataframe = pandas.read_csv ("D:/data/dataset_0914/PimaIndiansDiabetes.csv")
dataset = dataframe.values
X = dataset[:,0:8].astype(float)
Y = dataset[:,8]
                             
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
                             
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
                             
# Divide train, test
train_X, test_X , train_y , test_y = train_test_split (X, dummy_y , test_size =0.3, random_state =321)

# define model (DNN structure)
epochs = 100
batch_size = 20
                             
model = Sequential()
model.add(Dense(12, input_dim =8 , activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.summary() # show model structure
                             
# Compile model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                             
# model fitting (learning)
disp = model.fit(train_X , train_y ,batch_size = batch_size, epochs=epochs, verbose=1, validation_data = (test_X , test_y))
                             
# Test model
pred = model.predict(test_X)
print(pred)
y_classes = [np.argmax(y, axis=None, out=None) for y in pred]
print(y_classes) # result of prediction
                             
# model performance
score = model.evaluate(test_X , test_y , verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
                             
# summarize history for accuracy
plt.plot(disp.history ['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

'''
# load required modules
from keras.datasets import mnist
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

# load dataset
(train_X , train_y ), (test_X , test_y ) = mnist.load_data()
train_X, test_X = train_X / 255.0, test_X / 255.0
# one hot encoding
train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)

# define model (DNN structure)
epochs = 20
batch_size = 128
learning_rate = 0.01

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate = 0.4))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# Compile model
adam = optimizers.adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# model fitting (learning)
disp = model.fit(train_X , train_y ,batch_size=batch_size, epochs=epochs, verbose=1, validation_split = 0.2)

# Test model
pred = model.predict(test_X)
print(pred)
y_classes = [np.argmax (y, axis=None, out=None) for y in pred]
print(y_classes) # result of prediction
# model performance
score = model.evaluate(test_X , test_y , verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# summarize history for accuracy
plt.plot(disp.history ['accuracy'])
plt.plot(disp.history ['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('eopch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show
'''