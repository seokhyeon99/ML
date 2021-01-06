import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras import Sequential
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.utils import to_categorical
from keras.datasets import cifar10

#develop model
(train_X,train_y),(test_X,test_y)=cifar10.load_data()
train_X,val_X,train_y,val_y=train_test_split(train_X,train_y,test_size=.3) 
train_y=to_categorical(train_y)
val_y=to_categorical(val_y)
test_y=to_categorical(test_y)

train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )
val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True, zoom_range=.1) 

train_generator.fit(train_X)
val_generator.fit(val_X)
test_generator.fit(test_X)

lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3,  min_lr=1e-5) 

base_model_1 = VGG19(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=train_y.shape[1])

model_1= Sequential()
model_1.add(base_model_1)
model_1.add(Flatten())
model_1.add(Dense(1024,activation=('relu'),input_dim=512))
model_1.add(Dense(512,activation=('relu'))) 
model_1.add(Dense(256,activation=('relu'))) 
#model_1.add(Dropout(.3))
model_1.add(Dense(128,activation=('relu')))
#model_1.add(Dropout(.2))
model_1.add(Dense(10,activation=('softmax')))

batch_size= 100
epochs=50
learn_rate=.001

sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
adam=Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_1.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy']) 
model_1.fit_generator(train_generator.flow(train_X,train_y,batch_size=batch_size), epochs=epochs, steps_per_epoch=train_X.shape[0]//batch_size, validation_data=val_generator.flow(val_X,val_y,batch_size=batch_size),validation_steps=250, callbacks=[lrr], verbose=1)

#save model
model_1.save('cifar10_model.h5') 
model_json = model_1.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)
model_1.save_weights("model.h5")