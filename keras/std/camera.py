import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
import cv2
import matplotlib.pyplot as plot
import numpy as py
import math

num_train_samples = 10000
x_train = py.empty(num_train_samples)
y_train = py.empty(num_train_samples)
x_test = py.empty(10)
y_test = py.empty(10)

for i in range(0, num_train_samples):
    if i<10:
        x_test[i] = -i
        y_test[i] = 10*py.sin(i)
        
    x_train[i] = i
    y_train[i] = 10*py.sin(i)

x_train.reshape(num_train_samples,1,1)
y_train.reshape(num_train_samples,1,1)

y_train = y_train.astype('int32')
y_train = keras.utils.to_categorical(y_train, 100)

x_test.reshape(10,1,1)
y_test.reshape(10,1,1)

y_test = y_test.astype('int32')
y_test = keras.utils.to_categorical(y_test, 100)
plot.plot(x_test,y_test)
print(y_test)
#plot.show()

model = Sequential()
model.add(Dense(10, input_shape=(1,), activation='relu'))
model.add(Dense(10, activation='relu'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=30,
          verbose=1,
          validation_data=(x_test, y_test))

model.save("/home/xyzn/mygit/DL/keras/models/keras_std1_model.h5")
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
