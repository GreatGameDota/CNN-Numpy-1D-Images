from CNN.Sequential import Sequential
from CNN.Conv2D import Conv2D
from CNN.Dense import Dense
from CNN.MaxPool2D import MaxPool2D
from CNN.Flatten import Flatten
from CNN.adamGD import adamGD
# Data prep
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np.reshape(y_train, (len(y_train), 1))
x_train = np.reshape(x_train, (len(x_train), 28 * 28))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Conv2D(8, (5, 5), activation="relu",
                 kernal_initializer="he_uniform", padding="same", input_shape=(28, 28, 1)))
model.add(Conv2D(8, (5, 5), activation="relu",
                 kernal_initializer="he_uniform", padding="same"))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# Compile model
opt = adamGD(lr=0.01)
model.compile(opt, 'categorical_crossentropy')
model.fit(x_train, y_train, batch_size=100, epochs=100)
model.save('model.pkl')
