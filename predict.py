from CNN.Sequential import Sequential
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
model.load_model('model.pkl')
