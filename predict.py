from PIL import Image
from CNN.Sequential import Sequential
# Data prep
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np.reshape(y_train, (len(y_train), 1))
x_train = np.reshape(x_train, (len(x_train), 28 * 28))
y_test = np.reshape(y_test, (len(y_test), 1))
x_test = np.reshape(x_test, (len(x_test), 28 * 28))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.load_model('model.pkl')
pred, prob, conv1, conv2, maxpool1 = model.predict_classes(x_test[892])
print(pred)
print(y_test[892])
# acc = model.evaluate(x_test, y_test)
