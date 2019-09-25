from CNN.Sequential import Sequential
from CNN.Conv2D import Conv2D
from CNN.Dense import Dense
from CNN.MaxPool2D import MaxPool2D
from CNN.Flatten import Flatten

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu",
                 kernal_initializer="he_uniform", padding="same", input_shape=(32, 32, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
# opt = SGD(lr=0.001, momentum=0.9)
# model.compile(optimizer=opt, loss='categorical_crossentropy',
# metrics=['accuracy'])
# return model
