from CNN.Sequential import Sequential
from CNN.Conv2D import Conv2D
from CNN.Dense import Dense
from CNN.MaxPool2D import MaxPool2D
from CNN.Flatten import Flatten
from CNN.adamGD import adamGD

model = Sequential()
model.add(Conv2D(8, (5, 5), activation="relu",
                 kernal_initializer="he_uniform", padding="same", input_shape=(28, 28, 1)))
model.add(Conv2D(8, (5, 5), activation="relu",
                 kernal_initializer="he_uniform", padding="same"))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
opt = adamGD(lr=0.01)
model.compile(opt, 'categorical_crossentropy')
