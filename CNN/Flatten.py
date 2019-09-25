import numpy as np


class Flatten():

    def forward(self, arr):
        return arr.reshape((arr.shape[0]*arr.shape[1]*arr.shape[2], 1))

    def backward(self, arr, shape):
        return arr.reshape(shape)
