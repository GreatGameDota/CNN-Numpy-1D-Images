from CNN.Conv2D import Conv2D
from CNN.Dense import Dense
from CNN.MaxPool2D import MaxPool2D
from CNN.Flatten import Flatten
import numpy as np


class Sequential():

    def __init__(self, layers=None):
        self._layers = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)
        if (isinstance(layer, Conv2D)):
            if (len(self._layers) != 0):
                layer.setPrevFilters(
                    self._layers[-1]._filters)
            else:
                layer.setPrevFilters(1)

    def compile(self, opt, loss):
        self._opt = opt
        self._loss = loss
        self._inputShapes = []
        num = 0
        nextDim = 0
        for layer in self._layers:
            if isinstance(layer, Conv2D):
                _, _, f3, _ = layer._filter.shape
                if num == 0:
                    nextDim = int(
                        (layer._input_shape[0] - f3) / layer._stride[0]) + 1
                else:
                    nextDim = int(
                        (self._inputShapes[-1][0] - f3) / layer._stride[0]) + 1
            elif isinstance(layer, MaxPool2D):
                nextDim = int(
                    (self._inputShapes[-1][0] - layer._pool_size[0]) / layer._stride[0]) + 1
            if isinstance(layer, Conv2D) or isinstance(layer, MaxPool2D):
                self._inputShapes.append(
                    (nextDim, nextDim, self._layers[0]._input_shape[2]))
                num += 1
        idx = 0
        for layer in self._layers:
            if isinstance(layer, Dense):
                if isinstance(self._layers[idx - 1], Flatten):
                    if isinstance(self._layers[idx - 2], Conv2D):
                        layer.setPrevUnits(
                            self._inputShapes[-1][0]*self._inputShapes[-1][1]*self._inputShapes[-1][2]*self._layers[idx - 2]._filters)
                    elif isinstance(self._layers[idx - 2], MaxPool2D):
                        layer.setPrevUnits(
                            self._inputShapes[-1][0]*self._inputShapes[-1][1]*self._inputShapes[-1][2]*self._layers[idx - 3]._filters)
                else:
                    layer.setPrevUnits(self._layers[idx-1]._units)
            idx += 1

    def fit(self, x=None, y=None, batch_size=None, epochs=1, shuffle=True):
        # TODO
        x /= 255

    def categoricalCrossEntropy(self, probs, label):
        return -np.sum(label * np.log(probs))
