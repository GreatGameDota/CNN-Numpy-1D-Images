from Conv2D import Conv2D
from Dense import Dense
from MaxPool2D import MaxPool2D


class Sequential():

    def __init__(self, layers=None):
        self._layers = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)
        if (isinstance(layer, Conv2D)):
            if (self._layers):
                layer.setPrevFilters(
                    self._layers[len(self._layers) - 1]._filters)
            else:
                layer.setPrevFilters(1)

    def fit(self, X_Train, Y_train):
        # TODO
        X_Train /= 255
