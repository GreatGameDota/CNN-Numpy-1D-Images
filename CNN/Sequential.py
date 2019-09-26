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
            if (len(self._layers) != 1):
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
        self.num_classes = self._layers[-1]._units

    def fit(self, x=None, y=None, batch_size=None, epochs=1, shuffle=True):
        data = np.hstack((x, y))

        f1 = self._layers[0]._filter
        f2 = self._layers[1]._filter
        w3 = self._layers[4]._weights
        w4 = self._layers[5]._weights
        b1 = self._layers[0]._bias
        b2 = self._layers[1]._bias
        b3 = self._layers[4]._bias
        b4 = self._layers[5]._bias

        df1 = np.zeros(f1.shape)
        df2 = np.zeros(f2.shape)
        dw3 = np.zeros(w3.shape)
        dw4 = np.zeros(w4.shape)
        db1 = np.zeros(b1.shape)
        db2 = np.zeros(b2.shape)
        db3 = np.zeros(b3.shape)
        db4 = np.zeros(b4.shape)

        v1 = np.zeros(f1.shape)
        v2 = np.zeros(f2.shape)
        v3 = np.zeros(w3.shape)
        v4 = np.zeros(w4.shape)
        bv1 = np.zeros(b1.shape)
        bv2 = np.zeros(b2.shape)
        bv3 = np.zeros(b3.shape)
        bv4 = np.zeros(b4.shape)

        s1 = np.zeros(f1.shape)
        s2 = np.zeros(f2.shape)
        s3 = np.zeros(w3.shape)
        s4 = np.zeros(w4.shape)
        bs1 = np.zeros(b1.shape)
        bs2 = np.zeros(b2.shape)
        bs3 = np.zeros(b3.shape)
        bs4 = np.zeros(b4.shape)

        for epoch in range(epochs):
            cost_ = 0
            np.random.shuffle(data)
            y = data[:, -1]
            x = data[:, 0:-1]
            i1, i2, i3 = self._layers[0]._input_shape
            x = np.reshape(x, (i3, i1, i2))
            label = np.eye(self.num_classes)[
                int(y)].reshape(self.num_classes, 1)
            # Forward
            conv = []
            wIdx = -1
            num = 0
            for layer in self._layers:
                if num == 0:
                    out = layer.forward(x)
                else:
                    out = layer.forward(out)
                if num == wIdx:
                    z = out
                if isinstance(layer, Flatten):
                    flat = out
                    wIdx = num + 1
                if isinstance(layer, MaxPool2D):
                    pooled = out
                if isinstance(layer, Conv2D):
                    conv.append(out)
                num += 1
            # Loss and Prob
            probs = out
            loss = self.categoricalCrossEntropy(probs, label)
            # Backward
            dout = probs - label
            grads = []
            num = 0
            for layer in reversed(self._layers):
                if num == 0:
                    dw1, _db1 = layer.backwardFirst(dout, z)
                    grads.append(dw1)
                    grads.append(_db1)
                else:
                    if isinstance(layer, Dense):
                        dw, db, dz = layer.backward(
                            dout, self._layers[len(self._layers) - (num)]._weights, flat, z)
                        grads.append(dw)
                        grads.append(db)
                    elif isinstance(layer, MaxPool2D):
                        dfc = self._layers[wIdx]._weights.T.dot(dz)
                        dpool = dfc.reshape(pooled.shape)
                        dconv = layer.backward(dpool)
                    elif isinstance(layer, Conv2D):
                        dconv, df, db = layer.backward(dconv)
                        if num != len(self._layers) - 1:
                            dconv[layer._conv_in <= 0] = 0
                        grads.append(df)
                        grads.append(db)
                num += 1
            [dw4_, db4_, dw3_, db3_,  df2_, db2_, df1_, db1_] = grads
            df1 += df1_
            db1 += db1_
            df2 += df2_
            db2 += db2_
            dw3 += dw3_
            db3 += db3_
            dw4 += dw4_
            db4 += db4_
            cost_ += loss
            print(epoch)
        # params = self._opt(grads, loss, 1)

    def categoricalCrossEntropy(self, probs, label):
        return -np.sum(label * np.log(probs))
