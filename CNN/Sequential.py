from CNN.Conv2D import Conv2D
from CNN.Dense import Dense
from CNN.MaxPool2D import MaxPool2D
from CNN.Flatten import Flatten
import numpy as np
from tqdm import trange
import pickle


class Sequential():

    def __init__(self, layers=None):
        self._layers = []
        self.conv2D = []
        self.dense = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)
        if (isinstance(layer, Conv2D)):
            self.conv2D.append(len(self._layers) - 1)
            if (len(self._layers) != 1):
                layer.setPrevFilters(
                    self._layers[-1]._filters)
            else:
                layer.setPrevFilters(layer._input_shape[2])
        if isinstance(layer, Dense):
            self.dense.append(len(self._layers) - 1)

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
                    (nextDim, nextDim, 1))
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

    def fit(self, x=None, y=None, batch_size=32, epochs=1, shuffle=True):
        iterations = np.ceil(len(x) / batch_size)
        data = np.hstack((x, y))
        self.costs = []

        t = trange(epochs)
        for epoch in t:
            if shuffle:
                np.random.shuffle(data)
            cost_ = 0

            all_params = []
            for conv in self.conv2D:
                all_params.append(self._layers[conv]._filter)
                all_params.append(self._layers[conv]._bias)
            for dense in self.dense:
                all_params.append(self._layers[dense]._weights)
                all_params.append(self._layers[dense]._bias)

            all_grads = []
            for param in all_params:
                all_grads.append(np.zeros(param.shape))

            t = trange(batch_size)
            for i in t:
                y = data[:, -1]
                x = data[:, 0:-1]
                i1, i2, i3 = self._layers[0]._input_shape
                x_train = np.reshape(x[i], (i3, i1, i2))
                label = np.eye(self.num_classes)[
                    int(y[i])].reshape(self.num_classes, 1)
                # Forward
                conv = []
                pool = []
                wIdx = -1
                num = 0
                for layer in self._layers:
                    if num == 0:
                        out = layer.forward(x_train)
                    else:
                        out = layer.forward(out)
                    if num == wIdx:
                        z = out
                    if isinstance(layer, Flatten):
                        flat = out
                        wIdx = num + 1
                    if isinstance(layer, MaxPool2D):
                        pool.append(out)
                    if isinstance(layer, Conv2D):
                        conv.append(out)
                    num += 1
                # Loss and Prob
                probs = out
                if self._loss == "categorical_crossentropy":
                    loss = self.categoricalCrossEntropy(probs, label)
                # Backward
                dout = probs - label
                grads = []
                poolNum = 0
                num = 0
                for layer in reversed(self._layers):
                    if num == 0:
                        dw1, _db1 = layer.backwardFirst(dout, z)
                        grads.append(_db1)
                        grads.append(dw1)
                    else:
                        if isinstance(layer, Dense):
                            dw, db, dz = layer.backward(
                                dout, self._layers[len(self._layers) - (num)]._weights, flat, z)
                            grads.append(db)
                            grads.append(dw)
                        elif isinstance(layer, MaxPool2D):
                            dfc = self._layers[wIdx]._weights.T.dot(dz)
                            dpool = dfc.reshape(pool[poolNum].shape)
                            dconv = layer.backward(dpool)
                            poolNum += 1
                        elif isinstance(layer, Conv2D):
                            dconv, df, db = layer.backward(dconv)
                            if num != len(self._layers) - 1:
                                dconv[layer._conv_in <= 0] = 0
                            grads.append(db)
                            grads.append(df)
                    num += 1
                idx = len(all_grads) - 1
                for i, grad in enumerate(all_grads):
                    all_grads[i] = grad + grads[idx]
                    idx -= 1
                cost_ += loss
                t.set_description('Epoch %i: Cost %.3f' %
                                  ((epoch + 1), loss))
            params, self.costs = self._opt.update(
                all_params, all_grads, cost_, self.costs, batch_size)
            idx = 0
            for conv in self.conv2D:
                self._layers[conv]._filter = params[idx]
                idx += 1
                self._layers[conv]._bias = params[idx]
                idx += 1
            for dense in self.dense:
                self._layers[dense]._weights = params[idx]
                idx += 1
                self._layers[dense]._bias = params[idx]
                idx += 1
        # print(self.costs)
        print("\n")

    def predict_classes(self, image):
        i1, i2, i3 = self._layers[0]._input_shape
        x_train = np.reshape(image, (i3, i1, i2))
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                out = layer.forward(x_train)
                conv1 = out
            else:
                out = layer.forward(out)
                if isinstance(layer, Conv2D):
                    conv2 = out
                if isinstance(layer, MaxPool2D):
                    maxpool1 = out
        return np.argmax(out), out, conv1, conv2, maxpool1

    def evaluate(self, x_test, y_test):
        batch_size = len(x_test)
        amountCorrect = 0
        t = trange(batch_size)
        for i in t:
            if i > 0:
                t.set_description('Acc: %.2f' %
                                  (amountCorrect / float(i) * 100) + '%')
            pred, _, _, _, _ = self.predict_classes(x_test[i])
            if pred == y_test[i]:
                amountCorrect += 1
        return float(amountCorrect) / batch_size

    def categoricalCrossEntropy(self, probs, label):
        return -np.sum(label * np.log(probs))

    def save(self, fileName):
        to_save = [self._layers, self._inputShapes, self._loss,
                   self._opt, self.costs, self.num_classes, self.conv2D, self.dense]
        with open(fileName, 'wb') as file:
            pickle.dump(to_save, file)

    def load_model(self, path):
        self._layers, self._inputShapes, self._loss, self._opt, self.costs, self.num_classes, self.conv2D, self.dense = pickle.load(
            open(path, 'rb'))
