import numpy as np


class MaxPool2D():

    def __init__(self, pool_size=(2, 2), stride=None, padding=None):
        self._pool_size = pool_size
        if stride == None:
            self._stride = pool_size
        else:
            self._stride = stride
        self._padding = padding

    def forward(self, image):
        self._orig = image
        i1, i2, i3 = image.shape
        h = int((i2 - self._pool_size[1]) / self._stride[1]) + 1
        w = int((i3 - self._pool_size[0]) / self._stride[0]) + 1
        out = np.zeros((i1, h, w))

        for i in range(i1):
            curr_y = out_y = 0
            while curr_y + self._pool_size[1] <= i2:
                curr_x = out_x = 0
                while curr_x + self._pool_size[0] <= i3:
                    out[i, out_y, out_x] = np.max(
                        image[i, curr_y:curr_y+self._pool_size[1], curr_x:curr_x+self._pool_size[0]])
                    curr_x += self._stride[0]
                    out_x += 1
                curr_y += self._stride[1]
                out_y += 1
        return out

    def backward(self, dpool):
        o1, o2, _ = self._orig.shape
        dout = np.zeros(self._orig.shape)

        for curr_c in range(o1):
            curr_y = out_y = 0
            while curr_y + self._pool_size[1] <= o2:
                curr_x = out_x = 0
                while curr_x + self._pool_size[0] <= o2:
                    a, b = self.nanargmax(
                        self._orig[curr_c, curr_y:curr_y+self._pool_size[1], curr_x:curr_x+self._pool_size[0]])
                    dout[curr_c, curr_y+a, curr_x +
                         b] = dpool[curr_c, out_y, out_x]
                    curr_x += self._stride[0]
                    out_x += 1
                curr_y += self._stride[1]
                out_y += 1
        # Backprop ReLU
        dout[self._orig <= 0] = 0
        return dout

    def nanargmax(self, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs
