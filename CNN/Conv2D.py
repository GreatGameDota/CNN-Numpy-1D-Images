import numpy as np


class Conv2D():

    def __init__(self, filters, kernal_size, stride=(1, 1), padding="valid", activation=None, use_bias=True, kernal_initializer="glorot_uniform", bia_initializer="zeros", input_shape=None, prev_filters=1):
        self._filters = filters
        self._kernal_size = kernal_size
        self._stride = stride
        self._padding = padding
        self._activation = activation
        self._use_bias = use_bias
        self._kernal_initializer = kernal_initializer
        self._bias_initializer = bia_initializer
        self._input_shape = input_shape
        self._bias = np.zeros((filters, 1))

    def setPrevFilters(self, filters):
        self._prevUnits = filters
        self._filter = np.zeros(
            (self._filters, filters, self._kernal_size[0], self._kernal_size[1]))

    def forward(self, image):
        self._conv_in = image
        f1, f2, f3, _ = self._filter.shape
        i1, i2, _ = image.shape
        out_dim = int((i2 - f3) / self._stride[0]) + 1
        out = np.zeros((f1, out_dim, out_dim))
        assert f2 == i1, "Dimension mismatch at forward conv"

        for curr_f in range(f1):
            curr_y = out_y = 0
            while curr_y + f3 <= i2:
                curr_x = out_x = 0
                while curr_x + f3 <= i2:
                    out[curr_f, out_y, out_x] = np.sum(
                        self._filter[curr_f] * image[:, curr_y:curr_y+f3, curr_x:curr_x+f3]) + self._bias[curr_f]
                    curr_x += self._stride[0]
                    out_x += 1
                curr_y += self._stride[1]
                out_y += 1

        if(self._activation == "relu"):
            out[out <= 0] = 0
        return out

    def backward(self, dconv_prev):
        f1, _, f3, _ = self._filter.shape
        _, orig_dim, _ = self._conv_in
        dout = np.zeros(self._conv_in.shape)
        dfilt = np.zeros(self._filter.shape)
        dbias = np.zeros((f1, 1))

        for curr_f in range(f1):
            curr_y = out_y = 0
            while curr_y + f3 <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f3 <= orig_dim:
                    dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * \
                        self._conv_in[:, curr_y:curr_y+f3, curr_x:curr_x+f3]
                    dout[:, curr_y:curr_y+f3, curr_x:curr_x +
                         f3] += dconv_prev[curr_f, out_y, out_x] * self._filter[curr_f]
                    curr_x += self._stride[0]
                    out_x += 1
                curr_y += self._stride[1]
                out_y += 1
            dbias[curr_f] = np.sum(dconv_prev[curr_f])

        # if(self._activation == "relu"):
            # dout[self._conv_in <= 0] = 0
        return dout, dfilt, dbias
