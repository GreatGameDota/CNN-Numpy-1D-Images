class Conv2D:

    def __init__(self, filters, kernal_size, stride=(1, 1), padding="valid", activation=None, use_bias=True, kernal_initializer="glorot_uniform", bia_initializer="zeros", input_shape=None):
        self._filters = filters
        self._kernal_size = kernal_size
        self._stride = stride
        self._padding = padding
        self._activation = activation
        self._use_bias = use_bias
        self._kernal_initializer = kernal_initializer
        self._bias_initializer = bia_initializer
        self._input_shape = input_shape
