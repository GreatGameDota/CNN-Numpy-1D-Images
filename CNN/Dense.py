class Dense:

    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=None):
      self._units = units
      self._activation = activation
      self._use_bias = use_bias
      self._kernal_initializer = kernel_initializer
      self._bias_initializer = bias_initializer
