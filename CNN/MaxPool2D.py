class MaxPool2D:

  def __init__(self, pool_size=(2, 2), stride=None, padding=None):
      self._pool_size = pool_size
      if stride==None:
        self._stride = pool_size
      self._padding = padding
