import numpy as np


class adamGD():

    def __init__(self, lr):
        self._lr = lr
        self.beta1 = 0.95
        self.beta2 = 0.99

    def update(self, weights, params, cost, costs, batch_size):
        lr = self._lr
        beta1 = self.beta1
        beta2 = self.beta2

        for idx, grad in enumerate(params):
            v1 = (1-beta1)*grad/batch_size
            s1 = (1-beta2)*(grad/batch_size)**2
            weights[idx] -= lr * v1 / np.sqrt(s1 + 1e-7)

        cost /= batch_size
        costs.append(cost)
        return weights, costs
