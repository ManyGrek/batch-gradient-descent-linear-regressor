# Izael Manuel Rascón Durán A01562240

import numpy as np


class BGDLinearRegressor:

    def __init__(self):
        self.theta = None

    def fit(self, features: np.ndarray, y: np.ndarray,
            *, learning_rate, n_iterations):
        m = y.shape[0]
        X = np.c_[np.ones((m, 1)), features]  # add the intercept values

        n = X.shape[1]

        theta = np.random.randn(n, 1)  # random initialization of parameters

        # Gradient descent optimization
        for iteration in range(n_iterations):
            gradients = 2 / m * X.T.dot(X.dot(theta) - y)
            theta = theta - learning_rate * gradients

        self.theta = theta

    def predict(self, features):
        m = features.shape[0]
        X = np.c_[np.ones((m, 1)), features]  # add intercept values

        return X.dot(self.theta)
