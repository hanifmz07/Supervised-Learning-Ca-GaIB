import numpy as np
import pandas as pd


class LogisticRegression:
    """Logistic regression"""

    def __init__(self, epochs=500, learning_rate=0.01, reg="l2"):
        self._w = None
        self._b = None
        self._dw = None
        self._db = None
        self._cost = None

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._reg = reg

    def _logloss(self, y_pred, y_true):
        if self._reg == "l2":
            return np.sum(
                -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
            ) + 0.01 * np.sum(self._w**2)
        else:
            return np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _propagate(self, X, y_true):
        m = X.shape[1]

        # Forward propagation
        y_pred = self._sigmoid(np.dot(self._w.T, X) + self._b)
        self._cost = self._logloss(y_pred, y_true)

        # Backward propagation
        if self._reg == "l2":
            self._dw = (np.dot(X, (y_pred - y_true).T) / m) + (0.01 / m) * self._w
        else:
            self._dw = np.dot(X, (y_pred - y_true).T) / m
        self._db = np.sum(y_pred - y_true) / m

    def fit(self, X, y):
        self._w = np.zeros((X.shape[1], 1))
        self._b = 0

        y_true = np.array(y).reshape((1, X.shape[0]))

        for i in range(self._epochs):
            self._count = i
            self._propagate(X.T, y_true)
            self._w -= self._learning_rate * self._dw
            self._b -= self._learning_rate * self._db

            if i % 100 == 0:
                print(f"Cost = {self._cost}")

    def predict(self, X):
        X = X.T
        y_pred = self._sigmoid(np.dot(self._w.T, X) + self._b)
        return np.vectorize(lambda x: 0 if x < 0.5 else 1)(y_pred)[0]
