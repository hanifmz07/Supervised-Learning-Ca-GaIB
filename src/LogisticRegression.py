import numpy as np
import pandas as pd


class LogisticRegression:
    """Logistic regression"""

    def __init__(self, epochs=500, learning_rate=0.01, reg="l2"):
        self.__w = None
        self.__b = None
        self.__dw = None
        self.__db = None
        self.__cost = None

        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__reg = reg

    def __logloss(self, y_pred, y_true):
        if self.__reg == "l2":
            return np.sum(
                -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
            ) + 0.01 * np.sum(self.__w**2)
        else:
            return np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __propagate(self, X, y_true):
        m = X.shape[1]

        # Forward propagation
        y_pred = self.__sigmoid(np.dot(self.__w.T, X) + self.__b)
        self.__cost = self.__logloss(y_pred, y_true)

        # Backward propagation
        self.__dw = np.dot(X, (y_pred - y_true).T) / m
        self.__db = np.sum(y_pred - y_true) / m

    def fit(self, X, y):
        self.__w = np.zeros((X.shape[1], 1))
        self.__b = 0

        y_true = np.array(y).reshape((1, X.shape[0]))

        for i in range(self.__epochs):
            self.__propagate(X.T, y_true)
            self.__w -= self.__learning_rate * self.__dw
            self.__b -= self.__learning_rate * self.__db

            if i % 50 == 0:
                print(f"Cost = {self.__cost}")

    def predict(self, X):
        X = X.T
        y_pred = self.__sigmoid(np.dot(self.__w.T, X) + self.__b)
        return np.vectorize(lambda x: 0 if x < 0.5 else 1)(y_pred)
