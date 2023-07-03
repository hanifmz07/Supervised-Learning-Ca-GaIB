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

    def fit(self, X, y_true):
        X = X.T
        self.__w = np.zeros((X.shape[0], 1))
        self.__b = 0

        for i in range(self.__epochs):
            self.__propagate(X, y_true)
            self.__w -= self.__learning_rate * self.__dw
            self.__b -= self.__learning_rate * self.__db

            if i % 50 == 0:
                print(f"Cost = {self.__cost}")

    def predict(self, X):
        y_pred = self.__sigmoid(np.dot(self.__w.T, X) + self.__b)
        return np.vectorize(lambda x: 0 if x < 0.5 else 1)(y_pred)


# model = LogisticRegression(reg=0)
# df = pd.read_csv("Iris.csv")
# df_binary = df.drop(df[df["Species"] == "Iris-virginica"].index, axis=0).drop(
#     ["Id"], axis=1
# )
# encoding = {"Iris-setosa" : 0, "Iris-versicolor" : 1}
# df_binary["Species"] = df_binary["Species"].map(encoding)
