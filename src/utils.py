import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def f1_score(y_true, y_pred):
    fp = tp = fn = 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    categories = np.unique(y_true)
    print(f"y_true = {y_true}")
    print(f"y_pred = {y_pred}")
    for i in range(len(y_true)):
        if y_true[i] == categories[0] and y_pred[i] == categories[0]:
            tp += 1
        elif y_true[i] == categories[1] and y_pred[i] == categories[0]:
            fp += 1
        elif y_true[i] == categories[0] and y_pred[i] == categories[1]:
            fn += 1
    return (2 * tp) / (2 * tp + fp + fn)


def preprocess_iris():
    df = pd.read_csv("Iris.csv")
    df_binary = df.drop(df[df["Species"] == "Iris-virginica"].index, axis=0).drop(
        ["Id"], axis=1
    )
    encoding = {"Iris-setosa": 0, "Iris-versicolor": 1}
    df_binary["Species"] = df_binary["Species"].map(encoding)
    X_iris = df_binary.drop(["Species"], axis=1)
    y_iris = df_binary["Species"]
    return X_iris, y_iris


def preprocess_mushroom():
    df_mush = pd.read_csv("mushrooms.csv")
    X_mush = df_mush.drop(["class"], axis=1)
    y_mush = df_mush["class"].copy()
    return X_mush, y_mush


def preprocess_bcancer():
    df_canc = pd.read_csv("bcancer.csv")
    df_canc.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    encoding = {"B": 0, "M": 1}
    df_canc["diagnosis"] = df_canc["diagnosis"].map(encoding)
    X_canc = df_canc.drop(["diagnosis"], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(X_canc)
    X_transform = scaler.transform(X_canc)
    y_canc = df_canc["diagnosis"].copy()
    X_transform_df = pd.DataFrame(X_transform, columns=X_canc.columns)

    return X_transform_df, y_canc
