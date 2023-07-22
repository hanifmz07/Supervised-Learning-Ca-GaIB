from src.id3 import ID3
from src.knn import KNN
from src.logreg import LogisticRegression
from src.utils import *

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
models = ["knn", "log_reg", "id_3"]
parser.add_argument(
    "-m",
    "--model",
    choices=models,
    default="knn",
    help="Supervised learning model to be used",
)
parser.add_argument(
    "-d", "--data", default="iris.csv", type=str, help="CSV file directory"
)
parser.add_argument(
    "--epochs", type=int, default=1000, help="Epochs for logistic regression"
)
parser.add_argument(
    "--lr", type=float, default=0.1, help="Learning rate for logistic regression"
)
parser.add_argument(
    "--k_nearest", type=int, default=5, help="Amount of nearest neighbors for KNN"
)

args = parser.parse_args()

if args.data == "iris.csv":
    X, y = preprocess_iris()
elif args.data == "bcancer.csv":
    X, y = preprocess_bcancer()
elif args.data == "mushrooms.csv":
    X, y = preprocess_mushroom()
else:
    print("Data not available")
    quit()

if args.model == "knn":
    model = KNN(n_neighbors=args.k_nearest)
elif args.model == "log_reg":
    model = LogisticRegression(epochs=args.epochs, learning_rate=args.lr)
elif args.model == "id_3":
    model = ID3()
else:
    print("Invalid input")
    quit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"F1-score = {f1_score(y_test, y_pred)}")
