import numpy as np


class KNN:
    """K-nearest neighbors algorithm."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X.copy().reset_index(drop=["index"])
        self.y = y.copy().reset_index(drop=["index"])

    def predict(self, X_test):
        result = []
        for i in range(X_test.shape[0]):
            point = np.array(X_test.iloc[i, :])

            # Euclidean distance
            self.X["distance"] = self.X.apply(
                lambda x: np.linalg.norm(x - point), axis=1
            )
            self.X.sort_values(by=["distance"], inplace=True)
            indices = self.X.head(self.n_neighbors).index
            counts = self.y[indices].value_counts()
            if len(counts) == 1:
                result.append(counts.index[0])
            elif counts[0] == counts[1]:
                categories = np.unique(self.y[indices])
                sum_dist_0 = self.X.loc[
                    self.y[indices][self.y[indices] == categories[0]].index
                ].sum()["distance"]
                sum_dist_1 = self.X.loc[
                    self.y[indices][self.y[indices] == categories[1]].index
                ].sum()["distance"]
                result.append(0 if sum_dist_0 > sum_dist_1 else 1)
            else:
                result.append(counts.idxmax())
            self.X.drop(["distance"], axis=1, inplace=True)
        return np.array(result)
