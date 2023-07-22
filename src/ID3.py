import numpy as np


class Node:
    """Node class"""

    def __init__(self, parent=None):
        self.child = {}
        self.parent = parent
        self.feature = None
        self.final_label = None


class ID3:
    """Decision tree using ID3 algorithm"""

    def __init__(self):
        self._root = Node()
        self._current_node = self._root

    def fit(self, X_train, y_train):
        if len(np.unique(y_train)) == 1:
            self._current_node.final_label = y_train.iloc[0]
        else:
            max_gain = 0
            selected_feature = ""
            for feature in X_train.columns:
                current_gain = self._gain(X_train, y_train, feature)
                if max_gain < current_gain:
                    max_gain = current_gain
                    selected_feature = feature
            self._current_node.feature = selected_feature
            categories = np.unique(X_train[selected_feature])

            for category in categories:
                self._current_node.child[category] = Node(parent=self._current_node)

            for key in self._current_node.child:
                self._current_node = self._current_node.child[key]
                next_X = X_train[X_train[selected_feature] == key].drop(
                    [selected_feature], axis=1
                )
                next_y = y_train[next_X.index]
                self.fit(next_X, next_y)
        if self._current_node.parent != None:
            self._current_node = self._current_node.parent

    def predict(self, X_test):
        result = []
        for i in range(X_test.shape[0]):
            point = X_test.iloc[i, :]
            while self._current_node.final_label == None:
                self._current_node = self._current_node.child[
                    point[self._current_node.feature]
                ]
            result.append(self._current_node.final_label)
            self._current_node = self._root
        return np.array(result)

    def _entropy(self, labels):
        entropy = 0
        for n in labels.value_counts():
            entropy -= (n / len(labels)) * np.log2(n / len(labels))
        return entropy

    def _gain(self, X_train, y_train, feature):
        categories = np.unique(X_train[feature])
        childs_entropy = 0
        for category in categories:
            prob_feature = len(X_train[X_train[feature] == category]) / len(y_train)
            indices = X_train[X_train[feature] == category].index
            labels = y_train[indices]
            childs_entropy += prob_feature * self._entropy(labels)
        parent_entropy = self._entropy(y_train)
        return parent_entropy - childs_entropy
