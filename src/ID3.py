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
        self.__root = Node()
        self.__current_node = self.__root

    def fit(self, X_train, y_train):
        if len(np.unique(y_train)) == 1:
            self.__current_node.final_label = y_train.iloc[0]
        else:
            max_gain = 0
            selected_feature = ""
            for feature in X_train.columns:
                current_gain = self.__gain(X_train, y_train, feature)
                if max_gain < current_gain:
                    max_gain = current_gain
                    selected_feature = feature
            self.__current_node.feature = selected_feature
            categories = np.unique(X_train[selected_feature])

            for category in categories:
                self.__current_node.child[category] = Node(parent=self.__current_node)

            for key in self.__current_node.child:
                self.__current_node = self.__current_node.child[key]
                next_X = X_train[X_train[selected_feature] == key].drop(
                    [selected_feature], axis=1
                )
                next_y = y_train[next_X.index]
                self.fit(next_X, next_y)
        if self.__current_node.parent != None:
            self.__current_node = self.__current_node.parent

    def predict(self, X_test):
        result = []
        for i in range(X_test.shape[0]):
            point = X_test.iloc[i, :]
            while self.__current_node.final_label == None:
                self.__current_node = self.__current_node.child[
                    point[self.__current_node.feature]
                ]
            result.append(self.__current_node.final_label)
            self.__current_node = self.__root
        return np.array(result)

    def __entropy(self, labels):
        entropy = 0
        for n in labels.value_counts():
            entropy -= (n / len(labels)) * np.log2(n / len(labels))
        return entropy

    def __gain(self, X_train, y_train, feature):
        categories = np.unique(X_train[feature])
        childs_entropy = 0
        for category in categories:
            prob_feature = len(X_train[X_train[feature] == category]) / len(y_train)
            indices = X_train[X_train[feature] == category].index
            labels = y_train[indices]
            childs_entropy += prob_feature * self.__entropy(labels)
        parent_entropy = self.__entropy(y_train)
        return parent_entropy - childs_entropy
