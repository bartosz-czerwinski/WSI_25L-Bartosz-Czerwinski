import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index      # indeks cechy
        self.threshold = threshold              # próg do podziału
        self.left = left                        # lewe poddrzewo
        self.right = right                      # prawe poddrzewo
        self.value = value                      # wartość, jeśli liść

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=100, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self.grow_tree(X, y)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        if len(y) == 0:
            return None

        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self.most_common_label(y)
            return DecisionNode(value=leaf_value)

        best_feature, best_thresh = self.best_split(X, y, num_features)
        # if best_feature is None:
        #     return DecisionNode(value=self.most_common_label(y))

        left_indices = X[:, best_feature] < best_thresh
        right_indices = X[:, best_feature] >= best_thresh
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        if len(y_left) == 0 or len(y_right) == 0:
            leaf_value = self.most_common_label(y)
            return DecisionNode(value=leaf_value)

        left_node = self.grow_tree(X_left, y_left, depth + 1)
        right_node = self.grow_tree(X_right, y_right, depth + 1)

        return DecisionNode(feature_index=best_feature, threshold=best_thresh, left=left_node, right=right_node)

    def best_split(self, X, y, num_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self.information_gain(y, X[:, feature_index], threshold)
                if gain == 1:
                    return feature_index, threshold
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_thresh = threshold

        return split_idx, split_thresh

    def information_gain(self, y, feature_column, threshold):
        parent_entropy = self.entropy(y)

        left_indices = feature_column < threshold
        right_indices = feature_column >= threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])

        e_left = self.entropy(y[left_indices])
        e_right = self.entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def most_common_label(self, y):
        if len(y) == 0:
            return None
        counts = np.bincount(y)
        return np.argmax(counts)

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] < node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)


