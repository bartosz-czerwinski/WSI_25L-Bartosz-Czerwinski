import numpy as np
from collections import Counter
from cwiczenie4.DecisionTree import DecisionTreeClassifier
import random

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_indices_list = []

    def fit(self, X, y):
        self.trees = []
        num_features_total = X.shape[1]
        self.feature_indices_list = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]

            max_feats = int(np.sqrt(num_features_total))
            feature_indices = np.random.choice(num_features_total, max_feats, replace=False)
            self.feature_indices_list.append(feature_indices)

            X_sample_sub = X_sample[:, feature_indices]

            # Stworzenie drzewa
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )

            tree.fit(X_sample_sub, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = []
        for i in range(len(self.trees)):
            tree = self.trees[i]
            feature_indices = self.feature_indices_list[i]
            X_sub = X[:, feature_indices]
            preds = tree.predict(X_sub)
            tree_preds.append(preds)

        tree_preds = np.array(tree_preds).T

        final_preds = []
        for row in tree_preds:
            counts = np.bincount(row)
            most_common = np.argmax(counts)
            final_preds.append(most_common)

        return np.array(final_preds)
