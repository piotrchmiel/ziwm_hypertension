from cmath import sqrt
from sklearn import neighbors

import numpy as np
from sklearn.multiclass import OneVsOneClassifier, _predict_binary
from sklearn.neighbors import NearestNeighbors


class DynamicOneVsOneClassifier(OneVsOneClassifier):

    def __init__(self, estimator, n_jobs=-1, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, threshold=0.2, metric_params=None):
        OneVsOneClassifier.__init__(self, estimator, n_jobs)
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm=algorithm,
                                     leaf_size=leaf_size, metric=metric, p=p,
                                     metric_params=metric_params, n_jobs=n_jobs)
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self._fit_y = None

    def fit(self, X, y):
        self._fit_y = y
        self.nbrs.fit(X, y)
        return OneVsOneClassifier.fit(self, X, y)

    def decision_function(self, X):
        neighbors = self.nbrs.kneighbors(X, self.n_neighbors, return_distance=False)
        neighbors_list = []
        predictions = []
        confidences = []

        for neighbor in neighbors[0]:
            neighbors_list.append(self._fit_y[neighbor])

        neighbors_count = len(neighbors_list)
        neighbors_set = set(neighbors_list)
        neighbors_set_tmp = neighbors_set.copy()
        for neighbor in neighbors_set_tmp:
            if not neighbors_list.count(neighbor)/neighbors_count > self.threshold:
                neighbors_set.remove(neighbor)

        n_classes = int(((1 + sqrt(4 * 2 * len(self.estimators_) + 1)
                          ) / 2).real)  # n*(n-1)/2 binary classificators

        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                if i in neighbors_set or j in neighbors_set:
                    predictions.append(self.estimators_[k].predict(X))
                    confidences.append(_predict_binary(self.estimators_[k], X))
                else:
                    predictions.append(np.nan)
                    confidences.append(np.nan)
                k += 1

        predictions = np.vstack(predictions).T
        confidences = np.vstack(confidences).T
        return self._dynamic_ovr_decision_function(predictions, confidences, len(self.classes_))

    @staticmethod
    def _dynamic_ovr_decision_function(predictions, confidences, n_classes):
        """Compute a continuous, tie-breaking ovr decision function.

        It is important to include a continuous value, not only votes,
        to make computing AUC or calibration meaningful.

        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Predicted classes for each binary classifier.

        confidences : array-like, shape (n_samples, n_classifiers)
            Decision functions or predicted probabilities for positive class
            for each binary classifier.

        n_classes : int
            Number of classes. n_classifiers must be
            ``n_classes * (n_classes - 1 ) / 2``
        """
        n_samples = predictions.shape[0]
        votes = np.zeros((n_samples, n_classes))
        sum_of_confidences = np.zeros((n_samples, n_classes))

        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                if not np.isnan(confidences[:, k]) or not np.isnan(predictions[:, k]):
                    sum_of_confidences[:, i] -= confidences[:, k]
                    sum_of_confidences[:, j] += confidences[:, k]
                    votes[predictions[:, k] == 0, i] += 1
                    votes[predictions[:, k] == 1, j] += 1
                k += 1

        max_confidences = sum_of_confidences.max()
        min_confidences = sum_of_confidences.min()

        if max_confidences == min_confidences:
            return votes

        # Scale the sum_of_confidences to (-0.5, 0.5) and add it with votes.
        # The motivation is to use confidence levels as a way to break ties in
        # the votes without switching any decision made based on a difference
        # of 1 vote.
        eps = np.finfo(sum_of_confidences.dtype).eps
        max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
        scale = (0.5 - eps) / max_abs_confidence
        return votes + sum_of_confidences * scale
