from cmath import sqrt
import numpy as np

from sklearn.multiclass import OneVsOneClassifier, _predict_binary, _ovr_decision_function
from sklearn.neighbors import NearestNeighbors


class DynamicOneVsOneClassifier(OneVsOneClassifier):

    def __init__(self, estimator, n_jobs=-1, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None):
        OneVsOneClassifier.__init__(self, estimator, n_jobs)
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm=algorithm, leaf_size=leaf_size,
                                     metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
        self.n_neighbors = n_neighbors
        self._fit_X = None
        self._fit_Y = None

    def fit(self, X, y):
        self._fit_Y = y
        self._fit_X = self.nbrs.fit(X, y)
        return OneVsOneClassifier.fit(self, X, y)

    def decision_function(self, X):
        neighbors = self.nbrs.kneighbors(X, self.n_neighbors, return_distance=False)
        estimators_set = set([])
        estimators = list()
        for neighbor in neighbors[0]:
            estimators_set.add(self._fit_Y[neighbor])

        n_classes = int(((1 + sqrt(4 * 2 * len(self.estimators_) + 1)) / 2).real)  # n*(n-1)/2 binary classificators

        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                if i in estimators_set or j in estimators_set:
                    estimators.append(self.estimators_[k])
                k += 1

        predictions = np.vstack([est.predict(X) for est in estimators]).T
        confidences = np.vstack([_predict_binary(est, X) for est in estimators]).T
        return _ovr_decision_function(predictions, confidences, len(estimators_set))
