from cmath import sqrt

import array
import numpy as np
from sklearn.multiclass import OneVsRestClassifier, _predict_binary
from sklearn.neighbors import NearestNeighbors
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted, _num_samples
import scipy.sparse as sp


class DynamicOneVsRestClassifier(OneVsRestClassifier):
    def __init__(self, estimator, n_jobs=-1, n_neighbors=18, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, threshold=0.1, metric_params=None):
        OneVsRestClassifier.__init__(self, estimator, n_jobs)
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm=algorithm,
                                     leaf_size=leaf_size, metric=metric, p=p,
                                     metric_params=metric_params, n_jobs=n_jobs)
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self._fit_y = None

    def fit(self, X, y):
        self._fit_y = y
        self.nbrs.fit(X, y)
        return OneVsRestClassifier.fit(self, X, y)

    def predict_proba(self, X):
        print('kek')
        return OneVsRestClassifier(self, X)

    def predict(self, X):
        neighbors = self.nbrs.kneighbors(X, self.n_neighbors, return_distance=False)
        neighbors_list = []

        for neighbor in neighbors[0]:
            neighbors_list.append(self._fit_y[neighbor])

        neighbors_count = len(neighbors_list)
        neighbors_set = set(neighbors_list)
        neighbors_set_tmp = neighbors_set.copy()
        for neighbor in neighbors_set_tmp:
            if not neighbors_list.count(neighbor) / neighbors_count > self.threshold:
                neighbors_set.remove(neighbor)

        check_is_fitted(self, 'estimators_')
        if (hasattr(self.estimators_[0], "decision_function") and
                is_classifier(self.estimators_[0])):
            thresh = 0
        else:
            thresh = .5

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                if not i in neighbors_set:
                    continue
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.label_binarizer_.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for i, e in enumerate(self.estimators_):
                if not i in neighbors_set:
                    continue
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)
