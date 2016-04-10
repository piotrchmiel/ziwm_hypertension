from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import NearestNeighbors


class DynamicOneVsOneClassifier(OneVsOneClassifier):

    def __init__(self, estimator, n_jobs=-1, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None):
        OneVsOneClassifier.__init__(self, estimator, n_jobs)
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius= radius, algorithm=algorithm, leaf_size=leaf_size,
                                     metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
        self.n_neighbors = n_neighbors
        self._fit_X = None

    def fit(self, X, y):
        self._fit_X = self.nbrs.fit(X, y)
        return OneVsOneClassifier.fit(self, X, y)

    def decision_function(self, X):
        neighbors = self.nbrs.kneighbors(X, self.n_neighbors, return_distance=False)
        #neighbors zawiera numery wierszy w self._fit_x trzbea wziąć ich klasy spakować do seta
        return OneVsOneClassifier.decision_function(self, X)