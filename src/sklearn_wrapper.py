from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


class SklearnWrapper(object):

    def __init__(self, classifier, dtype=int, sparse=True, sort=False):

        self._classifier = classifier
        self._encoder = LabelEncoder()
        self._vectoricer = DictVectorizer(dtype=dtype, sparse=sparse, sort=sort)

    def __repr__(self):
        return "<SklearnWrapper(%r)>" % self._classifier

    def train(self, X, y):

        X = self._vectoricer.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._classifier.fit(X, y)

    def classify(self, feature_set):
        feature_set = self._vectoricer.transform(feature_set)
        return self._encoder.classes_[self._classifier.predict(feature_set)][0]
