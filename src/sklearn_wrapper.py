from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Imputer


class SklearnWrapper(object):

    def __init__(self, classifier, dtype=float, sparse=True, sort=False):

        self._classifier = classifier
        self._encoder = LabelEncoder()
        self._vectorizer = DictVectorizer(dtype=dtype, sparse=sparse, sort=sort)
        self._imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self._pipeline = Pipeline([('_vectorizer', self._vectorizer), ('_imputer', self._imputer)])

    def __repr__(self):
        return "<SklearnWrapper(%r)>" % self._classifier

    def train(self, X, y):

        X = self._pipeline.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._classifier.fit(X, y)

    def classify(self, feature_set):
        feature_set = self._pipeline.transform(feature_set)
        return self._encoder.classes_[self._classifier.predict(feature_set)][0]

    def get_classes(self):
        return self._encoder.classes_

    def accuracy(self, X, y):
        pred = []
        for feature_set in X:
            pred.append(self.classify(feature_set))
        return accuracy_score(y, pred)