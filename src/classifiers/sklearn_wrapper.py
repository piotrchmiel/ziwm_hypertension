import numpy
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Imputer


class SklearnWrapper(object):

    def __init__(self, classifier, dtype=float, sparse=True, sort=False):

        self._classifier = classifier
        self._encoder = LabelEncoder()
        self._vectorizer = DictVectorizer(dtype=dtype, sparse=sparse, sort=sort)
        self._imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self._clf_pipeline = make_pipeline(self._vectorizer, self._imputer, self._classifier)

    def __repr__(self):
        return "<SklearnWrapper(%r)>" % self._clf_pipeline

    def train(self, X, y):
        if isinstance(X, numpy.core.memmap) or isinstance(X, numpy.ndarray):
            self._clf_pipeline = self._classifier
        else:
            y = self._encoder.fit_transform(y)
        self._clf_pipeline.fit(X, y)

    def classify(self, feature_set):
        return self._encoder.classes_[self._clf_pipeline.predict(feature_set)][0]

    def get_classes(self):
        return self._encoder.classes_

    def accuracy(self, X, y):
        if isinstance(X, numpy.core.memmap) or isinstance(X, numpy.ndarray):
            pred = [self.classify(feature_set) for feature_set in X]
            return accuracy_score(y, pred) * 100
        else:
            return self._clf_pipeline.predict.score(X, y)

    def get_classifier(self):
        return self._clf_pipeline

    def fit_encoder(self, y):
        self._encoder.fit_transform(y)


class TwoLayerClassifier(object):

    def __init__(self, first_layer, second_layer, main_class):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.main_class = main_class

    def __repr__(self):
        return "<TwoLayerClassifier>"

    def train(self, X, y):
        X = list(X)
        first_layer_labels = []
        for label in y:
            if label != self.main_class:
                first_layer_labels.append("secondary_type")
            else:
                first_layer_labels.append(label)

        second_layer_data = []
        second_layer_labels = []
        for feature_set, label in zip(X, y):
            if label != self.main_class:
                second_layer_data.append(feature_set)
                second_layer_labels.append(label)

        self.first_layer.train(X, first_layer_labels)
        self.second_layer.train(second_layer_data, second_layer_labels)

    def classify(self, feature_set):
        if self.first_layer.classify(feature_set) != self.main_class:
            return self.second_layer.classify(feature_set)
        return self.main_class

    def accuracy(self, X, y):
        pred = [self.classify(feature_set) for feature_set in X]
        return accuracy_score(y, pred) * 100
