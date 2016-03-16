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


class TwoLayerClassifier(object):

    def __init__(self, first_layer, second_layer, main_class):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.main_class = main_class

    def __repr__(self):
        return "<TwoLayerClassifier>"

    def train(self, X, y):
        first_layer_labels = []
        for label in y:
            if label != self.main_class:
                first_layer_labels.append("secondary_type")
            else:
                first_layer_labels.append(label)

        self.first_layer.train(X, first_layer_labels)

        second_layer_data = []
        second_layer_labels = []
        for feature_set, label in zip(X,y):
            if label != self.main_class:
                second_layer_data.append(feature_set)
                second_layer_labels.append(label)

        self.second_layer.train(second_layer_data, second_layer_labels)

    def classify(self, feature_set):
        if self.first_layer.classify(feature_set) != self.main_class:
            return self.second_layer.classify(feature_set)
        return self.main_class

    def accuracy(self, X, y):
        pred = []
        for feature_set in X:
            pred.append(self.classify(feature_set))
        return accuracy_score(y, pred)