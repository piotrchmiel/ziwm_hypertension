from copy import deepcopy

from src.sklearn_wrapper import SklearnWrapper, TwoLayerClassifier
from src.utils.tools import load_object


class ClassifierFactory(object):

    @classmethod
    def make_multiclass_classifier(cls, classifier_class, X, y, *args, **kwargs):
        classifier = SklearnWrapper(classifier_class(*args, **kwargs))
        classifier.train(X, y)
        return classifier

    @classmethod
    def make_two_layer_classifier(cls, classifier_class, X, y, *args, **kwargs):
        classifier = SklearnWrapper(classifier_class(*args, **kwargs))
        two_layer_classifier = TwoLayerClassifier(deepcopy(classifier), deepcopy(classifier),
                                                  'essent')
        two_layer_classifier.train(X, y)
        return two_layer_classifier

    @staticmethod
    def make_classifier_from_file(filename):
        return load_object(filename)


