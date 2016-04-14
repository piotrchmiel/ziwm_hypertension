from copy import deepcopy
from os import path

from joblib import delayed

from src.classifiers.sklearn_wrapper import SklearnWrapper, TwoLayerClassifier
from src.settings import METHODS, CLASSIFIERS_DIR, ENSEMBLE
from src.utils.tools import load_object, save_object


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


def get_creator(method, train_set, train_labels):
    return (delayed(create_classifiers)(method, classifier_name, algorithm_info, train_set, train_labels)
            for classifier_name, algorithm_info in METHODS[method].items())


def create_classifiers(method, classifier_name, algorithm_info, train_set, train_labels):
    multiclass_classifier = ClassifierFactory.make_multiclass_classifier(algorithm_info[0], train_set, train_labels,
                                                                         **algorithm_info[1])

    save_object(path.join(CLASSIFIERS_DIR, method, "".join(['multiclass_', classifier_name, '.pickle'])),
                multiclass_classifier)

    if method == ENSEMBLE:
        two_layer_classifier = ClassifierFactory.make_two_layer_classifier(algorithm_info[0], train_set, train_labels,
                                                                           **algorithm_info[1])

        save_object(path.join(CLASSIFIERS_DIR, ENSEMBLE,  "".join(['two_layer_', classifier_name, '.pickle'])),
                    two_layer_classifier)
