from collections import defaultdict
from copy import deepcopy
from itertools import chain
from os import path
from random import shuffle

from src.settings import TRAINING_SET_DIR, TRAINING_SET_FILENAME, SHEET_NAME
from src.sklearn_wrapper import SklearnWrapper, TwoLayerClassifier
from src.utils import ExcelParser, load_object


class MulticlassClassifierFactory(object):

    @classmethod
    def make_default_classifier(self, Class, X, y, *args, **kwargs):
        classifier = SklearnWrapper(Class(*args, **kwargs))
        classifier.train(X,y)
        return classifier

    @classmethod
    def make_default_two_layer_classifier(self, Class, X, y, *args, **kwargs):
        classifier = SklearnWrapper(Class(*args, **kwargs))
        two_layer_classifier = TwoLayerClassifier(deepcopy(classifier), deepcopy(classifier), 'essent')
        two_layer_classifier.train(X,y)
        return two_layer_classifier

    @staticmethod
    def make_classifier_from_file(filename):
        return load_object(filename)


class LearningSetFactory(object):

    @staticmethod
    def get_learning_sets_and_labels(percent_of_train_set):

        parser = ExcelParser(path.join(TRAINING_SET_DIR, TRAINING_SET_FILENAME), SHEET_NAME)

        feature_dict = defaultdict(list)
        for row in parser.get_rows():
            feature_dict[row['wy']].append(row)

        train_set = []
        test_set = []

        for key, value in feature_dict.items():
            shuffle(value)
            slice_point = int(percent_of_train_set * len(value))
            train_set.extend(value[:slice_point])
            test_set.extend(value[slice_point:])

        train_labels = [feature_set['wy'] for feature_set in train_set]
        test_labels = [feature_set['wy'] for feature_set in test_set]

        for feature_set in chain(train_set, test_set):
            del feature_set['wy']

        return train_set, train_labels, test_set, test_labels
