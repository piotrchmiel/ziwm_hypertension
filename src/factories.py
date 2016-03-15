from collections import defaultdict
from itertools import chain
from os import path
from random import shuffle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from src.settings import TRAINING_SET_DIR, TRAINING_SET_FILENAME, SHEET_NAME
from src.sklearn_wrapper import SklearnWrapper
from src.utils import ExcelParser, load_object


class MultiClassClassifierFactory:

    @classmethod
    def make_default_classifier(self, Class, X, y):
        classifier = SklearnWrapper(Class())
        classifier.train(X,y)
        return classifier

    @staticmethod
    def make_ada_boost_classifier(X, y):
        classifier = SklearnWrapper(AdaBoostClassifier(SVC(), algorithm='SAMME'))
        classifier.train(X,y)
        return classifier

    @staticmethod
    def make_classifier_from_file(filename):
        return load_object(filename)

class LearningSetFactory:

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
