from collections import defaultdict
from itertools import chain
from os import path
from random import shuffle

from src.settings import TRAINING_SET_DIR, TRAINING_SET_FILENAME, SHEET_NAME
from src.utils.excel import ExcelParser


class LearningSetFactory(object):

    @staticmethod
    def get_learning_sets_and_labels(percent_of_train_set):

        parser = ExcelParser(path.join(TRAINING_SET_DIR, TRAINING_SET_FILENAME), SHEET_NAME)

        feature_dict = defaultdict(list)
        for row in parser.get_rows():
            feature_dict[row['wy']].append(row)

        train_set = []
        test_set = []

        for value in feature_dict.values():
            shuffle(value)
            slice_point = int(percent_of_train_set * len(value))
            train_set.extend(value[:slice_point])
            test_set.extend(value[slice_point:])

        train_labels = [feature_set['wy'] for feature_set in train_set]
        test_labels = [feature_set['wy'] for feature_set in test_set]

        for feature_set in chain(train_set, test_set):
            del feature_set['wy']

        return train_set, train_labels, test_set, test_labels