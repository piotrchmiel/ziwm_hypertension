from collections import defaultdict
from enum import Enum
from itertools import chain
from os import path
from random import shuffle

from src.settings import TRAINING_SET_DIR, HYPER_TRAINING_SET, HYPER_SHEET_NAME
from src.utils.excel import ExcelParser


class LearningSetFactory(object):

    class DataSource(Enum):
        hypertension = 0

    @staticmethod
    def get_learning_sets_and_labels(percent_of_train_set, data_source):
        if data_source == LearningSetFactory.DataSource.hypertension:
            return LearningSetFactory.get_excel_training_sets_and_labels(percent_of_train_set, path.join(
                TRAINING_SET_DIR,HYPER_TRAINING_SET), HYPER_SHEET_NAME, 'wy')

    @staticmethod
    def get_full_learning_set_with_labels(data_source):
        if data_source == LearningSetFactory.DataSource.hypertension:
            return LearningSetFactory.get_full_excel_learning_set_with_labels(path.join(
                TRAINING_SET_DIR, HYPER_TRAINING_SET), HYPER_SHEET_NAME, 'wy')

    @staticmethod
    def get_excel_training_sets_and_labels(percent_of_train_set, learning_set_path, sheet_name, classname_column):
        parser = ExcelParser(learning_set_path, sheet_name)

        feature_dict = defaultdict(list)
        for row in parser.get_rows():
            feature_dict[row[classname_column]].append(row)

        train_set = []
        test_set = []

        for value in feature_dict.values():
            shuffle(value)
            slice_point = int(percent_of_train_set * len(value))
            train_set.extend(value[:slice_point])
            test_set.extend(value[slice_point:])

        train_labels = [feature_set[classname_column] for feature_set in train_set]
        test_labels = [feature_set[classname_column] for feature_set in test_set]

        for feature_set in chain(train_set, test_set):
            del feature_set[classname_column]

        return train_set, train_labels, test_set, test_labels

    @staticmethod
    def get_full_excel_learning_set_with_labels(learning_set_path, sheet_name, classname_column):
        parser = ExcelParser(learning_set_path, sheet_name)
        learning_set = [row for row in parser.get_rows()]
        learning_labels = []
        for feature_set in learning_labels:
            learning_labels.append(feature_set[classname_column])
            del feature_set[classname_column]
        return learning_set, learning_labels
