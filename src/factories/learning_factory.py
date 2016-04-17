from collections import defaultdict
from enum import Enum
from itertools import chain
from os import path
from random import shuffle

from mnist import MNIST

from src.settings import TRAINING_SET_DIR, HYPER_SHEET_NAME, \
    HYPER_TRAINING_SET, ISOLET_TRAINING_SET, AUSLAN_TRAINING_SET, \
    ABALONE_TRAINING_SET, LETTER_TRAINING_SET, KDDCUP_TRAINING_SET
from src.utils.excel import ExcelParser
from src.utils.csv_parser import CsvParser

TRAINING_SET_MAP = {
    0: HYPER_TRAINING_SET,
    1: "",
    2: ISOLET_TRAINING_SET,
    3: AUSLAN_TRAINING_SET,
    4: ABALONE_TRAINING_SET,
    5: LETTER_TRAINING_SET,
    6: KDDCUP_TRAINING_SET
}

class LearningSetFactory(object):

    class DataSource(Enum):
        hypertension = 0
        mnist = 1
        isolet = 2
        auslan = 3
        abalone = 4
        letter = 5
        kddcup = 6

    @staticmethod
    def get_learning_sets_and_labels(percent_of_train_set, data_source):
        if data_source == LearningSetFactory.DataSource.hypertension:
            return LearningSetFactory.get_excel_training_sets_and_labels(percent_of_train_set, path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]), HYPER_SHEET_NAME, 'wy')
        elif data_source == LearningSetFactory.DataSource.mnist:
            return LearningSetFactory.get_mnist_training_set_and_labels(percent_of_train_set, TRAINING_SET_DIR)
        else:
            return LearningSetFactory.get_csv_training_sets_and_labels(percent_of_train_set, path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]))

    @staticmethod
    def get_full_learning_set_with_labels(data_source):
        if data_source == LearningSetFactory.DataSource.hypertension:
            return LearningSetFactory.get_full_excel_learning_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]), HYPER_SHEET_NAME, 'wy')
        elif data_source == LearningSetFactory.DataSource.mnist:
            return LearningSetFactory.get_full_mnist_training_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]))
        else:
            return LearningSetFactory.get_full_csv_learning_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]))

    @staticmethod
    def get_excel_training_sets_and_labels(percent_of_train_set, learning_set_path, sheet_name, classname_column):
        parser = ExcelParser(learning_set_path, sheet_name)
        return LearningSetFactory._get_training_sets_and_labels(percent_of_train_set,
                                                                parser.get_rows(), classname_column)

    @staticmethod
    def get_full_excel_learning_set_and_labels(learning_set_path, sheet_name, classname_column):
        parser = ExcelParser(learning_set_path, sheet_name)
        learning_set = [row for row in parser.get_rows()]
        learning_labels = LearningSetFactory._extract_labels(learning_set, classname_column)
        return learning_set, learning_labels

    @staticmethod
    def get_csv_training_sets_and_labels(percent_of_train_set, learning_set_path):
        parser = CsvParser(learning_set_path)
        return LearningSetFactory._get_training_sets_and_labels(percent_of_train_set, parser.get_rows(),
                                                                parser.get_keys()[-1])

    @staticmethod
    def get_full_csv_learning_set_and_labels(learning_set_path):
        parser = CsvParser(learning_set_path)
        learning_set = [row for row in parser.get_rows()]
        learning_labels = LearningSetFactory._extract_labels(learning_set, parser.get_keys()[-1])
        return learning_set, learning_labels

    @staticmethod
    def get_mnist_training_set_and_labels(percent_of_train_set, learning_set_path):
        images, labels = LearningSetFactory.get_full_mnist_training_set_and_labels(learning_set_path)
        records = [data + [label] for data, label in zip(images, labels)]
        return LearningSetFactory._get_training_sets_and_labels(percent_of_train_set, records, 'Class')

    @staticmethod
    def get_full_mnist_training_set_and_labels(learning_set_path):
        mndata = MNIST(learning_set_path)
        training_images, training_labels = mndata.load_training()
        testing_images, testing_labels = mndata.load_testing()
        dataset = training_images + testing_images
        labels = training_labels + testing_labels
        keys = ['Atr-%d' % i for i in range(1, len(dataset[0]) + 1)]
        keys.append('Class')
        dataset_with_attributes = [dict(zip(keys, record)) for record in dataset]
        return dataset_with_attributes, labels

    @staticmethod
    def _extract_labels(learning_set, classname_column):
        learning_labels = []
        for feature_set in learning_set:
            learning_labels.append(feature_set[classname_column])
            del feature_set[classname_column]
        return learning_labels

    @staticmethod
    def _get_training_sets_and_labels(percent_of_train_set, records, classname_column):
        feature_dict = defaultdict(list)
        for row in records:
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
