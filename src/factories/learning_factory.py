from enum import Enum
from os import path

from sklearn.datasets import fetch_mldata

from src.settings import TRAINING_SET_DIR, HYPER_SHEET_NAME, \
    HYPER_TRAINING_SET, ISOLET_TRAINING_SET, AUSLAN_TRAINING_SET, \
    ABALONE_TRAINING_SET, LETTER_TRAINING_SET, KDDCUP_TRAINING_SET
from src.utils.csv_parser import CsvParser
from src.utils.excel import ExcelParser

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
    def get_full_learning_set_with_labels(data_source):
        if data_source == LearningSetFactory.DataSource.hypertension:
            return LearningSetFactory.get_full_excel_learning_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]), HYPER_SHEET_NAME, 'wy')
        elif data_source == LearningSetFactory.DataSource.mnist:
            return LearningSetFactory.get_full_mnist_training_set_and_labels()
        else:
            return LearningSetFactory.get_full_csv_learning_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]))



    @staticmethod
    def get_full_excel_learning_set_and_labels(learning_set_path, sheet_name, classname_column):
        parser = ExcelParser(learning_set_path, sheet_name)
        learning_set = [row for row in parser.get_rows()]
        learning_labels = LearningSetFactory._extract_labels(learning_set, classname_column)
        return learning_set, learning_labels


    @staticmethod
    def get_full_csv_learning_set_and_labels(learning_set_path):
        parser = CsvParser(learning_set_path)
        learning_set = [row for row in parser.get_rows()]
        learning_labels = LearningSetFactory._extract_labels(learning_set, parser.get_keys()[-1])
        return learning_set, learning_labels

    @staticmethod
    def get_full_mnist_training_set_and_labels():
        print("Getting mnist..")
        mnist = fetch_mldata('MNIST original')
        print("Done")
        return mnist.data, mnist.target

    @staticmethod
    def _extract_labels(learning_set, classname_column):
        learning_labels = []
        for feature_set in learning_set:
            learning_labels.append(feature_set[classname_column])
            del feature_set[classname_column]
        return learning_labels


