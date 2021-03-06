from itertools import islice
from enum import Enum, unique
from os import path

import numpy as np
from sklearn.datasets import fetch_mldata

from src.settings import TRAINING_SET_DIR, HYPER_SHEET_NAME, HYPER_TRAINING_SET, \
    ISOLET_TRAINING_SET, AUSLAN_TRAINING_SET, KDDCUP_TRAINING_SET, STUDENT_ALCOHOL_TRAINING_SET, \
    ADULT_TRAINING_SET, WINE_QUALITY_TRAINING_SET, YEAST_TRAINING_SET
from src.utils.csv_parser import CsvParser
from src.utils.excel import ExcelParser

TRAINING_SET_MAP = {
    0: {'name': HYPER_TRAINING_SET},
    1: {'name': ISOLET_TRAINING_SET},
    2: {'name': AUSLAN_TRAINING_SET},
    3: {'name': KDDCUP_TRAINING_SET},
    4: {'name': 'abalone', 'kwargs': {}},
    5: {'name': 'letter', 'kwargs': {}},
    6: {'name': 'MNIST original', 'kwargs': {}},
    7: {'name': 'datasets-UCI vowel', 'kwargs': {'target_name': 'Class', 'data_name': 'double0'}},
    8: {'name': 'uci-20070111 ecoli', 'kwargs': {'target_name': 'class', 'data_name': 'double0'}},
    9: {'name': 'segment', 'kwargs': {}},
    10: {'name': 'shuttle', 'kwargs': {}},
    11: {'name': 'satimage', 'kwargs': {}},
    12: [{'name': "uci-20070111 solar-flare_1",
          'kwargs': {'target_name': 'class', 'data_name': 'int0'}},
         {'name': "uci-20070111 solar-flare_2",
          'kwargs': {'target_name': 'class', 'data_name': 'int0'}}],
    13: {'name': YEAST_TRAINING_SET},
    14: {'name': STUDENT_ALCOHOL_TRAINING_SET},
    15: {'name': ADULT_TRAINING_SET},
    16: {'name': WINE_QUALITY_TRAINING_SET},
}


class LearningSetFactory(object):

    @unique
    class DataSource(Enum):
        hypertension = 0
        isolet = 1
        auslan = 2
        kddcup = 3
        abalone = 4
        letter = 5
        mnist = 6
        vowel = 7
        ecoli = 8
        segment = 9
        shuttle = 10
        satimage = 11
        flare = 12
        yeast = 13
        student_alcohol_consumption = 14
        adult = 15
        wine_quality = 16

    @staticmethod
    def get_full_learning_set_with_labels(data_source):
        if data_source == LearningSetFactory.DataSource.hypertension:
            return LearningSetFactory.get_full_excel_learning_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]['name']), HYPER_SHEET_NAME, 'wy')
        elif data_source in list(LearningSetFactory.DataSource)\
            [LearningSetFactory.DataSource.abalone.value:LearningSetFactory.DataSource.flare.value + 1]:
            return LearningSetFactory.get_full_mldata_training_set_and_labels(
                TRAINING_SET_MAP[data_source.value])
        else:
            return LearningSetFactory.get_full_csv_learning_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]['name']))

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
    def get_full_mldata_training_set_and_labels(repository):
        if isinstance(repository, list) and len(repository) is not 0:
            data, target = LearningSetFactory.fetch_mldata_data_set(repository[0])
            for repository_info in islice(repository, 1, None):
                data_set = LearningSetFactory.fetch_mldata_data_set(repository_info)
                data = np.concatenate([data, data_set[0]])
                target = np.concatenate([target, data_set[1]])
            return data, target
        else:
            return LearningSetFactory.fetch_mldata_data_set(repository)

    @staticmethod
    def fetch_mldata_data_set(repository_info):
        learning_set = fetch_mldata(repository_info['name'], **repository_info['kwargs'])
        print("Done.")
        target = np.array([label[0] for label in learning_set.target]) if repository_info['kwargs'] \
            else learning_set.target
        return learning_set.data, target

    @staticmethod
    def _extract_labels(learning_set, classname_column):
        learning_labels = []
        for feature_set in learning_set:
            learning_labels.append(feature_set[classname_column])
            del feature_set[classname_column]
        return learning_labels
