from enum import Enum
from os import path

import numpy as np
from sklearn.datasets import fetch_mldata

from src.settings import TRAINING_SET_DIR, HYPER_SHEET_NAME, HYPER_TRAINING_SET, ISOLET_TRAINING_SET, \
    AUSLAN_TRAINING_SET, KDDCUP_TRAINING_SET
from src.utils.csv_parser import CsvParser
from src.utils.excel import ExcelParser

TRAINING_SET_MAP = {
    0: HYPER_TRAINING_SET,
    1: "MNIST original",
    2: ISOLET_TRAINING_SET,
    3: AUSLAN_TRAINING_SET,
    # 4: "abalone",
    # 5: "letter",
    6: KDDCUP_TRAINING_SET,
    7: "datasets-UCI vowel",
    8: "yeast",
    9: "uci-20070111 ecoli",
    10: "segment",
    11: "shuttle",
    12: "satimage",
    13: ["uci-20070111 solar-flare_1", "uci-20070111 solar-flare_2"],
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
        vowel = 7
        yeast = 8
        ecoli = 9
        segment = 10
        shuttle = 11
        satimage = 12
        flare = 13

    @staticmethod
    def get_full_learning_set_with_labels(data_source):
        if data_source == LearningSetFactory.DataSource.hypertension:
            return LearningSetFactory.get_full_excel_learning_set_and_labels(path.join(
                TRAINING_SET_DIR, TRAINING_SET_MAP[data_source.value]), HYPER_SHEET_NAME, 'wy')
        elif data_source in [LearningSetFactory.DataSource.mnist] + \
                list(LearningSetFactory.DataSource)[LearningSetFactory.DataSource.vowel.value:]:
            return LearningSetFactory.get_full_mldata_training_set_and_labels(TRAINING_SET_MAP[data_source.value])
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
    def get_full_mldata_training_set_and_labels(repository_name):
        if type(repository_name) == list:
            data, target = None, None
            for repository in repository_name:
                dataset = LearningSetFactory.fetch_mldata_dataset(repository)
                if data is None and target is None:
                    data, target = dataset
                else:
                    data = np.concatenate([data, dataset[0]])
                    target = np.concatenate([target, dataset[1]])
            return data, target
        else:
            return fetch_mldata(repository_name)

    @staticmethod
    def fetch_mldata_dataset(repository_name):
        print("Getting,", repository_name, "...")
        learning_set = fetch_mldata(repository_name)
        print("Done.")
        return learning_set.data, learning_set.target

    @staticmethod
    def _extract_labels(learning_set, classname_column):
        learning_labels = []
        for feature_set in learning_set:
            learning_labels.append(feature_set[classname_column])
            del feature_set[classname_column]
        return learning_labels
