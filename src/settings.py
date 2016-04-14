from os import path

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier

from src.dynamic_OvO import DynamicOneVsOneClassifier
BASE_DIR = path.dirname(path.dirname(__file__))

TRAINING_SET_DIR = path.join(BASE_DIR, "Training set")
TRAINING_SET_FILENAME = "hyper.xlsx"
SHEET_NAME = "hyper"
CLASSIFIERS_DIR = path.join(BASE_DIR, "Classifiers")
ALGORITHMS = {'ada_boost_classifier': AdaBoostClassifier, 'random_forest': RandomForestClassifier,
              'bagging_classifier': BaggingClassifier, 'one_vs_one': OneVsOneClassifier,
              'dynamic_one_vs_one': DynamicOneVsOneClassifier}
METHODS = {'ensemble': ['ada_boost_classifier', 'random_forest', 'bagging_classifier'],
           'multiclass': ['one_vs_one', 'dynamic_one_vs_one']}
